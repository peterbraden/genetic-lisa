"""Fit translucent ellipses to an image; write the best SVG.

fitness = MSE(hard render, source) + lam * N
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import density
from .loss import photometric
from .model import Model
from .render import FIELDS, composite_pairs, hard_pairs, inv_sigmoid, mse, natural, reindex, render
from .svg import to_svg


@dataclass
class Config:
    image: str
    out: str = "out"
    iters: int = 2500
    lam: float = 1e-5               # MSE an ellipse must save to be kept
    max_size: int | None = None     # downscale the source so its longest side is this
    # Progressive resolution: start at the coarsest halving of the image whose
    # longest side is at least min_size (0 disables), and expand one level at a
    # density step once fitness improves by less than expand_tol (relative)
    # over an interval; from expand_by of the run, expansion is forced.
    min_size: int = 0
    expand_tol: float = 0.01
    expand_by: float = 0.5
    device: str = "cpu"             # with culling, MPS offers no gain at this scale
    seed: int = 0
    # Defaults from an Optuna search over the corpus tuning split (scripts/tune.py,
    # trial 78, rounded); see README.
    # Rendering: soft edge width (pixels) annealed from tau_start to tau_end.
    tau_start: float = 1.0
    tau_end: float = 0.31
    anneal_until: float = 0.76      # fraction of iters
    ramp_from: float = 0.78         # then train on the anti-aliased (SVG) edge
    # Loss.
    ssim_weight: float = 0.18
    opacity_penalty: float = 1.8    # per unit opacity, in multiples of lam
    # Adam learning rates. Positions are in fractions of the longest side.
    lr_mu: float = 0.003
    lr_scale: float = 0.009         # on log semi-axes
    lr_theta: float = 0.005
    lr_col: float = 0.04            # on colour logits
    lr_op: float = 0.034            # on opacity logits
    lr_bg: float = 0.01
    mu_lr_decay: float = 0.01       # final position learning rate, relative
    # Density control.
    init_shapes: int = 8
    densify_every: int = 200
    densify_from: int = 100
    densify_until: float = 0.84     # fraction of iters
    grad_ratio: float = 9.3         # clone/split above this times the median positional gradient
    size_threshold: float = 0.015   # fraction of longest side; larger shapes split, smaller clone
    clone_nudge: float = 0.5        # clone offset, in minor semi-axes
    split_offset: float = 0.45      # children's offset, in major semi-axes
    split_major: float = 0.55       # children's major axis, relative
    split_minor: float = 0.85       # children's minor axis, relative
    max_growth: float = 0.19        # fraction of N added by clone/split per step
    max_prune: float = 0.19         # fraction of N removed per step
    min_opacity: float = 0.01
    spawn_k: int = 14
    spawn_max: float = 0.15         # largest spawn box half-width, fraction of longest side
    spawn_opacity: float = 0.54
    glaze_k: int = 0                # translucent layers per step (fitted opacity); 0 disables
    glaze_max: float = 0.5          # largest glaze box half-width, fraction of longest side
    glaze_min_opacity: float = 0.05
    relocate_k: int = 4             # relocations per step; 0 disables
    relocate_margin: float = 1.0    # site saving must exceed this times the shape's value
    reorder_moves: int = 3          # moves per step; 0 disables
    # Logging.
    log_every: int = 100
    snapshot_every: int = 250       # 0 disables
    quiet: bool = False


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def load_image(path: str, max_size: int | None) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    if max_size and max(im.size) > max_size:
        f = max_size / max(im.size)
        im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1).contiguous()


def save_png(img: torch.Tensor, path: Path):
    arr = (img.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def tau_at(cfg: Config, it: int) -> float:
    t = min(1.0, it / max(1, cfg.anneal_until * cfg.iters))
    return math.exp((1 - t) * math.log(cfg.tau_start) + t * math.log(cfg.tau_end))


def empty_params(device) -> dict[str, torch.Tensor]:
    shapes = {"mu": (0, 2), "log_s": (0, 2), "theta": (0,), "col": (0, 3), "op": (0,)}
    return {k: torch.zeros(shapes[k], device=device) for k in FIELDS}


def spawn_half_widths(largest: float) -> list[int]:
    """Powers of two from 1 up to `largest` pixels."""
    out = [1]
    while out[-1] * 2 <= largest:
        out.append(out[-1] * 2)
    return out


def fitness(m: float, n: int, lam: float) -> float:
    return m + lam * n


def pyramid(cfg: Config, dev) -> list[torch.Tensor]:
    """Targets from coarsest to full size, halving the longest side."""
    full = load_image(cfg.image, cfg.max_size).to(dev)
    sizes, size = [], max(full.shape[1:])
    while cfg.min_size and round(size / 2) >= cfg.min_size:
        size = round(size / 2)
        sizes.append(size)
    return [load_image(cfg.image, s).to(dev) for s in reversed(sizes)] + [full]


def in_frame(p: dict[str, torch.Tensor], f: float) -> dict[str, torch.Tensor]:
    """Parameters with the image plane scaled by `f`."""
    return {**p, "mu": p["mu"] * f, "log_s": p["log_s"] + math.log(f)}


def train(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    dev = pick_device(cfg.device)
    levels = pyramid(cfg, dev)
    level = 0
    full_w, full_h = levels[-1].shape[2], levels[-1].shape[1]

    def at_level(target):
        _, h, w = target.shape
        size = max(h, w)
        min_gain = cfg.lam * 3 * h * w  # lam in summed-squared-error units
        half_widths = spawn_half_widths(cfg.spawn_max * size)
        glaze_widths = [x for x in spawn_half_widths(cfg.glaze_max * size) if x > half_widths[-1]]
        return target, h, w, size, min_gain, half_widths, glaze_widths

    target, h, w, size, min_gain, half_widths, glaze_widths = at_level(levels[0])
    out = Path(cfg.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    say = (lambda *a: None) if cfg.quiet else print
    split = dict(nudge=cfg.clone_nudge, offset=cfg.split_offset, major=cfg.split_major, minor=cfg.split_minor)

    lr = {"mu": cfg.lr_mu * size, "log_s": cfg.lr_scale, "theta": cfg.lr_theta,
          "col": cfg.lr_col, "op": cfg.lr_op, "bg": cfg.lr_bg}
    model = Model(empty_params(dev), inv_sigmoid(target.mean(dim=(1, 2))), lr)
    density.spawn(model, render(model.p, model.bg, h, w, cfg.tau_start).detach(), target,
                  half_widths, cfg.init_shapes, min_gain, cfg.spawn_opacity)
    log = []
    prev_fit = None
    start = time.perf_counter()

    def expand():
        nonlocal level, target, h, w, size, min_gain, half_widths, glaze_widths
        old = size
        level += 1
        target, h, w, size, min_gain, half_widths, glaze_widths = at_level(levels[level])
        model.rescale(size / old)
        lr["mu"] = cfg.lr_mu * size
    say(f"# {cfg.image} {w}x{h} on {dev}, lam={cfg.lam}")

    for it in range(cfg.iters + 1):
        tau = tau_at(cfg, it)
        ramp = it >= cfg.ramp_from * cfg.iters
        model.set_lr("mu", lr["mu"] * cfg.mu_lr_decay ** (it / cfg.iters))
        img = render(model.p, model.bg, h, w, tau, ramp)
        loss = photometric(img, target, cfg.ssim_weight)
        loss = loss + cfg.opacity_penalty * cfg.lam * torch.sigmoid(model.p["op"]).sum()
        model.opt.zero_grad(set_to_none=True)
        loss.backward()
        model.accumulate()
        model.opt.step()
        with torch.no_grad():
            model.p["log_s"].clamp_(math.log(0.3), math.log(2 * max(h, w)))

        event = ""
        trained_on = target
        densify = cfg.densify_from <= it < cfg.densify_until * cfg.iters and it % cfg.densify_every == 0
        if densify and level < len(levels) - 1:
            fit = fitness(float(mse(img.detach(), target)), model.n, cfg.lam)
            stalled = prev_fit is not None and prev_fit - fit < cfg.expand_tol * prev_fit
            prev_fit = fit
            if stalled or it >= cfg.expand_by * cfg.iters:
                expand()
                prev_fit, densify = None, False
                event = f" expand to {w}x{h}"
        if densify:
            n = model.n
            pruned = density.prune(model, target, tau, cfg.lam, cfg.min_opacity,
                                   math.ceil(cfg.max_prune * n))
            with torch.no_grad():
                cur = render(model.p, model.bg, h, w, tau)
            relocated = density.relocate(model, cur, target, tau, half_widths,
                                         cfg.relocate_k, cfg.relocate_margin, cfg.spawn_opacity)
            moved = density.reorder(model, target, tau, cfg.reorder_moves, 0.1 * cfg.lam)
            g = model.mean_grad()
            gq = torch.quantile(g, torch.tensor([0.5, 0.9], device=dev)).tolist() if model.n > 1 else [0, 0]
            threshold = cfg.grad_ratio * gq[0]
            clones, splits = density.clone_split(model, threshold, cfg.size_threshold * size,
                                                 math.ceil(cfg.max_growth * model.n) + 1, **split)
            with torch.no_grad():
                cur = render(model.p, model.bg, h, w, tau)
            spawned = density.spawn(model, cur, target, half_widths, cfg.spawn_k, min_gain, cfg.spawn_opacity)
            glazed = 0
            if cfg.glaze_k and glaze_widths:
                with torch.no_grad():
                    cur = render(model.p, model.bg, h, w, tau)
                glazed = density.glaze(model, cur, target, glaze_widths, cfg.glaze_k, min_gain,
                                       cfg.glaze_min_opacity)
            model.reset_stats()
            event += (f" -{pruned} clone+{clones} split+{splits} spawn+{spawned} glaze+{glazed}"
                     f" reloc={relocated} moved={moved}"
                     f" grad p50={gq[0]:.1e} p90={gq[1]:.1e}")

        if it % cfg.log_every == 0 or event:
            m = float(mse(img.detach(), trained_on))
            row = {"it": it, "n": model.n, "mse": m, "fitness": fitness(m, model.n, cfg.lam), "size": [w, h],
                   "tau": tau, "ramp": ramp, "t": time.perf_counter() - start, "event": event.strip()}
            log.append(row)
            say(f"{it:5d} n={model.n:4d} mse={m:.5f} fit={row['fitness']:.5f} tau={tau:.2f}"
                  f"{' ramp' if ramp else ''} {row['t']:.0f}s{event}")
        if cfg.snapshot_every and it % cfg.snapshot_every == 0:
            frame = in_frame(model.detached(), max(full_w, full_h) / size)
            (out / "frames" / f"{it:06d}.svg").write_text(to_svg(frame, model.bg, full_w, full_h))

    while level < len(levels) - 1:
        expand()
    # Final: hard-edged evaluation and exact greedy pruning against the fitness.
    p = model.detached()
    pairs = hard_pairs(p, h, w)
    colour, bg = torch.sigmoid(p["col"]), torch.sigmoid(model.bg.detach())
    before_n, before_mse = model.n, float(mse(composite_pairs(pairs, colour, bg), target))
    keep = density.greedy_prune(pairs, colour, bg, target, cfg.lam)
    model.restructure(keep)
    p = model.detached()
    final = composite_pairs(reindex(pairs, keep), colour[keep], bg)
    m = float(mse(final, target))
    result = {"n": model.n, "mse": m, "fitness": fitness(m, model.n, cfg.lam),
              "before_prune": {"n": before_n, "mse": before_mse,
                               "fitness": fitness(before_mse, before_n, cfg.lam)},
              "seconds": time.perf_counter() - start}
    say(f"# hard: n={before_n} mse={before_mse:.5f} -> greedy prune n={model.n} "
          f"mse={m:.5f} fitness={result['fitness']:.5f}")

    (out / "best.svg").write_text(to_svg(p, model.bg, w, h))
    save_png(final, out / "best.png")
    save_png(target, out / "target.png")
    params = {k: v.cpu().tolist() for k, v in natural(p).items()}
    params["background"] = torch.sigmoid(model.bg.detach()).cpu().tolist()
    (out / "best.json").write_text(json.dumps(params))
    (out / "run.json").write_text(json.dumps({"config": asdict(cfg), "result": result, "log": log}, indent=1))
    return result


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    defaults = Config(image="")
    for name, value in asdict(defaults).items():
        if name == "image":
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        elif value is None:
            ap.add_argument(flag, type=int, default=None)
        else:
            ap.add_argument(flag, type=type(value), default=value)
    args = ap.parse_args(argv)
    train(Config(**vars(args)))


if __name__ == "__main__":
    main()
