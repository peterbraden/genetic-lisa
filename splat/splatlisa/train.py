"""Fit translucent ellipses to an image; write the best SVG.

fitness = MSE(hard render, source) + lam * N
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import density
from .loss import photometric
from .model import Model
from .render import FIELDS, hard_alphas, inv_sigmoid, mse, natural, render, render_hard
from .svg import to_svg


@dataclass
class Config:
    image: str
    out: str = "out"
    iters: int = 2500
    lam: float = 1e-5               # MSE an ellipse must save to be kept
    max_size: int | None = None     # downscale the source so its longest side is this
    device: str = "cpu"             # with culling, MPS offers no gain at this scale
    seed: int = 0
    # Rendering: soft edge width (pixels) annealed from tau_start to tau_end.
    tau_start: float = 2.0
    tau_end: float = 0.5
    anneal_until: float = 0.6       # fraction of iters
    ramp_from: float = 0.8          # then train on the anti-aliased (SVG) edge
    # Loss.
    ssim_weight: float = 0.2
    opacity_penalty: float = 2e-5   # per ellipse, on sigmoid(opacity)
    # Density control.
    init_shapes: int = 8
    densify_every: int = 100
    densify_from: int = 100
    densify_until: float = 0.75     # fraction of iters
    grad_threshold: float = 2e-5
    size_threshold: float = 6.0     # pixels; larger shapes split, smaller clone
    max_growth: float = 0.1         # fraction of N added by clone/split per step
    max_prune: float = 0.1          # fraction of N removed per step
    min_opacity: float = 0.01
    spawn_k: int = 8
    relocate_k: int = 0             # relocations per step; no measured gain, so off
    relocate_margin: float = 1.0    # site saving must exceed this times the shape's value
    reorder_moves: int = 0          # moves per step; no measured gain, so off
    spawn_half_widths: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    # Logging.
    log_every: int = 100
    snapshot_every: int = 250


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


def fitness(m: float, n: int, lam: float) -> float:
    return m + lam * n


def train(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    dev = pick_device(cfg.device)
    target = load_image(cfg.image, cfg.max_size).to(dev)
    _, h, w = target.shape
    out = Path(cfg.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    min_gain = cfg.lam * 3 * h * w  # lam in summed-squared-error units

    model = Model(empty_params(dev), inv_sigmoid(target.mean(dim=(1, 2))))
    density.spawn(model, render(model.p, model.bg, h, w, cfg.tau_start).detach(), target,
                  cfg.spawn_half_widths, cfg.init_shapes, min_gain)
    log = []
    lr_mu = model.lr["mu"]
    start = time.time()
    print(f"# {cfg.image} {w}x{h} on {dev}, lam={cfg.lam}")

    for it in range(cfg.iters + 1):
        tau = tau_at(cfg, it)
        ramp = it >= cfg.ramp_from * cfg.iters
        model.set_lr("mu", lr_mu * 0.01 ** (it / cfg.iters))
        img = render(model.p, model.bg, h, w, tau, ramp)
        loss = photometric(img, target, cfg.ssim_weight)
        loss = loss + cfg.opacity_penalty * torch.sigmoid(model.p["op"]).sum()
        model.opt.zero_grad(set_to_none=True)
        loss.backward()
        model.accumulate()
        model.opt.step()
        with torch.no_grad():
            model.p["log_s"].clamp_(math.log(0.3), math.log(2 * max(h, w)))

        event = ""
        if (cfg.densify_from <= it < cfg.densify_until * cfg.iters and it % cfg.densify_every == 0):
            n = model.n
            pruned = density.prune(model, target, tau, cfg.lam, cfg.min_opacity,
                                   math.ceil(cfg.max_prune * n))
            with torch.no_grad():
                cur = render(model.p, model.bg, h, w, tau)
            relocated = density.relocate(model, cur, target, tau, cfg.spawn_half_widths,
                                         cfg.relocate_k, cfg.relocate_margin)
            moved = density.reorder(model, target, tau, cfg.reorder_moves, 0.1 * cfg.lam)
            g = model.mean_grad()
            gq = torch.quantile(g, torch.tensor([0.5, 0.9], device=dev)).tolist() if model.n > 1 else [0, 0]
            clones, splits = density.clone_split(model, cfg.grad_threshold, cfg.size_threshold,
                                                 math.ceil(cfg.max_growth * model.n) + 1)
            with torch.no_grad():
                cur = render(model.p, model.bg, h, w, tau)
            spawned = density.spawn(model, cur, target, cfg.spawn_half_widths, cfg.spawn_k, min_gain)
            model.reset_stats()
            event = (f" -{pruned} clone+{clones} split+{splits} spawn+{spawned}"
                     f" reloc={relocated} moved={moved}"
                     f" grad p50={gq[0]:.1e} p90={gq[1]:.1e}")

        if it % cfg.log_every == 0 or event:
            m = float(mse(img.detach(), target))
            row = {"it": it, "n": model.n, "mse": m, "fitness": fitness(m, model.n, cfg.lam),
                   "tau": tau, "ramp": ramp, "t": time.time() - start, "event": event.strip()}
            log.append(row)
            print(f"{it:5d} n={model.n:4d} mse={m:.5f} fit={row['fitness']:.5f} tau={tau:.2f}"
                  f"{' ramp' if ramp else ''} {row['t']:.0f}s{event}")
        if it % cfg.snapshot_every == 0:
            (out / "frames" / f"{it:06d}.svg").write_text(to_svg(model.p, model.bg, w, h))

    # Final: hard-edged evaluation and exact greedy pruning against the fitness.
    p = model.detached()
    a = hard_alphas(p, h, w)
    colour, bg = torch.sigmoid(p["col"]), torch.sigmoid(model.bg.detach())
    before_n, before_mse = model.n, float(mse(render_hard(p, model.bg, h, w), target))
    keep = density.greedy_prune(a, colour, bg, target, cfg.lam)
    model.restructure(keep)
    p = model.detached()
    final = render_hard(p, model.bg, h, w)
    m = float(mse(final, target))
    result = {"n": model.n, "mse": m, "fitness": fitness(m, model.n, cfg.lam),
              "before_prune": {"n": before_n, "mse": before_mse,
                               "fitness": fitness(before_mse, before_n, cfg.lam)},
              "seconds": time.time() - start}
    print(f"# hard: n={before_n} mse={before_mse:.5f} -> greedy prune n={model.n} "
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
        if isinstance(value, list):
            ap.add_argument(flag, type=int, nargs="+", default=value)
        elif isinstance(value, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        elif value is None:
            ap.add_argument(flag, type=int, default=None)
        else:
            ap.add_argument(flag, type=type(value), default=value)
    args = ap.parse_args(argv)
    train(Config(**vars(args)))


if __name__ == "__main__":
    main()
