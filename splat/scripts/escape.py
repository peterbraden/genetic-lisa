"""Is a fitted result trapped where a large translucent ellipse would help?

Loads an SVG written by splatlisa, then fine-tunes it for a few hundred steps
(anti-aliased edges, no density control, then the exact greedy prune) in two
kinds of arm:

- null: a negligible ellipse is added. Any perturbation changes the outcome
  by a few per cent, so these give the distribution to compare against;
- glaze: a large (semi-axes 15-50% of the longest side) faint (opacity
  0.1-0.3) ellipse is added at a random place and depth, coloured with the
  mean target colour beneath it.

    uv run python scripts/escape.py ../couch-db/_attachments/image/lisa.jpg ../images/splat-99.svg --lam 1e-5
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch

from splatlisa import density
from splatlisa.loss import photometric
from splatlisa.model import Model
from splatlisa.render import FIELDS, hard_alphas, inv_sigmoid, make_params, mse, render, render_hard
from splatlisa.train import Config, load_image

ELLIPSE = re.compile(r'<ellipse cx="([-\d.]+)" cy="([-\d.]+)" rx="([\d.]+)" ry="([\d.]+)" '
                     r'transform="rotate\(([-\d.]+) [^)]*\)" fill="#(\w{6})" fill-opacity="([\d.]+)"/>')
BACKGROUND = re.compile(r'<rect width="\d+" height="\d+" fill="#(\w{6})"/>')


def rgb(hexa: str) -> list[float]:
    return [int(hexa[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def load_svg(path: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Raw parameters and background logit from an SVG written by splatlisa."""
    svg = Path(path).read_text()
    rows = [m.groups() for m in ELLIPSE.finditer(svg)]
    p = make_params([[float(r[0]), float(r[1])] for r in rows],
                    [[float(r[2]), float(r[3])] for r in rows],
                    [math.radians(float(r[4])) for r in rows],
                    [rgb(r[5]) for r in rows],
                    [min(max(float(r[6]), 1e-4), 1 - 1e-4) for r in rows])
    p["col"] = p["col"].clamp(-9, 9)
    bg = inv_sigmoid(torch.tensor(rgb(BACKGROUND.search(svg).group(1))).clamp(1e-3, 1 - 1e-3))
    return p, bg


def finish(model: Model, target: torch.Tensor, lam: float) -> tuple[int, float]:
    _, h, w = target.shape
    p = model.detached()
    keep = density.greedy_prune(hard_alphas(p, h, w), torch.sigmoid(p["col"]),
                                torch.sigmoid(model.bg.detach()), target, lam)
    model.restructure(keep)
    m = float(mse(render_hard(model.detached(), model.bg, h, w), target))
    return model.n, m


def fine_tune(p, bg, extra, depth, target, cfg: Config, steps: int) -> tuple[int, float]:
    """Insert `extra` (one raw row) at `depth` (0 = bottom), fine-tune, prune."""
    _, h, w = target.shape
    size = max(h, w)
    p = {k: torch.cat([p[k][:depth], extra[k], p[k][depth:]]).clone() for k in FIELDS}
    lr = {"mu": cfg.lr_mu * size * 0.1, "log_s": cfg.lr_scale, "theta": cfg.lr_theta,
          "col": cfg.lr_col, "op": cfg.lr_op, "bg": cfg.lr_bg}
    model = Model(p, bg.clone(), lr)
    for _ in range(steps):
        img = render(model.p, model.bg, h, w, cfg.tau_end, ramp=True)
        loss = photometric(img, target, cfg.ssim_weight)
        loss = loss + cfg.opacity_penalty * cfg.lam * torch.sigmoid(model.p["op"]).sum()
        model.opt.zero_grad(set_to_none=True)
        loss.backward()
        model.opt.step()
        with torch.no_grad():
            model.p["log_s"].clamp_(math.log(0.3), math.log(2 * size))
    return finish(model, target, cfg.lam)


def glaze(rng: np.random.Generator, target: torch.Tensor) -> dict[str, torch.Tensor]:
    _, h, w = target.shape
    size = max(h, w)
    centre = [rng.uniform(0, w), rng.uniform(0, h)]
    axes = rng.uniform(0.15, 0.5, 2) * size
    theta = rng.uniform(0, math.pi)
    ys, xs = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    dx, dy = xs - centre[0], ys - centre[1]
    c, s = math.cos(theta), math.sin(theta)
    inside = ((c * dx + s * dy) / axes[0]) ** 2 + ((-s * dx + c * dy) / axes[1]) ** 2 <= 1
    colour = target[:, inside].mean(1) if inside.any() else target.mean((1, 2))
    return make_params([centre], [axes.tolist()], [theta], [colour.clamp(0.01, 0.99).tolist()],
                       [rng.uniform(0.1, 0.3)])


def null(rng: np.random.Generator, target: torch.Tensor) -> dict[str, torch.Tensor]:
    _, h, w = target.shape
    return make_params([[rng.uniform(0, w), rng.uniform(0, h)]], [[0.5, 0.5]], [0.0], [[0.5] * 3], [1e-3])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("svg")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--nulls", type=int, default=6)
    ap.add_argument("--glazes", type=int, default=18)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    target = load_image(args.image, None)
    cfg = Config(image=args.image, lam=args.lam)
    p, bg = load_svg(args.svg)
    n0 = p["mu"].shape[0]
    m0 = float(mse(render_hard(p, bg, *target.shape[1:]), target))
    print(f"start: n={n0} mse={m0:.5f} fitness={m0 + args.lam * n0:.5f}", flush=True)
    rng = np.random.default_rng(0)
    results = {"null": [], "glaze": []}
    for kind, count, make in [("null", args.nulls, null), ("glaze", args.glazes, glaze)]:
        for i in range(count):
            depth = int(rng.integers(0, n0 + 1))
            n, m = fine_tune(p, bg, make(rng, target), depth, target, cfg, args.steps)
            f = m + args.lam * n
            results[kind].append(f)
            print(f"{kind:5s} {i:2d} depth={depth:4d} n={n:4d} mse={m:.5f} fitness={f:.5f}", flush=True)
    for kind, fs in results.items():
        fs = np.array(fs)
        print(f"{kind:5s} fitness mean {fs.mean():.5f} sd {fs.std():.5f} min {fs.min():.5f} max {fs.max():.5f}")


if __name__ == "__main__":
    main()
