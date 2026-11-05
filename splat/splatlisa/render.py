"""Ellipse parameters, differentiable rendering and leave-one-out costs.

Conventions match SVG: pixel (x, y) has its centre at (x + 0.5, y + 0.5), y
points down, shapes are painted in index order with the "over" operator in
sRGB, on top of an opaque background colour.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# Parameter names, in the order used by the optimiser.
FIELDS = ("mu", "log_s", "theta", "col", "op")


def inv_sigmoid(x: Tensor | float) -> Tensor:
    x = torch.as_tensor(x).clamp(1e-4, 1 - 1e-4)
    return torch.log(x / (1 - x))


def make_params(mu, scale, theta, colour, opacity, device=None) -> dict[str, Tensor]:
    """Build raw (unconstrained) parameters from natural values."""
    t = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
    return {
        "mu": t(mu).reshape(-1, 2),
        "log_s": torch.log(t(scale).reshape(-1, 2)),
        "theta": t(theta).reshape(-1),
        "col": inv_sigmoid(t(colour).reshape(-1, 3)),
        "op": inv_sigmoid(t(opacity).reshape(-1)),
    }


def pixel_grid(h: int, w: int, ss: int = 1, device=None) -> Tensor:
    """(h*ss, w*ss, 2) sample positions in pixel units, as (x, y)."""
    ys = (torch.arange(h * ss, device=device, dtype=torch.float32) + 0.5) / ss
    xs = (torch.arange(w * ss, device=device, dtype=torch.float32) + 0.5) / ss
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)


def coverage(p: dict[str, Tensor], grid: Tensor, tau: float | None, ramp: bool = False) -> Tensor:
    """Per-shape coverage in [0, 1], shape (N, H, W), excluding opacity.

    With `tau` set, the edge is a sigmoid of the approximate signed distance
    (in pixels, measured along the ray from the centre) with width `tau`.
    With `ramp` the edge is instead a linear ramp one pixel wide, which is
    the box-filtered coverage of a straight edge: what an anti-aliasing SVG
    rasteriser produces. With `tau=None` the edge is hard (binary).
    """
    d = grid[None] - p["mu"][:, None, None, :]
    c = torch.cos(p["theta"])[:, None, None]
    s = torch.sin(p["theta"])[:, None, None]
    u = d[..., 0] * c + d[..., 1] * s
    v = -d[..., 0] * s + d[..., 1] * c
    sc = torch.exp(p["log_s"])
    r2 = (u / sc[:, 0, None, None]) ** 2 + (v / sc[:, 1, None, None]) ** 2
    if tau is None:
        return (r2 <= 1).float()
    r = torch.sqrt(r2 + 1e-12)
    # Ellipse radius in the direction of the sample: |(u, v)| / r. At the
    # centre this is ill-defined, so fall back to the geometric mean radius.
    rho = torch.where(
        r > 1e-3,
        torch.sqrt(u * u + v * v + 1e-12) / r,
        torch.sqrt(sc[:, 0] * sc[:, 1])[:, None, None],
    )
    dist = (1 - r) * rho
    if ramp:
        return (0.5 + dist).clamp(0, 1)
    return torch.sigmoid(dist / tau)


def alphas(p: dict[str, Tensor], grid: Tensor, tau: float | None, ramp: bool = False) -> Tensor:
    return coverage(p, grid, tau, ramp) * torch.sigmoid(p["op"])[:, None, None]


def transmittance_above(alpha: Tensor) -> tuple[Tensor, Tensor]:
    """Returns (T, T_bg): the fraction of each layer, and of the background,
    visible through all layers painted after it."""
    log1m = torch.log1p(-alpha.clamp(max=1 - 1e-6))
    inclusive = torch.flip(torch.cumsum(torch.flip(log1m, [0]), 0), [0])
    t = torch.exp(inclusive - log1m)
    t_bg = torch.exp(inclusive[0]) if alpha.shape[0] else torch.ones_like(alpha[0:1].sum(0))
    return t, t_bg


def composite(alpha: Tensor, colour: Tensor, bg: Tensor) -> Tensor:
    """Painter's-order "over" compositing. alpha (N, H, W), colour (N, 3),
    bg (3,). Returns (3, H, W)."""
    if alpha.shape[0] == 0:
        return bg[:, None, None].expand(3, *alpha.shape[1:]).clone()
    t, t_bg = transmittance_above(alpha)
    img = torch.einsum("nhw,nc->chw", alpha * t, colour)
    return img + t_bg[None] * bg[:, None, None]


def render(p: dict[str, Tensor], bg_logit: Tensor, h: int, w: int, tau: float, ramp: bool = False) -> Tensor:
    """Differentiable soft render, (3, H, W)."""
    grid = pixel_grid(h, w, device=bg_logit.device)
    return composite(alphas(p, grid, tau, ramp), torch.sigmoid(p["col"]), torch.sigmoid(bg_logit))


@torch.no_grad()
def hard_alphas(p: dict[str, Tensor], h: int, w: int, ss: int = 4) -> Tensor:
    """Alpha per shape for hard-edged ellipses, with edge coverage estimated
    by `ss`x`ss` supersampling, as an anti-aliasing SVG rasteriser does."""
    device = p["mu"].device
    grid = pixel_grid(h, w, ss, device=device)
    op = torch.sigmoid(p["op"])
    out = torch.empty((p["mu"].shape[0], h, w), device=device)
    # Chunk over shapes to bound memory at high supersampling.
    step = max(1, int(4e7 // (h * w * ss * ss)))
    for i in range(0, p["mu"].shape[0], step):
        sub = {k: v[i : i + step] for k, v in p.items()}
        cov = coverage(sub, grid, None)
        cov = cov.reshape(-1, h, ss, w, ss).mean(dim=(2, 4))
        out[i : i + step] = cov * op[i : i + step, None, None]
    return out


@torch.no_grad()
def render_hard(p: dict[str, Tensor], bg_logit: Tensor, h: int, w: int, ss: int = 4) -> Tensor:
    a = hard_alphas(p, h, w, ss)
    return composite(a, torch.sigmoid(p["col"]), torch.sigmoid(bg_logit))


def mse(img: Tensor, target: Tensor) -> Tensor:
    return ((img - target) ** 2).mean()


@torch.no_grad()
def removal_cost(alpha: Tensor, colour: Tensor, bg: Tensor, target: Tensor) -> Tensor:
    """Exact increase in MSE caused by removing each shape on its own, (N,).

    Removing layer i changes the composite by T_i * a_i * (B_i - c_i), where
    T_i is the transmittance above it and B_i the composite of everything
    beneath it. A negative value means the shape makes the image worse.
    """
    n = alpha.shape[0]
    out = torch.empty(n, device=alpha.device)
    if n == 0:
        return out
    t, _ = transmittance_above(alpha)
    img = composite(alpha, colour, bg)
    err = img - target
    base = (err**2).mean()
    below = bg[:, None, None].expand_as(img).clone()
    for i in range(n):
        a = alpha[i]
        c = colour[i][:, None, None]
        changed = err + t[i] * a * (below - c)
        out[i] = (changed**2).mean() - base
        below = below * (1 - a) + a * c
    return out


def natural(p: dict[str, Tensor]) -> dict[str, Tensor]:
    """Constrained values: centre, semi-axes, angle (radians), colour, opacity."""
    return {
        "mu": p["mu"],
        "scale": torch.exp(p["log_s"]),
        "theta": p["theta"],
        "colour": torch.sigmoid(p["col"]),
        "opacity": torch.sigmoid(p["op"]),
    }


def degrees(theta: Tensor) -> Tensor:
    return theta * (180 / math.pi)
