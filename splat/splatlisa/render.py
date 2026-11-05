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
    expand = lambda x: x[:, None, None]
    return _cover(
        grid[None] - p["mu"][:, None, None, :],
        expand(p["theta"]),
        expand(torch.exp(p["log_s"][:, 0])),
        expand(torch.exp(p["log_s"][:, 1])),
        tau,
        ramp,
    )


def _cover(d: Tensor, theta: Tensor, sx: Tensor, sy: Tensor, tau: float | None, ramp: bool) -> Tensor:
    """Coverage at offsets `d` (..., 2) from the centre; other arguments
    broadcast against d[..., 0]."""
    c, s = torch.cos(theta), torch.sin(theta)
    u = d[..., 0] * c + d[..., 1] * s
    v = -d[..., 0] * s + d[..., 1] * c
    r2 = (u / sx) ** 2 + (v / sy) ** 2
    if tau is None:
        return (r2 <= 1).float()
    r = torch.sqrt(r2 + 1e-12)
    # Ellipse radius in the direction of the sample: |(u, v)| / r. At the
    # centre this is ill-defined, so fall back to the geometric mean radius.
    rho = torch.where(r > 1e-3, torch.sqrt(u * u + v * v + 1e-12) / r, torch.sqrt(sx * sy).expand_as(r))
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


def render_dense(p: dict[str, Tensor], bg_logit: Tensor, h: int, w: int, tau: float, ramp: bool = False) -> Tensor:
    """Differentiable soft render evaluating every shape at every pixel."""
    grid = pixel_grid(h, w, device=bg_logit.device)
    return composite(alphas(p, grid, tau, ramp), torch.sigmoid(p["col"]), torch.sigmoid(bg_logit))


def edge_margin(tau: float | None, ramp: bool) -> float:
    """Distance beyond the ellipse at which soft coverage is negligible."""
    if tau is None or ramp:
        return 1.0
    return 6.0 * tau  # sigmoid(-6) < 0.0025


@torch.no_grad()
def footprint(p: dict[str, Tensor], h: int, w: int, margin: float) -> tuple[Tensor, Tensor]:
    """(shape, pixel) pairs for every pixel in each shape's bounding box,
    grown by `margin`. Pairs are sorted by pixel, then by shape index."""
    mu, sc, th = p["mu"], torch.exp(p["log_s"]), p["theta"]
    c, s = torch.cos(th), torch.sin(th)
    hx = torch.sqrt((sc[:, 0] * c) ** 2 + (sc[:, 1] * s) ** 2) + margin
    hy = torch.sqrt((sc[:, 0] * s) ** 2 + (sc[:, 1] * c) ** 2) + margin
    x0 = torch.floor(mu[:, 0] - hx).clamp(0, w).long()
    x1 = torch.ceil(mu[:, 0] + hx).clamp(0, w).long()
    y0 = torch.floor(mu[:, 1] - hy).clamp(0, h).long()
    y1 = torch.ceil(mu[:, 1] + hy).clamp(0, h).long()
    bw, bh = x1 - x0, y1 - y0
    count = bw * bh
    ids = torch.repeat_interleave(torch.arange(mu.shape[0], device=mu.device), count)
    offset = torch.arange(ids.shape[0], device=mu.device) - (torch.cumsum(count, 0) - count)[ids]
    pix = (y0[ids] + offset // bw[ids]) * w + x0[ids] + offset % bw[ids]
    order = torch.sort(pix, stable=True).indices
    return ids[order], pix[order]


def render(p: dict[str, Tensor], bg_logit: Tensor, h: int, w: int, tau: float, ramp: bool = False) -> Tensor:
    """Differentiable soft render, (3, H, W), evaluating each shape only
    within its bounding box (as 3DGS evaluates Gaussians only in their tiles).

    Pairs are packed into a (pixels, K) matrix, K being the deepest pixel's
    shape count, in painting order, and composited along K.
    """
    dev = bg_logit.device
    bg = torch.sigmoid(bg_logit)
    ids, pix = footprint(p, h, w, edge_margin(tau, ramp))
    if ids.numel() == 0:
        return bg[:, None, None].expand(3, h, w).clone()
    with torch.no_grad():
        per_pixel = torch.zeros(h * w, dtype=torch.long, device=dev).index_add_(0, pix, torch.ones_like(pix))
        first = torch.cumsum(per_pixel, 0) - per_pixel
        rank = torch.arange(pix.shape[0], device=dev) - first[pix]
        depth = int(per_pixel.max())
    q = {k: v[ids] for k, v in p.items()}
    centres = torch.stack([pix % w, pix // w], dim=1).float() + 0.5
    a = _pair_alpha(q, centres, tau, ramp)
    layers = torch.zeros(h * w, depth, device=dev).index_put((pix, rank), a)
    colours = torch.zeros(h * w, depth, 3, device=dev).index_put((pix, rank), torch.sigmoid(q["col"]))
    # Composite each pixel's list as in `composite`, with layers on dim 1.
    log1m = torch.log1p(-layers.clamp(max=1 - 1e-6))
    inclusive = torch.flip(torch.cumsum(torch.flip(log1m, [1]), 1), [1])
    weight = layers * torch.exp(inclusive - log1m)
    img = torch.einsum("pk,pkc->pc", weight, colours) + torch.exp(inclusive[:, 0])[:, None] * bg
    return img.T.reshape(3, h, w)


def _pair_alpha(q: dict[str, Tensor], centres: Tensor, tau: float | None, ramp: bool) -> Tensor:
    """Alpha of shape q[i] at point centres[i], (M,)."""
    cov = _cover(centres - q["mu"], q["theta"], torch.exp(q["log_s"][:, 0]), torch.exp(q["log_s"][:, 1]), tau, ramp)
    return cov * torch.sigmoid(q["op"])


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


@torch.no_grad()
def move_cost(alpha: Tensor, colour: Tensor, bg: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Exact change in MSE from moving each shape to the top, or to the
    bottom, of the painting order. Returns (top, bottom), each (N,).

    Both start from the image without shape i, W_i = C + T_i a_i (B_i - c_i):
    - top:    W_i (1 - a_i) + a_i c_i
    - bottom: W_i + P_i T_i a_i (c_i - bg), where P_i T_i is the background's
      transmittance through every shape other than i.
    """
    n = alpha.shape[0]
    top = torch.empty(n, device=alpha.device)
    bottom = torch.empty(n, device=alpha.device)
    if n == 0:
        return top, bottom
    t, _ = transmittance_above(alpha)
    img = composite(alpha, colour, bg)
    base = ((img - target) ** 2).mean()
    below = bg[:, None, None].expand_as(img).clone()
    beneath = torch.ones_like(alpha[0])  # P_i: transmittance of shapes below i
    bgc = bg[:, None, None]
    for i in range(n):
        a = alpha[i]
        c = colour[i][:, None, None]
        without = img + t[i] * a * (below - c)
        top[i] = ((without * (1 - a) + a * c - target) ** 2).mean() - base
        bottom[i] = ((without + beneath * t[i] * a * (c - bgc) - target) ** 2).mean() - base
        below = below * (1 - a) + a * c
        beneath = beneath * (1 - a)
    return top, bottom


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
