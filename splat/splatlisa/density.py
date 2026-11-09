"""Adaptive density control: directed addition and removal of ellipses.

- clone / split (3DGS): driven by the accumulated positional gradient.
- spawn (after Image-GS): placed where a flat patch would most reduce error.
- prune: removes shapes whose exact leave-one-out MSE saving is below the
  per-shape cost `lam` of the fitness function.
- relocate (after 3DGS-MCMC): moves the least valuable shapes to spawn sites
  whose estimated saving exceeds what the shape currently provides.
- reorder: moves shapes to the top or bottom of the painting order when that
  exactly reduces the error.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import Model
from .render import FIELDS, Pairs, inv_sigmoid, move_cost_pairs, reindex, removal_cost_pairs, soft_pairs

# Radius of a disc with the same area as a square of half-width h, over h.
DISC_PER_BOX = 2 / math.sqrt(math.pi)


def clone_split(
    model: Model,
    grad_threshold: float,
    size_threshold: float,
    max_new: int,
    nudge: float = 0.5,
    offset: float = 0.45,
    major: float = 0.55,
    minor: float = 0.85,
) -> tuple[int, int]:
    """Replace each high-gradient shape with two: a clone if it is small,
    two halves along its major axis if it is large. Returns (clones, splits)."""
    n = model.n
    if n == 0 or model.grad_count == 0 or max_new <= 0:
        return 0, 0
    g = model.mean_grad()
    idx = torch.nonzero(g > grad_threshold).flatten()
    if idx.numel() == 0:
        return 0, 0
    idx = idx[torch.argsort(g[idx], descending=True)][:max_new]

    p = model.detached()
    scale = torch.exp(p["log_s"][idx])
    big = scale.max(dim=1).values > size_threshold
    rows = {k: torch.repeat_interleave(p[k][idx], 2, dim=0).clone() for k in FIELDS}

    # Clones: halve the opacity in the sense that two overlapping copies
    # reproduce the parent, 1 - (1 - a')^2 = a, then nudge one copy down the
    # accumulated gradient so the pair can separate.
    small = ~big
    if small.any():
        a = torch.sigmoid(p["op"][idx[small]])
        a_half = 1 - torch.sqrt(1 - a)
        sel = torch.nonzero(small).flatten()
        rows["op"][2 * sel] = inv_sigmoid(a_half)
        rows["op"][2 * sel + 1] = inv_sigmoid(a_half)
        direction = -F.normalize(model.grad_vec[idx[small]], dim=1, eps=1e-12)
        step = nudge * scale[small].min(dim=1).values[:, None]
        rows["mu"][2 * sel + 1] += direction * step

    # Splits: two ellipses along the major axis, each spanning half of it.
    if big.any():
        sel = torch.nonzero(big).flatten()
        s = scale[big]
        th = p["theta"][idx[big]]
        major_is_x = s[:, 0] >= s[:, 1]
        axis_angle = torch.where(major_is_x, th, th + math.pi / 2)
        axis = torch.stack([torch.cos(axis_angle), torch.sin(axis_angle)], dim=1)
        shift = axis * s.max(dim=1).values[:, None] * offset
        rows["mu"][2 * sel] -= shift
        rows["mu"][2 * sel + 1] += shift
        shrink = torch.where(
            major_is_x[:, None],
            torch.tensor([major, minor], device=s.device),
            torch.tensor([minor, major], device=s.device),
        )
        child = torch.log(s * shrink)
        rows["log_s"][2 * sel] = child
        rows["log_s"][2 * sel + 1] = child

    # Children take the parent's place in the painting order.
    slot = {int(i): j for j, i in enumerate(idx.tolist())}
    order = []
    for i in range(n):
        if i in slot:
            j = slot[i]
            order += [n + 2 * j, n + 2 * j + 1]
        else:
            order.append(i)
    model.restructure(torch.tensor(order), rows)
    return int(small.sum()), int(big.sum())


@torch.no_grad()
def spawn_candidates(
    img: Tensor,
    target: Tensor,
    half_widths: list[int],
    k: int,
    min_gain: float,
) -> list[tuple[float, float, float, Tensor, float]]:
    """Locations and sizes where painting an opaque disc of the local mean
    target colour would most reduce the summed squared error.

    For a box of half-width h, the saving is
        A * (mean |img - target|^2 - var(target))
    computed for every pixel with box filters. Returns up to `k`
    non-overlapping (x, y, radius, colour, saving), best first, whose saving
    exceeds `min_gain`.
    """
    _, h, w = target.shape
    err2 = ((img - target) ** 2).sum(0, keepdim=True)
    t2 = (target**2).sum(0, keepdim=True)
    gains, colours = [], []
    for hw in half_widths:
        box = lambda x: F.avg_pool2d(
            x[None], 2 * hw + 1, stride=1, padding=hw, count_include_pad=False
        )[0]
        mean_t = box(target)
        var_t = (box(t2) - (mean_t**2).sum(0, keepdim=True)).clamp(min=0)
        area = (2 * hw + 1) ** 2
        gains.append((area * (box(err2) - var_t))[0])
        colours.append(mean_t)
    gain = torch.stack(gains)  # (S, H, W)
    ys = torch.arange(h, device=img.device, dtype=torch.float32)[:, None]
    xs = torch.arange(w, device=img.device, dtype=torch.float32)[None, :]

    out = []
    for _ in range(k):
        flat = int(torch.argmax(gain))
        best = float(gain.flatten()[flat])
        if best <= min_gain:
            break
        si, rem = divmod(flat, h * w)
        y, x = divmod(rem, w)
        hw = half_widths[si]
        radius = (2 * hw + 1) / 2 * DISC_PER_BOX
        out.append((x + 0.5, y + 0.5, radius, colours[si][:, y, x].clamp(0.02, 0.98), best))
        # Suppress neighbouring candidates so spawns do not stack.
        near = (xs - x) ** 2 + (ys - y) ** 2 <= radius**2
        gain[:, near] = -float("inf")
    return out


def glaze_candidates(
    img: Tensor,
    target: Tensor,
    half_widths: list[int],
    k: int,
    min_gain: float,
    min_opacity: float = 0.05,
) -> list[tuple[float, float, float, Tensor, float, float]]:
    """Sites for translucent layers: per box, the least-squares opacity `a`
    and colour `c` of a layer composited over the current image.

    With r = target - img and C = img over a box of n pixels, the layer
    leaves r - u + a·C (u = a·c), and the saving in summed squared error is
        n·|mean r|² - 2a·S_rC - a²·S_CC
    (S: centred sums, over channels). a = 1 recovers the opaque rule of
    `spawn_candidates`; where the image already has the right structure
    (S_rC ≈ 0), the optimum is a faint wash that corrects only the bias.
    `a` is bounded below by `min_opacity` and by keeping `c` within [0, 1].
    Returns up to `k` non-overlapping (x, y, radius, colour, saving, opacity),
    best first, whose saving exceeds `min_gain`.
    """
    _, h, w = target.shape
    r = target - img
    gains, colours, opacities = [], [], []
    for hw in half_widths:
        box = lambda x: F.avg_pool2d(
            x[None], 2 * hw + 1, stride=1, padding=hw, count_include_pad=False
        )[0]
        n = (2 * hw + 1) ** 2
        m_r, m_c = box(r), box(img)
        s_rc = n * (box(r * img) - m_r * m_c).sum(0)
        s_cc = n * (box(img * img) - m_c**2).sum(0).clamp(min=0)
        # c = m_c + m_r / a must lie in [0, 1].
        room = torch.where(m_r > 0, 1 - m_c, m_c).clamp(min=1e-6)
        a_lo = (m_r.abs() / room).amax(0).clamp(min=min_opacity, max=1)
        a = torch.where(s_cc > 1e-12, -s_rc / s_cc.clamp(min=1e-12), a_lo)
        a = torch.maximum(a, a_lo).clamp(max=1)
        gains.append(n * (m_r**2).sum(0) - 2 * a * s_rc - a**2 * s_cc)
        colours.append((m_c + m_r / a).clamp(0.02, 0.98))
        opacities.append(a)
    gain = torch.stack(gains)  # (S, H, W)
    ys = torch.arange(h, device=img.device, dtype=torch.float32)[:, None]
    xs = torch.arange(w, device=img.device, dtype=torch.float32)[None, :]

    out = []
    for _ in range(k):
        flat = int(torch.argmax(gain))
        best = float(gain.flatten()[flat])
        if best <= min_gain:
            break
        si, rem = divmod(flat, h * w)
        y, x = divmod(rem, w)
        radius = (2 * half_widths[si] + 1) / 2 * DISC_PER_BOX
        out.append((x + 0.5, y + 0.5, radius, colours[si][:, y, x], best, float(opacities[si][y, x])))
        near = (xs - x) ** 2 + (ys - y) ** 2 <= radius**2
        gain[:, near] = -float("inf")
    return out


def glaze(
    model: Model,
    img: Tensor,
    target: Tensor,
    half_widths: list[int],
    k: int,
    min_gain: float,
    min_opacity: float = 0.05,
) -> int:
    """Append translucent layers on top of the stack at the best glaze sites."""
    cands = glaze_candidates(img, target, half_widths, k, min_gain, min_opacity)
    if not cands:
        return 0
    rows = candidate_rows(cands, 0.5, model.device)
    rows["op"] = inv_sigmoid(torch.tensor([c[5] for c in cands], device=model.device).clamp(1e-3, 1 - 1e-3))
    model.restructure(torch.arange(model.n + len(cands)), rows)
    return len(cands)


def spawn(
    model: Model,
    img: Tensor,
    target: Tensor,
    half_widths: list[int],
    k: int,
    min_gain: float,
    opacity: float = 0.8,
) -> int:
    """Append new ellipses on top of the stack at the best spawn sites."""
    cands = spawn_candidates(img, target, half_widths, k, min_gain)
    if not cands:
        return 0
    order = torch.arange(model.n + len(cands))
    model.restructure(order, candidate_rows(cands, opacity, model.device))
    return len(cands)


def candidate_rows(cands, opacity: float, device) -> dict[str, Tensor]:
    """Parameter rows for circles at spawn candidates."""
    return {
        "mu": torch.tensor([[c[0], c[1]] for c in cands], device=device),
        "log_s": torch.log(torch.tensor([[c[2], c[2]] for c in cands], device=device)),
        "theta": torch.zeros(len(cands), device=device),
        "col": inv_sigmoid(torch.stack([c[3] for c in cands]).to(device)),
        "op": inv_sigmoid(torch.full((len(cands),), opacity, device=device)),
    }


def select_relocations(cost: Tensor, gains: list[float], margin: float) -> list[tuple[int, int]]:
    """Pair the cheapest shapes with the best sites, (shape, site), while a
    site's estimated saving exceeds `margin` times the shape's value. Both are
    in MSE units; `gains` must be in descending order."""
    pairs = []
    for site, victim in zip(range(len(gains)), torch.argsort(cost).tolist()):
        if gains[site] <= margin * max(float(cost[victim]), 0.0):
            break
        pairs.append((victim, site))
    return pairs


@torch.no_grad()
def relocate(
    model: Model,
    img: Tensor,
    target: Tensor,
    tau: float,
    half_widths: list[int],
    k: int,
    margin: float = 1.0,
    opacity: float = 0.8,
) -> int:
    """Move up to `k` low-value shapes to high-error sites, keeping N fixed.
    Relocated shapes go to the top of the stack with fresh optimiser state."""
    if model.n == 0 or k <= 0:
        return 0
    p = model.detached()
    _, h, w = target.shape
    cost = removal_cost_pairs(soft_pairs(p, h, w, tau), torch.sigmoid(p["col"]), torch.sigmoid(model.bg), target)
    cands = spawn_candidates(img, target, half_widths, k, 0.0)
    # Spawn gains are summed squared error; convert to MSE.
    pairs = select_relocations(cost, [c[4] / (3 * h * w) for c in cands], margin)
    if not pairs:
        return 0
    victims = {v for v, _ in pairs}
    keep = [i for i in range(model.n) if i not in victims]
    rows = candidate_rows([cands[s] for _, s in pairs], opacity, model.device)
    model.restructure(torch.tensor(keep + list(range(model.n, model.n + len(pairs)))), rows)
    return len(pairs)


@torch.no_grad()
def reorder(model: Model, target: Tensor, tau: float, max_moves: int, min_gain: float) -> int:
    """Greedily move single shapes to the top or bottom of the stack while
    that reduces MSE by more than `min_gain`. Exact; returns moves made."""
    if model.n < 2 or max_moves <= 0:
        return 0
    p = model.detached()
    _, h, w = target.shape
    pairs = soft_pairs(p, h, w, tau)
    colour, bg = torch.sigmoid(p["col"]), torch.sigmoid(model.bg)
    perm = torch.arange(model.n, device=model.device)
    moves = 0
    for _ in range(max_moves):
        top, bottom = move_cost_pairs(reindex(pairs, perm), colour[perm], bg, target)
        i_top, i_bot = int(torch.argmin(top)), int(torch.argmin(bottom))
        if min(float(top[i_top]), float(bottom[i_bot])) >= -min_gain:
            break
        if top[i_top] <= bottom[i_bot]:
            perm = torch.cat([perm[:i_top], perm[i_top + 1 :], perm[i_top : i_top + 1]])
        else:
            perm = torch.cat([perm[i_bot : i_bot + 1], perm[:i_bot], perm[i_bot + 1 :]])
        moves += 1
    if moves:
        model.restructure(perm)
    return moves


def select_prunable(cost: Tensor, opacity: Tensor, lam: float, min_opacity: float, max_remove: int) -> Tensor:
    """Boolean mask of shapes to remove: the cheapest `max_remove` of those
    that are near-transparent or save less than `lam` MSE."""
    weak = (cost < lam) | (opacity < min_opacity)
    idx = torch.nonzero(weak).flatten()
    idx = idx[torch.argsort(cost[idx])][:max_remove]
    mask = torch.zeros_like(weak)
    mask[idx] = True
    return mask


@torch.no_grad()
def prune(model: Model, target: Tensor, tau: float, lam: float, min_opacity: float, max_remove: int) -> int:
    if model.n == 0 or max_remove <= 0:
        return 0
    p = model.detached()
    _, h, w = target.shape
    cost = removal_cost_pairs(soft_pairs(p, h, w, tau), torch.sigmoid(p["col"]), torch.sigmoid(model.bg), target)
    mask = select_prunable(cost, torch.sigmoid(p["op"]), lam, min_opacity, max_remove)
    if not mask.any():
        return 0
    model.restructure(torch.nonzero(~mask).flatten())
    return int(mask.sum())


@torch.no_grad()
def greedy_prune(pairs: Pairs, colour: Tensor, bg: Tensor, target: Tensor, lam: float) -> Tensor:
    """Repeatedly remove the shape whose removal most improves the fitness
    `MSE + lam * N`, until none does. Exact; returns the kept indices."""
    keep = torch.arange(pairs.n, device=colour.device)
    while keep.numel():
        cost = removal_cost_pairs(reindex(pairs, keep), colour[keep], bg, target)
        i = int(torch.argmin(cost))
        if cost[i] >= lam:
            break
        keep = torch.cat([keep[:i], keep[i + 1 :]])
    return keep
