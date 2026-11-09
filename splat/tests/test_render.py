import io
import math

import numpy as np
import pytest
import torch
from PIL import Image

from splatlisa.render import (
    alphas,
    composite,
    composite_pairs,
    coverage,
    hard_alphas,
    hard_pairs,
    make_params,
    move_cost,
    move_cost_pairs,
    mse,
    pixel_grid,
    reindex,
    removal_cost,
    removal_cost_pairs,
    render,
    render_dense,
    render_hard,
    soft_pairs,
)
from splatlisa.svg import to_svg


def random_layers(n=5, h=7, w=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, h, w, generator=g) * 0.9, torch.rand(n, 3, generator=g), torch.rand(3, generator=g)


def paint_sequentially(alpha, colour, bg):
    img = bg[:, None, None].expand(3, *alpha.shape[1:]).clone()
    for a, c in zip(alpha, colour):
        img = img * (1 - a) + a * c[:, None, None]
    return img


def test_composite_matches_painters_algorithm():
    alpha, colour, bg = random_layers()
    assert torch.allclose(composite(alpha, colour, bg), paint_sequentially(alpha, colour, bg), atol=1e-5)


def test_composite_with_no_shapes_is_background():
    _, _, bg = random_layers()
    img = composite(torch.zeros(0, 4, 5), torch.zeros(0, 3), bg)
    assert torch.allclose(img, bg[:, None, None].expand(3, 4, 5))


def test_removal_cost_matches_brute_force():
    alpha, colour, bg = random_layers(n=6)
    target = torch.rand(3, *alpha.shape[1:], generator=torch.Generator().manual_seed(1))
    base = mse(paint_sequentially(alpha, colour, bg), target)
    expected = torch.stack([
        mse(paint_sequentially(torch.cat([alpha[:i], alpha[i + 1:]]), torch.cat([colour[:i], colour[i + 1:]]), bg), target) - base
        for i in range(alpha.shape[0])
    ])
    assert torch.allclose(removal_cost(alpha, colour, bg, target), expected, atol=1e-6)


def test_move_cost_matches_brute_force():
    alpha, colour, bg = random_layers(n=6, seed=2)
    target = torch.rand(3, *alpha.shape[1:], generator=torch.Generator().manual_seed(3))
    base = mse(paint_sequentially(alpha, colour, bg), target)
    n = alpha.shape[0]
    top, bottom = move_cost(alpha, colour, bg, target)
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        for order, got in ((rest + [i], top[i]), ([i] + rest, bottom[i])):
            expected = mse(paint_sequentially(alpha[order], colour[order], bg), target) - base
            assert got.item() == pytest.approx(expected.item(), abs=1e-6)


def test_removal_cost_is_negative_for_a_harmful_shape():
    # Target is plain grey; a red disc over it only adds error.
    p = make_params([[10, 10]], [[4, 4]], [0], [[1, 0, 0]], [0.9])
    grid = pixel_grid(20, 20)
    a = alphas(p, grid, None)
    target = torch.full((3, 20, 20), 0.5)
    cost = removal_cost(a, torch.sigmoid(p["col"]), torch.full((3,), 0.5), target)
    assert cost.item() < 0


def svg_transform_point(x, y, cx, cy, deg):
    """Apply SVG `rotate(deg cx cy)` to a point, as the SVG spec defines it."""
    t = math.radians(deg)
    dx, dy = x - cx, y - cy
    return cx + dx * math.cos(t) - dy * math.sin(t), cy + dx * math.sin(t) + dy * math.cos(t)


@pytest.mark.parametrize("deg", [0, 30, 90, 137])
def test_rotation_convention_matches_svg(deg):
    cx, cy, rx, ry = 20.0, 15.0, 9.0, 3.0
    p = make_params([[cx, cy]], [[rx, ry]], [math.radians(deg)], [[0, 0, 0]], [0.5])
    inside, outside = [], []
    for k in range(16):
        phi = 2 * math.pi * k / 16
        inside.append(svg_transform_point(cx + 0.95 * rx * math.cos(phi), cy + 0.95 * ry * math.sin(phi), cx, cy, deg))
        outside.append(svg_transform_point(cx + 1.05 * rx * math.cos(phi), cy + 1.05 * ry * math.sin(phi), cx, cy, deg))
    pts = lambda xs: torch.tensor(xs, dtype=torch.float32)[None]  # (1, K, 2) as a 1xK grid
    assert coverage(p, pts(inside), None).min() == 1
    assert coverage(p, pts(outside), None).max() == 0


def test_hard_coverage_area_matches_ellipse_area():
    p = make_params([[32.3, 30.7]], [[20, 7]], [0.6], [[0, 0, 0]], [0.9999])
    area = hard_alphas(p, 64, 64, ss=8).sum().item()
    assert area == pytest.approx(math.pi * 20 * 7, rel=0.01)


def test_soft_coverage_converges_to_hard_edge():
    p = make_params([[16, 16]], [[10, 5]], [0.3], [[0, 0, 0]], [0.5])
    grid = pixel_grid(32, 32)
    hard = coverage(p, grid, None)
    soft = coverage(p, grid, 0.01)
    assert (soft - hard).abs().max() < 0.05 or (soft.round() != hard).sum() <= 2


def test_soft_edge_is_half_on_the_boundary():
    p = make_params([[0, 0]], [[10, 4]], [0.0], [[0, 0, 0]], [0.5])
    pts = torch.tensor([[[10.0, 0.0], [0.0, 4.0], [10 / math.sqrt(2), 4 / math.sqrt(2)]]])
    assert torch.allclose(coverage(p, pts, 1.0), torch.full((1, 1, 3), 0.5), atol=1e-4)


def test_svg_export_matches_independent_rasteriser():
    """resvg is an independent SVG implementation: our hard render of the
    exported SVG must agree with it, which checks geometry, rotation,
    painting order, colour and opacity together."""
    resvg_py = pytest.importorskip("resvg_py")
    g = torch.Generator().manual_seed(3)
    n, h, w = 12, 40, 50
    p = make_params(
        torch.rand(n, 2, generator=g) * torch.tensor([w, h]),
        torch.rand(n, 2, generator=g) * 12 + 2,
        torch.rand(n, generator=g) * math.pi,
        torch.rand(n, 3, generator=g) * 0.9 + 0.05,
        torch.rand(n, generator=g) * 0.8 + 0.1,
    )
    bg = torch.logit(torch.tensor([0.2, 0.4, 0.6]))
    png = bytes(resvg_py.svg_to_bytes(svg_string=to_svg(p, bg, w, h)))
    ref = torch.from_numpy(np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)
    ours = render_hard(p, bg, h, w, ss=8)
    diff = (ours - ref).abs()
    # Exported values are rounded to 8-bit colour and 2 decimal places.
    assert diff.mean() < 0.005
    assert diff.max() < 0.08


def test_soft_render_approaches_svg_at_small_tau():
    g = torch.Generator().manual_seed(4)
    n, h, w = 10, 30, 30
    p = make_params(torch.rand(n, 2, generator=g) * 30, torch.rand(n, 2, generator=g) * 8 + 3,
                    torch.rand(n, generator=g) * 3, torch.rand(n, 3, generator=g), torch.rand(n, generator=g) * 0.8 + 0.1)
    bg = torch.zeros(3)
    hard = render_hard(p, bg, h, w, ss=8)
    assert mse(render(p, bg, h, w, 0.4), hard) < mse(render(p, bg, h, w, 2.0), hard)
    assert mse(render(p, bg, h, w, 0.4), hard) < 1e-3


def test_ramp_edge_approximates_anti_aliased_svg_better_than_sigmoid():
    g = torch.Generator().manual_seed(5)
    n, h, w = 10, 30, 30
    p = make_params(torch.rand(n, 2, generator=g) * 30, torch.rand(n, 2, generator=g) * 8 + 3,
                    torch.rand(n, generator=g) * 3, torch.rand(n, 3, generator=g), torch.rand(n, generator=g) * 0.8 + 0.1)
    bg = torch.zeros(3)
    hard = render_hard(p, bg, h, w, ss=16)
    ramp = render(p, bg, h, w, 0.5, ramp=True)
    assert mse(ramp, hard) < 0.2 * mse(render(p, bg, h, w, 0.5), hard)


@pytest.mark.parametrize("tau,ramp", [(2.0, False), (0.5, False), (0.5, True)])
def test_culled_render_matches_dense_render_and_gradients(tau, ramp):
    g = torch.Generator().manual_seed(6)
    n, h, w = 15, 33, 41
    p = make_params(
        torch.rand(n, 2, generator=g) * torch.tensor([w + 10.0, h + 10.0]) - 5,  # some partly off-canvas
        torch.rand(n, 2, generator=g) * 10 + 0.5,
        torch.rand(n, generator=g) * 3,
        torch.rand(n, 3, generator=g),
        torch.rand(n, generator=g) * 0.8 + 0.1,
    )
    p["log_s"][0] = math.log(60.0)  # one shape covering the whole canvas
    bg = torch.tensor([0.1, -0.3, 0.2])
    target = torch.rand(3, h, w, generator=g)
    grads = []
    for fn in (render, render_dense):
        q = {k: v.clone().requires_grad_(True) for k, v in p.items()}
        img = fn(q, bg, h, w, tau, ramp)
        mse(img, target).backward()
        grads.append((img.detach(), {k: v.grad for k, v in q.items()}))
    (culled, g_culled), (dense, g_dense) = grads
    assert (culled - dense).abs().max() < 3e-3
    for k in p:
        assert torch.allclose(g_culled[k], g_dense[k], atol=1e-4, rtol=0.02), k


def test_culled_render_of_no_shapes_is_background():
    p = {k: v[:0] for k, v in make_params([[1, 1]], [[1, 1]], [0], [[0, 0, 0]], [0.5]).items()}
    bg = torch.tensor([0.0, 1.0, -1.0])
    assert torch.allclose(render(p, bg, 4, 5, 1.0), torch.sigmoid(bg)[:, None, None].expand(3, 4, 5))


def overlapping_shapes(n=14, h=23, w=29, seed=8):
    g = torch.Generator().manual_seed(seed)
    p = make_params(
        torch.rand(n, 2, generator=g) * torch.tensor([w + 8.0, h + 8.0]) - 4,  # some partly off-canvas
        torch.rand(n, 2, generator=g) * 9 + 0.7,
        torch.rand(n, generator=g) * math.pi,
        torch.rand(n, 3, generator=g),
        torch.rand(n, generator=g) * 0.9 + 0.05,
    )
    return p, torch.rand(3, generator=g), torch.rand(3, h, w, generator=g)


def dense_hard(p, h, w):
    return hard_alphas(p, h, w, ss=2)


@pytest.mark.parametrize("order", [None, [5, 0, 13, 2, 9, 7], list(range(13, -1, -1))])
def test_sparse_costs_match_dense(order):
    """Sparse composite, removal and move costs equal the dense ones, also
    after reindexing to a subset in a new order."""
    p, bg, target = overlapping_shapes()
    _, h, w = target.shape
    colour = torch.sigmoid(p["col"])
    a, pairs = dense_hard(p, h, w), hard_pairs(p, h, w, ss=2)
    assert torch.allclose(pairs_to_dense(pairs), a, atol=1e-6)
    if order is not None:
        order = torch.tensor(order)
        a, colour, pairs = a[order], colour[order], reindex(pairs, order)
    assert torch.allclose(composite_pairs(pairs, colour, bg), composite(a, colour, bg), atol=1e-6)
    assert torch.allclose(removal_cost_pairs(pairs, colour, bg, target), removal_cost(a, colour, bg, target), atol=1e-7)
    for got, want in zip(move_cost_pairs(pairs, colour, bg, target), move_cost(a, colour, bg, target)):
        assert torch.allclose(got, want, atol=1e-7)


def pairs_to_dense(pairs):
    out = torch.zeros(pairs.n, pairs.h * pairs.w)
    out[pairs.ids, pairs.pix] = pairs.alpha
    return out.reshape(pairs.n, pairs.h, pairs.w)


def test_soft_pairs_match_dense_ramp_alphas():
    # The ramp edge is exactly zero beyond half a pixel, inside the footprint margin.
    p, _, target = overlapping_shapes(seed=9)
    _, h, w = target.shape
    dense = alphas(p, pixel_grid(h, w), 0.5, ramp=True)
    assert torch.allclose(pairs_to_dense(soft_pairs(p, h, w, 0.5, ramp=True)), dense, atol=1e-6)
