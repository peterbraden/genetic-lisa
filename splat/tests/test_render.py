import io
import math

import numpy as np
import pytest
import torch
from PIL import Image

from splatlisa.render import (
    alphas,
    composite,
    coverage,
    hard_alphas,
    make_params,
    mse,
    pixel_grid,
    removal_cost,
    render,
    render_hard,
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
