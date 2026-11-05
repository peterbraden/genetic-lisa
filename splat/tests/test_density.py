import math

import torch

from splatlisa import density
from splatlisa.model import Model
from splatlisa.render import alphas, composite, inv_sigmoid, make_params, pixel_grid, render, render_hard


def model_of(params, bg=(0.5, 0.5, 0.5)):
    return Model(params, inv_sigmoid(torch.tensor(bg)))


def two_shapes():
    return make_params([[5, 5], [15, 15]], [[2, 2], [10, 4]], [0, 0.4], [[1, 0, 0], [0, 0, 1]], [0.6, 0.7])


def set_grad_stats(model, norms, vecs=None):
    model.grad_norm = torch.tensor(norms, dtype=torch.float32)
    model.grad_vec = torch.tensor(vecs, dtype=torch.float32) if vecs else torch.zeros(len(norms), 2)
    model.grad_count = 1


def test_restructure_carries_optimiser_state_and_zeroes_new_rows():
    m = model_of(two_shapes())
    render(m.p, m.bg, 20, 20, 1.0).sum().backward()
    m.opt.step()
    before = {k: m.opt.state[m.p[k]]["exp_avg"].clone() for k in m.p}
    second = m.p["mu"][1].detach().clone()
    new = make_params([[1, 1]], [[1, 1]], [0], [[0, 1, 0]], [0.5])
    m.restructure(torch.tensor([1, 2]), new)  # drop shape 0, keep 1, append new
    assert m.n == 2
    assert torch.equal(m.p["mu"][0].detach(), second)
    for k in m.p:
        state = m.opt.state[m.p[k]]["exp_avg"]
        assert torch.equal(state[0], before[k][1])
        assert torch.count_nonzero(state[1]) == 0
    # The optimiser must step the new tensors.
    render(m.p, m.bg, 20, 20, 1.0).sum().backward()
    old = m.p["mu"].detach().clone()
    m.opt.step()
    assert not torch.equal(old, m.p["mu"].detach())


def test_clone_split_respects_gradient_threshold():
    m = model_of(two_shapes())
    set_grad_stats(m, [1e-6, 1e-6])
    assert density.clone_split(m, 1e-5, 6.0, 10) == (0, 0)
    assert m.n == 2


def test_clone_split_chooses_by_size_and_keeps_order():
    m = model_of(two_shapes())
    set_grad_stats(m, [1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]])
    clones, splits = density.clone_split(m, 1e-5, 6.0, 10)
    assert (clones, splits) == (1, 1)
    assert m.n == 4
    col = torch.sigmoid(m.p["col"]).round()
    # Children occupy the parent's slot: red, red, blue, blue.
    assert col[:, 0].tolist() == [1, 1, 0, 0]
    # Split children lie along the major axis (x rotated by 0.4 rad), either side of the centre.
    mid = (m.p["mu"][2] + m.p["mu"][3]) / 2
    assert torch.allclose(mid, torch.tensor([15.0, 15.0]), atol=1e-4)
    d = (m.p["mu"][3] - m.p["mu"][2]).detach()
    assert math.atan2(d[1], d[0]) == __import__("pytest").approx(0.4, abs=1e-4)
    # Clone moved against the gradient.
    assert m.p["mu"][1][0] < m.p["mu"][0][0]


def test_clone_split_caps_growth_to_highest_gradients():
    m = model_of(two_shapes())
    set_grad_stats(m, [1.0, 2.0])
    assert density.clone_split(m, 1e-5, 6.0, 1) == (0, 1)
    assert m.n == 3


def test_clone_preserves_the_image_where_copies_overlap():
    p = make_params([[10, 10]], [[4, 4]], [0], [[1, 0, 0]], [0.6])
    m = model_of(p)
    # Binary coverage (no supersampling): the identity is exact per pixel.
    before = render_hard(m.p, m.bg, 20, 20, ss=1)
    set_grad_stats(m, [1.0], [[0.0, 0.0]])  # zero direction: copies coincide
    density.clone_split(m, 1e-5, 6.0, 10)
    after = render_hard(m.p, m.bg, 20, 20, ss=1)
    assert torch.allclose(before, after, atol=1e-4)


def square_target(h=40, w=40):
    target = torch.full((3, h, w), 0.5)
    target[:, 10:22, 20:32] = torch.tensor([0.9, 0.1, 0.1])[:, None, None]
    return target


def test_spawn_targets_largest_error_region_with_its_colour():
    target = square_target()
    img = torch.full_like(target, 0.5)
    cands = density.spawn_candidates(img, target, [1, 2, 4, 8], k=1, min_gain=0)
    (x, y, r, c) = cands[0]
    assert 20 <= x <= 32 and 10 <= y <= 22
    assert r > 3
    assert torch.allclose(c, torch.tensor([0.9, 0.1, 0.1]), atol=0.05)


def test_spawn_respects_minimum_gain_and_suppression():
    target = square_target()
    img = torch.full_like(target, 0.5)
    assert density.spawn_candidates(img, target, [1, 2, 4], k=5, min_gain=1e9) == []
    cands = density.spawn_candidates(img, target, [4], k=5, min_gain=0)
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            assert math.hypot(a[0] - b[0], a[1] - b[1]) > a[2]
    # Nothing to gain once the image already matches.
    assert density.spawn_candidates(target, target, [1, 2, 4], k=5, min_gain=0) == []


def test_select_prunable():
    cost = torch.tensor([0.5, 1e-7, -1e-3, 2e-6, 1.0])
    opacity = torch.tensor([0.005, 0.5, 0.5, 0.5, 0.5])
    mask = density.select_prunable(cost, opacity, lam=1e-5, min_opacity=0.01, max_remove=10)
    assert mask.tolist() == [True, True, True, True, False]
    # Capped removals take the cheapest first.
    mask = density.select_prunable(cost, opacity, lam=1e-5, min_opacity=0.01, max_remove=2)
    assert mask.tolist() == [False, True, True, False, False]


def test_greedy_prune_removes_redundant_duplicate_but_not_both():
    # Two identical opaque discs: either alone reproduces the target.
    p = make_params([[10, 10], [10, 10]], [[5, 5], [5, 5]], [0, 0], [[1, 0, 0]] * 2, [0.9999, 0.9999])
    a = alphas(p, pixel_grid(20, 20), None)
    colour, bg = torch.sigmoid(p["col"]), torch.full((3,), 0.5)
    target = composite(a, colour, bg)
    keep = density.greedy_prune(a, colour, bg, target, lam=1e-5)
    assert keep.numel() == 1


def test_greedy_prune_trades_error_for_count():
    # A faint disc that helps a little: kept at small lam, dropped at large lam.
    p = make_params([[10, 10]], [[3, 3]], [0], [[0.6, 0.5, 0.5]], [0.5])
    a = alphas(p, pixel_grid(20, 20), None)
    colour, bg = torch.sigmoid(p["col"]), torch.full((3,), 0.5)
    target = composite(a, colour, bg)
    from splatlisa.render import removal_cost
    saving = removal_cost(a, colour, bg, target).item()
    assert density.greedy_prune(a, colour, bg, target, lam=saving / 2).numel() == 1
    assert density.greedy_prune(a, colour, bg, target, lam=saving * 2).numel() == 0


def test_prune_removes_near_transparent_shapes():
    p = make_params([[10, 10], [5, 5]], [[5, 5], [3, 3]], [0, 0], [[1, 0, 0], [0, 1, 0]], [0.9, 0.001])
    m = model_of(p)
    target = render(m.p, m.bg, 20, 20, 0.5).detach()
    assert density.prune(m, target, 0.5, lam=0.0, min_opacity=0.01, max_remove=5) == 1
    assert m.n == 1
    assert torch.sigmoid(m.p["col"][0, 0]) > 0.9
