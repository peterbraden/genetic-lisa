import math

import torch

from splatlisa import density
from splatlisa.model import Model
from splatlisa.render import alphas, composite, hard_pairs, inv_sigmoid, make_params, pixel_grid, render, render_hard


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
    x, y, r, c, _ = cands[0]
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
    keep = density.greedy_prune(hard_pairs(p, 20, 20, ss=1), colour, bg, target, lam=1e-5)
    assert keep.numel() == 1


def test_greedy_prune_trades_error_for_count():
    # A faint disc that helps a little: kept at small lam, dropped at large lam.
    p = make_params([[10, 10]], [[3, 3]], [0], [[0.6, 0.5, 0.5]], [0.5])
    a = alphas(p, pixel_grid(20, 20), None)
    colour, bg = torch.sigmoid(p["col"]), torch.full((3,), 0.5)
    target = composite(a, colour, bg)
    from splatlisa.render import removal_cost
    saving = removal_cost(a, colour, bg, target).item()
    pairs = hard_pairs(p, 20, 20, ss=1)
    assert density.greedy_prune(pairs, colour, bg, target, lam=saving / 2).numel() == 1
    assert density.greedy_prune(pairs, colour, bg, target, lam=saving * 2).numel() == 0


def test_prune_removes_near_transparent_shapes():
    p = make_params([[10, 10], [5, 5]], [[5, 5], [3, 3]], [0, 0], [[1, 0, 0], [0, 1, 0]], [0.9, 0.001])
    m = model_of(p)
    target = render(m.p, m.bg, 20, 20, 0.5).detach()
    assert density.prune(m, target, 0.5, lam=0.0, min_opacity=0.01, max_remove=5) == 1
    assert m.n == 1
    assert torch.sigmoid(m.p["col"][0, 0]) > 0.9


def test_select_relocations_pairs_cheapest_shapes_with_best_sites():
    cost = torch.tensor([5e-4, 1e-5, 2e-4, -1e-6])
    # Best site beats shape 3 (harmful) and shape 1, but not shape 2.
    assert density.select_relocations(cost, [1e-3, 5e-5, 1e-4], margin=1.0) == [(3, 0), (1, 1)]
    # A higher margin rejects the marginal swap.
    assert density.select_relocations(cost, [1e-3, 5e-5], margin=10.0) == [(3, 0)]
    assert density.select_relocations(cost, [], margin=1.0) == []


def test_relocate_moves_a_useless_shape_to_the_error():
    # A grey disc on grey background contributes nothing; the red square is unexplained.
    target = square_target()
    p = make_params([[5, 30]], [[3, 3]], [0], [[0.5, 0.5, 0.5]], [0.8])
    m = model_of(p)
    img = render(m.p, m.bg, 40, 40, 0.5).detach()
    assert density.relocate(m, img, target, 0.5, [2, 4, 8], k=2) == 1
    assert m.n == 1
    x, y = m.p["mu"][0].tolist()
    assert 20 <= x <= 32 and 10 <= y <= 22
    assert torch.sigmoid(m.p["col"][0, 0]) > 0.8


def test_relocate_keeps_shapes_that_earn_more_than_any_site():
    target = square_target()
    p = make_params([[26, 16]], [[6.5, 6.5]], [0], [[0.9, 0.1, 0.1]], [0.95])
    m = model_of(p)
    img = render(m.p, m.bg, 40, 40, 0.5).detach()
    before = m.p["mu"].detach().clone()
    assert density.relocate(m, img, target, 0.5, [2, 4, 8], k=2) == 0
    assert torch.equal(m.p["mu"].detach(), before)


def test_reorder_moves_occluding_shape_beneath():
    # Target: small red disc visible on a large blue disc. The model paints them
    # the wrong way round, so the blue disc hides the red one.
    red = ([[20, 20]], [[4, 4]], [[0.9, 0.1, 0.1]])
    blue = ([[20, 20]], [[12, 12]], [[0.1, 0.1, 0.9]])
    right = make_params(blue[0] + red[0], blue[1] + red[1], [0, 0], blue[2] + red[2], [0.95, 0.95])
    wrong = make_params(red[0] + blue[0], red[1] + blue[1], [0, 0], red[2] + blue[2], [0.95, 0.95])
    target = render(right, inv_sigmoid(torch.full((3,), 0.5)), 40, 40, 0.5).detach()
    m = model_of(wrong)
    assert density.reorder(m, target, 0.5, max_moves=3, min_gain=1e-7) == 1
    # Red now on top.
    assert torch.sigmoid(m.p["col"][-1, 0]) > 0.8
    # Already optimal: no further moves.
    assert density.reorder(m, target, 0.5, max_moves=3, min_gain=1e-7) == 0


def textured(h=40, w=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 0.2 + 0.6 * torch.rand(3, h, w, generator=g)


def test_glaze_recovers_translucent_layer_over_texture():
    # Target: the current image under a 20% red wash in one region. The
    # opaque rule would erase the texture; the glaze fits the wash exactly.
    img = textured()
    red = torch.tensor([0.9, 0.1, 0.1])[:, None, None]
    target = img.clone()
    target[:, 8:32, 8:32] = 0.8 * img[:, 8:32, 8:32] + 0.2 * red
    x, y, r, c, gain, a = density.glaze_candidates(img, target, [8], k=1, min_gain=0)[0]
    assert 8 <= x <= 32 and 8 <= y <= 32
    assert abs(a - 0.2) < 0.02
    assert torch.allclose(c, red.flatten(), atol=0.05)
    i, j = int(y), int(x)
    box_error = float(((target - img)[:, i - 8:i + 9, j - 8:j + 9] ** 2).sum())
    assert abs(gain - box_error) < 1e-3 * box_error
    opaque = density.spawn_candidates(img, target, [8], k=1, min_gain=0)
    assert not opaque or opaque[0][4] < 0.5 * gain


def test_glaze_on_flat_image_keeps_colour_in_range():
    # No structure beneath, so opacity is set by feasibility: a lighter grey
    # wash over grey needs a >= 0.1 / 0.5.
    img = torch.full((3, 40, 40), 0.5)
    target = img.clone()
    target[:, 10:30, 10:30] = 0.6
    _, _, _, c, _, a = density.glaze_candidates(img, target, [4], k=1, min_gain=0)[0]
    assert a >= 0.2 - 1e-4
    assert ((c >= 0) & (c <= 1)).all()
    assert torch.allclose(a * c + (1 - a) * 0.5, torch.full((3,), 0.6), atol=0.01)
    assert density.glaze_candidates(target, target, [4], k=1, min_gain=0) == []


def test_glaze_appends_rows_with_fitted_opacity():
    img = textured()
    target = 0.7 * img + 0.3 * torch.tensor([0.1, 0.1, 0.9])[:, None, None]
    model = model_of(two_shapes())
    added = density.glaze(model, img, target, [8, 16], k=2, min_gain=0)
    assert added >= 1 and model.n == 2 + added
    assert abs(float(torch.sigmoid(model.p["op"][2].detach())) - 0.3) < 0.03


def test_rescale_preserves_hard_coverage_exactly():
    # 4x4 samples at (h, w) coincide with 2x2 samples at (2h, 2w).
    from splatlisa.render import hard_alphas
    model = model_of(two_shapes())
    before = hard_alphas(model.detached(), 20, 24, ss=4)
    model.rescale(2.0)
    after = hard_alphas(model.detached(), 40, 48, ss=2)
    pooled = torch.nn.functional.avg_pool2d(after[:, None], 2)[:, 0]
    assert torch.allclose(pooled, before, atol=1e-6)


def test_rescale_makes_adam_steps_scale_with_the_image():
    import copy
    c = torch.tensor([[3.0, 4.0], [11.0, 9.0]])

    def step(model, f):
        mu, log_s = model.p["mu"] / f, model.p["log_s"] - math.log(f)
        loss = ((mu - c) ** 2 * torch.tensor([1.0, 3.0])).sum() + (log_s**2).sum()
        model.opt.zero_grad()
        loss.backward()
        model.opt.step()

    a = model_of(two_shapes())
    for _ in range(3):
        step(a, 1.0)
    b = copy.deepcopy(a)
    f = 2.5
    b.rescale(f)
    b.set_lr("mu", a.lr["mu"] * f)
    step(a, 1.0)
    step(b, f)
    assert torch.allclose(b.p["mu"], f * a.p["mu"], atol=1e-4)
    assert torch.allclose(b.p["log_s"], a.p["log_s"] + math.log(f), atol=1e-5)
    assert b.grad_count == 0
