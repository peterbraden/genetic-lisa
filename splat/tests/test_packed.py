import math

import numpy as np

from splatlisa.packed import HEADER, quantise, size, to_svg_natural
from splatlisa.score import rasterise

W, H = 60, 90
FINE = {"pos": 16, "size": 16, "angle": 16, "colour": 8, "alpha": 8}


def params(cases, opacity=0.7):
    t = np.array(cases, dtype=np.float64)
    n = len(cases)
    return {"mu": t[:, :2].tolist(), "scale": t[:, 2:4].tolist(), "theta": np.radians(t[:, 4]).tolist(),
            "colour": [[0.9, 0.2, 0.1]] * n, "opacity": [opacity] * n, "background": [0.05, 0.05, 0.1]}


def test_size_counts_header_and_whole_bytes():
    bits = {"pos": 7, "size": 6, "angle": 5, "colour": 4, "alpha": 3}  # 14 + 12 + 5 + 12 + 3 = 46
    assert size(0, bits) == HEADER
    assert size(1, bits) == HEADER + 6
    assert size(16, bits) == HEADER + 92


def test_fine_quantisation_renders_as_original():
    # rx < ry and an angle outside [0, 180) exercise the normalisation.
    q = params([(25, 40, 18, 9, 30), (30, 50, 6, 20, 250), (-3, 70, 20, 6, -45)])
    a = rasterise(to_svg_natural(q, W, H), W, H)
    b = rasterise(to_svg_natural(quantise(q, W, H, FINE), W, H), W, H)
    assert (np.abs(a - b) > 2 / 255).mean() < 0.005


def test_coarse_fields_round_to_their_grids():
    bits = {"pos": 4, "size": 3, "angle": 2, "colour": 2, "alpha": 2}
    q = quantise(params([(25, 40, 18, 9, 179), (500, -100, 0.1, 0.1, 0)], opacity=0.01), W, H, bits)
    # The angle wraps: 179 degrees is nearer 180 (= 0) than 135.
    assert q["theta"][0] == 0
    # Positions and sizes outside the range clamp to its ends.
    assert q["mu"][1] == [1.25 * W, -0.25 * H]
    assert math.isclose(q["scale"][1][0], 0.5)
    # Opacity is never zero; colours land on 2-bit levels.
    assert q["opacity"][1] == 0.25
    assert all(c * 3 == round(c * 3) for c in q["colour"][0])
    step = 1.5 * W / 15
    assert abs(q["mu"][0][0] - 25) <= step / 2 + 1e-9
