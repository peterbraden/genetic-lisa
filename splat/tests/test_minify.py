import math

import numpy as np
import pytest
import torch

from splatlisa.minify import colour, num, to_min_svg
from splatlisa.render import inv_sigmoid
from splatlisa.score import rasterise
from splatlisa.svg import to_svg


@pytest.mark.parametrize("v,dp,s", [(0.5, 1, ".5"), (-0.5, 1, "-.5"), (12.0, 1, "12"), (-0.04, 1, "0"),
                                    (3.14159, 2, "3.14"), (100.0, 0, "100"), (0.0, 2, "0")])
def test_num_drops_redundant_characters(v, dp, s):
    assert num(v, dp) == s


def test_colour_folds_opacity_and_shortens():
    assert colour([1, 0, 0]) == "#f00"
    assert colour([1, 0, 0], 1.0) == "#f00"
    assert colour([1, 0, 0], 0.5) == "#ff000080"
    assert colour([1, 1, 2 / 3], 0) == "#ffa0"
    assert colour([26 / 255, 51 / 255, 77 / 255]) == "#1a334d"


# Each case exercises one branch of the angle normalisation: general, rx < ry,
# a half turn, zero and right angles (no transform), and a circle.
CASES = [(25, 40, 18, 9, 30), (25, 40, 9, 18, 30), (25, 40, 18, 9, 210), (25, 40, 18, 9, 0),
         (25, 40, 18, 9, 90), (25, 40, 9, 18, -90), (25, 40, 12, 12, 47), (-3, 70, 20, 6, 135)]


def params(cases):
    t = torch.tensor(cases, dtype=torch.float64)
    return {"mu": t[:, :2].tolist(), "scale": t[:, 2:4].tolist(), "theta": torch.deg2rad(t[:, 4]).tolist(),
            "colour": [[0.9, 0.2, 0.1], [0.1, 0.8, 0.3], [0.2, 0.3, 0.9], [0.8, 0.8, 0.1],
                       [0.5, 0.1, 0.6], [0.1, 0.6, 0.6], [1.0, 1.0, 1.0], [0.3, 0.3, 0.3]][:len(cases)],
            "opacity": [0.7] * len(cases), "background": [0.05, 0.05, 0.1]}


def full_svg(q, w, h):
    p = {"mu": torch.tensor(q["mu"]), "log_s": torch.tensor(q["scale"]).log(), "theta": torch.tensor(q["theta"]),
         "col": inv_sigmoid(torch.tensor(q["colour"]).clamp(1e-6, 1 - 1e-6)),
         "op": inv_sigmoid(torch.tensor(q["opacity"]))}
    return to_svg(p, inv_sigmoid(torch.tensor(q["background"])), w, h)


@pytest.mark.parametrize("i", range(len(CASES)))
def test_minified_ellipse_renders_as_original(i):
    w, h = 60, 90
    q = params([CASES[i]])
    a = rasterise(full_svg(q, w, h), w, h)
    b = rasterise(to_min_svg(q, w, h, dp=2), w, h)
    # Rounding (8-bit opacity) and the rasteriser's edge flattening, which
    # differs between <circle> and <ellipse>, alter a few boundary pixels; a
    # wrong orientation or exchanged axes alters several percent.
    assert (np.abs(a - b) > 2 / 255).mean() < 0.005


def test_minified_stack_preserves_painting_order():
    w, h = 60, 90
    q = params(CASES)
    a = rasterise(full_svg(q, w, h), w, h)
    b = rasterise(to_min_svg(q, w, h, dp=2), w, h)
    assert np.abs(a - b).mean() < 1e-3
