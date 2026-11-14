"""Size and fidelity of a hypothetical packed binary ellipse format.

Not a format: no bitstream is written. Each field is quantised uniformly to a
given number of bits over a fixed range, so that the image a decoder would
show can be rendered and scored, and the size counted exactly as

    HEADER + ceil(N * bits_per_ellipse / 8)

Fields, per ellipse, after normalising to rx >= ry and an angle in [0, 180):
  pos     cx, cy over [-1/4, 5/4] of the width and height
  size    log2 rx, log2 ry over [log2 1/2, log2 (2 * longest side)]
  angle   [0, 180), wrapping
  colour  r, g, b over [0, 1]
  alpha   (k + 1) / 2^bits: zero opacity is never useful
The header holds width and height (2 bytes each), the background colour (3),
the count (2) and the five bit widths (4 bits each, 3 bytes).
"""

from __future__ import annotations

import math

import numpy as np

HEADER = 12
# Values per field in each group; bits are per value.
GROUPS = {"pos": 2, "size": 2, "angle": 1, "colour": 3, "alpha": 1}


def bits_per_ellipse(bits: dict[str, int]) -> int:
    return sum(GROUPS[g] * b for g, b in bits.items())


def size(n: int, bits: dict[str, int]) -> int:
    return HEADER + math.ceil(n * bits_per_ellipse(bits) / 8)


def uniform(v: np.ndarray, lo: float, hi: float, b: int) -> np.ndarray:
    """Round to the nearest of 2^b levels spanning [lo, hi], clamping."""
    k = np.clip(np.round((v - lo) / (hi - lo) * (2**b - 1)), 0, 2**b - 1)
    return lo + k * (hi - lo) / (2**b - 1)


def quantise(params: dict, w: int, h: int, bits: dict[str, int]) -> dict:
    """`params` as in best.json; returns the same structure, as decoded."""
    mu = np.asarray(params["mu"], dtype=np.float64)
    s = np.asarray(params["scale"], dtype=np.float64)
    deg = np.degrees(np.asarray(params["theta"], dtype=np.float64))
    swap = s[:, 1] > s[:, 0]
    s = np.where(swap[:, None], s[:, ::-1], s)
    deg = np.mod(deg + 90 * swap, 180)

    dims = np.array([w, h], dtype=np.float64)
    mu = uniform(mu / dims, -0.25, 1.25, bits["pos"]) * dims
    s = 2 ** uniform(np.log2(s), -1, math.log2(2 * max(w, h)), bits["size"])
    step = 180 / 2 ** bits["angle"]
    deg = np.mod(np.round(deg / step), 2 ** bits["angle"]) * step
    col = uniform(np.asarray(params["colour"], dtype=np.float64), 0, 1, bits["colour"])
    k = np.clip(np.round(np.asarray(params["opacity"]) * 2 ** bits["alpha"] - 1), 0, 2 ** bits["alpha"] - 1)
    return {"mu": mu.tolist(), "scale": s.tolist(), "theta": np.radians(deg).tolist(), "colour": col.tolist(),
            "opacity": ((k + 1) / 2 ** bits["alpha"]).tolist(),
            "background": uniform(np.asarray(params["background"]), 0, 1, 8).tolist()}


def to_svg_natural(params: dict, w: int, h: int) -> str:
    """`svg.to_svg` from best.json-style values (2 decimals, 0.1 degree)."""
    import torch

    from .render import inv_sigmoid
    from .svg import to_svg

    def logit(v):
        return inv_sigmoid(torch.tensor(v, dtype=torch.float32).clamp(1e-6, 1 - 1e-6))

    p = {"mu": torch.tensor(params["mu"], dtype=torch.float32).reshape(-1, 2),
         "log_s": torch.tensor(params["scale"], dtype=torch.float32).reshape(-1, 2).log(),
         "theta": torch.tensor(params["theta"], dtype=torch.float32),
         "col": logit(params["colour"]).reshape(-1, 3), "op": logit(params["opacity"])}
    return to_svg(p, logit(params["background"]), w, h)
