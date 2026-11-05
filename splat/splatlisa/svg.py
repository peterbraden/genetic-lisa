"""SVG export of an ellipse set."""

from __future__ import annotations

import torch
from torch import Tensor

from .render import degrees, natural


def _hex(c: Tensor) -> str:
    r, g, b = (int(round(float(v) * 255)) for v in c.clamp(0, 1))
    return f"#{r:02x}{g:02x}{b:02x}"


def to_svg(p: dict[str, Tensor], bg_logit: Tensor, w: int, h: int, out_w: int | None = None, out_h: int | None = None) -> str:
    """Shapes in painting order; coordinates are in source pixels, so the
    output can be scaled freely via width/height."""
    q = {k: v.detach().cpu() for k, v in natural(p).items()}
    bg = torch.sigmoid(bg_logit.detach().cpu())
    out_w, out_h = out_w or w, out_h or h
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{out_w}" height="{out_h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="{_hex(bg)}"/>',
    ]
    for i in range(q["mu"].shape[0]):
        cx, cy = (float(v) for v in q["mu"][i])
        rx, ry = (float(v) for v in q["scale"][i])
        deg = float(degrees(q["theta"][i])) % 360
        lines.append(
            f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" '
            f'transform="rotate({deg:.1f} {cx:.2f} {cy:.2f})" '
            f'fill="{_hex(q["colour"][i])}" fill-opacity="{float(q["opacity"][i]):.3f}"/>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"
