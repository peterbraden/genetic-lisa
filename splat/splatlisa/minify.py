"""Compact SVG export, for measuring the representation as a compressed image.

Differs from `svg.to_svg` only in encoding: numbers are rounded to `dp`
decimal places (angles to whole degrees, opacity to 8 bits) and written
without redundant zeros; opacity is folded into an 8-digit hex colour; the
angle is normalised so that unrotated ellipses need no transform. The
rounding changes the image slightly, so the result must be scored as written.
"""

from __future__ import annotations

import math


def num(v: float, dp: int) -> str:
    s = f"{round(v, dp):.{dp}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


def pair(a: str, b: str) -> str:
    """Two numbers in a transform list; a minus sign is itself a separator."""
    return a + ("" if b.startswith("-") else " ") + b


def colour(rgb: list[float], alpha: float = 1.0) -> str:
    v = [min(255, max(0, round(c * 255))) for c in (*rgb, alpha)]
    if v[3] == 255:
        v = v[:3]
    if all(c % 17 == 0 for c in v):
        return "#" + "".join(f"{c // 17:x}" for c in v)
    return "#" + "".join(f"{c:02x}" for c in v)


def ellipse(cx: float, cy: float, rx: float, ry: float, theta: float, rgb: list[float], alpha: float, dp: int) -> str:
    # An ellipse is unchanged by a half turn, or by a quarter turn with its
    # axes exchanged; choose rx >= ry and an angle in [0, 180).
    deg = math.degrees(theta)
    if ry > rx:
        rx, ry, deg = ry, rx, deg + 90
    deg = round(deg) % 180
    fill = f' fill="{colour(rgb, alpha)}"/>'
    x, y, a, b = num(cx, dp), num(cy, dp), num(rx, dp), num(ry, dp)
    if a == b:
        return f'<circle cx="{x}" cy="{y}" r="{a}"{fill}'
    if deg == 0:
        return f'<ellipse cx="{x}" cy="{y}" rx="{a}" ry="{b}"{fill}'
    if deg == 90:
        return f'<ellipse cx="{x}" cy="{y}" rx="{b}" ry="{a}"{fill}'
    return f'<ellipse transform="translate({pair(x, y)})rotate({deg})" rx="{a}" ry="{b}"{fill}'


def to_min_svg(params: dict, w: int, h: int, dp: int = 1) -> str:
    """`params` as written to best.json: natural values, angles in radians."""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
           f'<rect width="{w}" height="{h}" fill="{colour(params["background"])}"/>']
    for (cx, cy), (rx, ry), t, c, o in zip(params["mu"], params["scale"], params["theta"],
                                           params["colour"], params["opacity"]):
        out.append(ellipse(cx, cy, rx, ry, t, c, o, dp))
    out.append("</svg>")
    return "".join(out)
