"""Describe the shapes in SVGs: size, opacity, value and stacking depth.

Works on any SVG of simple shapes (including the evolutionary results), by
rasterising each shape alone with resvg for its coverage, and the drawing
without it for its value: the MSE increase on removing it.

    uv run python scripts/shapes.py ../couch-db/_attachments/image/lisa.jpg ../images/36m.svg out/best.svg
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

from splatlisa.score import rasterise

ELEMENT = re.compile(r"<(ellipse|circle|polygon|rect)\b[^>]*/>")
FILL = re.compile(r"""\sfill=(['"])(.*?)\1""")
FILL_OPACITY = re.compile(r"""\sfill-opacity=(['"])(.*?)\1""")
AREA_BINS = [0, 0.001, 0.01, 0.1, np.inf]           # fraction of the canvas
AREA_NAMES = ["<0.1%", "0.1-1%", "1-10%", ">10%"]
OPACITY_BINS = [0, 0.25, 0.5, 0.75, 1.01]
OPACITY_NAMES = ["<0.25", "0.25-0.5", "0.5-0.75", ">0.75"]


def opacity(el: str) -> float:
    if m := FILL_OPACITY.search(el):
        return float(m.group(2))
    if (m := FILL.search(el)) and m.group(2).startswith("rgba"):
        return float(m.group(2).rstrip(")").split(",")[-1])
    return 1.0


def split(svg: str, w: int, h: int) -> tuple[list[str], list[tuple[int, int]]]:
    """Shapes and their spans, excluding a full-canvas background rect."""
    out, spans = [], []
    for m in ELEMENT.finditer(svg):
        el = m.group(0)
        if el.startswith("<rect") and f'width="{w}" height="{h}"' in el and " x=" not in el:
            continue
        out.append(el)
        spans.append(m.span())
    return out, spans


def describe(svg_path: str, target: np.ndarray) -> dict:
    h, w, _ = target.shape
    svg = Path(svg_path).read_text()
    shapes, spans = split(svg, w, h)
    head = svg[: svg.index(">", svg.index("<svg")) + 1]
    base = float(((rasterise(svg, w, h) - target) ** 2).mean())

    area, opac, value, covers = [], [], [], []
    for el, (a, b) in zip(shapes, spans):
        white = FILL_OPACITY.sub("", FILL.sub(' fill="#fff"', el))
        cov = rasterise(f"{head}{white}</svg>", w, h)[..., 0]
        without = rasterise(svg[:a] + svg[b:], w, h)
        area.append(float(cov.mean()))
        opac.append(opacity(el))
        value.append(float(((without - target) ** 2).mean()) - base)
        covers.append(cov > 0.5)
    area, opac, value = map(np.array, (area, opac, value))
    depth = np.sum(covers, axis=0) if covers else np.zeros((h, w))
    return {"svg": svg_path, "n": len(shapes), "mse": base, "area": area, "opacity": opac,
            "value": value, "depth": depth}


def report(d: dict):
    a, o, v = d["area"], d["opacity"], d["value"]
    print(f"\n{d['svg']}: {d['n']} shapes, MSE {d['mse']:.5f}")
    print(f"  area (fraction of canvas) p10/p50/p90 = {np.percentile(a, 10):.4f} / "
          f"{np.median(a):.4f} / {np.percentile(a, 90):.4f}")
    print(f"  opacity p10/p50/p90 = {np.percentile(o, 10):.2f} / {np.median(o):.2f} / {np.percentile(o, 90):.2f}")
    print(f"  shapes covering a pixel (coverage > 0.5): mean {d['depth'].mean():.1f}, "
          f"p90 {np.percentile(d['depth'], 90):.0f}, max {d['depth'].max()}")
    # Count, and share of total removal value, by area and opacity.
    ai = np.digitize(a, AREA_BINS) - 1
    oi = np.digitize(o, OPACITY_BINS) - 1
    total = max(v.clip(min=0).sum(), 1e-12)
    print(f"  {'area / opacity':14s}" + "".join(f"{n:>16s}" for n in OPACITY_NAMES))
    for i, name in enumerate(AREA_NAMES):
        cells = []
        for j in range(len(OPACITY_NAMES)):
            sel = (ai == i) & (oi == j)
            cells.append(f"{sel.sum():5d} {100 * v[sel].clip(min=0).sum() / total:5.1f}% val")
        print(f"  {name:14s}" + "".join(f"{c:>16s}" for c in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("svgs", nargs="+")
    args = ap.parse_args()
    target = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255
    for s in args.svgs:
        report(describe(s, target))


if __name__ == "__main__":
    main()
