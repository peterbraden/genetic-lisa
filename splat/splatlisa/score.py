"""Score any SVG against a source image: shape count, MSE, fitness.

Rasterises with resvg, independently of the training renderer, so results
from different implementations are comparable.
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import numpy as np
import resvg_py
from PIL import Image

SHAPE = re.compile(r"<(ellipse|circle|polygon|rect|path)\b")


def rasterise(svg: str, width: int, height: int, background: str = "#000000") -> np.ndarray:
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height, background=background))
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32) / 255


def score(svg_path: str, image_path: str, lam: float, background: str = "#000000") -> dict:
    svg = Path(svg_path).read_text()
    target = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255
    h, w, _ = target.shape
    img = rasterise(svg, w, h, background)
    # A full-canvas background <rect> is not counted as a shape.
    n = len(SHAPE.findall(svg)) - len(re.findall(r'<rect width="\d+" height="\d+"', svg))
    m = float(((img - target) ** 2).mean())
    return {"svg": svg_path, "n": n, "mse": m, "fitness": m + lam * n}


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("svgs", nargs="+")
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--background", default="#000000", help="canvas colour for SVGs that set it via CSS")
    args = ap.parse_args(argv)
    for s in args.svgs:
        r = score(s, args.image, args.lam, args.background)
        print(f"{r['n']:5d} shapes  mse={r['mse']:.5f}  fitness={r['fitness']:.5f}  {s}")


if __name__ == "__main__":
    main()
