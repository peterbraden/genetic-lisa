"""Animate a run's snapshots (frames/*.svg) as a GIF, rasterised by resvg.

    uv run python scripts/animate.py out/mona-large --width 480
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from splatlisa.score import rasterise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="output directory of a training run")
    ap.add_argument("--width", type=int, default=400, help="width of the animation in pixels")
    ap.add_argument("--ms", type=int, default=120, help="milliseconds per frame")
    ap.add_argument("--hold", type=int, default=2000, help="milliseconds on the final frame")
    ap.add_argument("--out", help="default: <run>/progress.gif")
    args = ap.parse_args()

    run = Path(args.run)
    paths = sorted((run / "frames").glob("*.svg"))
    if (run / "best.svg").exists():
        paths.append(run / "best.svg")
    frames = []
    for p in paths:
        svg = p.read_text()
        w, h = (int(v) for v in re.search(r'width="(\d+)" height="(\d+)"', svg).groups())
        height = round(h * args.width / w)
        img = Image.fromarray((rasterise(svg, args.width, height) * 255).round().astype(np.uint8))
        label = "final" if p.name == "best.svg" else f"step {int(p.stem)}"
        label += f"  {svg.count('<ellipse')} ellipses"
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 8 + 6 * len(label), 14), fill=(0, 0, 0))
        draw.text((4, 2), label, fill=(255, 255, 255))
        frames.append(img)
    durations = [args.ms] * (len(frames) - 1) + [args.hold]
    out = args.out or run / "progress.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"{len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
