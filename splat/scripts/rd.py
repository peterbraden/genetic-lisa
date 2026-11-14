"""Rate-distortion: error against file size, ellipse SVGs and raster codecs.

Each run directory's best.json is written as a minified SVG (`minify.py`),
rasterised by resvg and scored as written; its size is measured raw, gzipped
and brotli-compressed. JPEG, WebP and AVIF are swept over quality with
Pillow. Every MSE is against the decoded source.

    uv run --with matplotlib python scripts/rd.py ../couch-db/_attachments/image/lisa.jpg \
        out/rd/lam-* --out out/rd/rd.png
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from splatlisa.minify import to_min_svg
from splatlisa.score import rasterise

# Reference categorical palette (dataviz skill), slots in fixed order.
SERIES = {"ellipses": "#2a78d6", "jpeg": "#eb6834", "webp": "#1baf7a", "avif": "#eda100"}
SCALES = (1, 0.5, 0.25)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).mean())


def brotli(data: bytes) -> int:
    return len(subprocess.run(["brotli", "-c", "-q", "11"], input=data, capture_output=True, check=True).stdout)


def ellipses(target: np.ndarray, runs: list[Path], dp: int) -> list[dict]:
    h, w, _ = target.shape
    rows = []
    for run in runs:
        params = json.loads((run / "best.json").read_text())
        svg = to_min_svg(params, w, h, dp)
        data = svg.encode()
        lam = json.loads((run / "run.json").read_text())["config"]["lam"]
        rows.append({"run": str(run), "lam": lam, "n": len(params["mu"]), "mse": mse(rasterise(svg, w, h), target),
                     "bytes": len(data), "gzip": len(gzip.compress(data, 9, mtime=0)), "brotli": brotli(data)})
    return sorted(rows, key=lambda r: r["bytes"])


def codec(target: np.ndarray, fmt: str, qualities, **opts) -> list[dict]:
    """Lower envelope over quality and resolution: below a codec's minimum
    size at full resolution, an encoder would downscale (and the viewer
    upscale, here bicubically)."""
    full = Image.fromarray((target * 255).round().astype(np.uint8))
    rows = []
    for scale in SCALES:
        size = (round(full.width * scale), round(full.height * scale))
        src = full if scale == 1 else full.resize(size, Image.LANCZOS)
        for q in qualities:
            buf = io.BytesIO()
            src.save(buf, fmt, quality=q, **opts)
            img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB").resize(full.size, Image.BICUBIC)
            rows.append({"scale": scale, "quality": q, "bytes": buf.tell(),
                         "mse": mse(np.asarray(img, dtype=np.float32) / 255, target)})
    rows.sort(key=lambda r: r["bytes"])
    envelope = []
    for r in rows:
        if not envelope or r["mse"] < envelope[-1]["mse"]:
            envelope.append(r)
    return envelope


def at(rows: list[dict], key: str, size: float) -> float:
    """MSE of a codec at a given size, interpolated in log-log."""
    xs = np.log([r[key] for r in rows])
    ys = np.log([r["mse"] for r in rows])
    if not xs[0] <= np.log(size) <= xs[-1]:
        return float("nan")
    return float(np.exp(np.interp(np.log(size), xs, ys)))


def plot(res: dict, out: Path):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="#fcfcfb")
    e = res["ellipses"]
    for ax in (a, b):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Left: error against ellipse count, labelled with the minified size.
    a.plot([r["n"] for r in e], [r["mse"] for r in e], "-o", color=SERIES["ellipses"], lw=2, ms=5)
    last = 0
    for r in e:
        if r["n"] < last * 1.2:  # neighbours too close to label legibly
            continue
        last = r["n"]
        a.annotate(f"{r['bytes'] / 1000:.1f} / {r['gzip'] / 1000:.1f} kB", (r["n"], r["mse"]), xytext=(5, 4),
                   textcoords="offset points", color=MUTED, fontsize=7.5)
    for ax in (a, b):
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(plt.FuncFormatter(
                lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "25" else ""))
    a.set_xlabel("ellipses")
    a.set_ylabel("MSE against source")
    a.set_title("Error against ellipse count (labels: minified / gzip size)", loc="left", fontsize=10)

    # Right: error against bytes for every encoding.
    b.plot([r["bytes"] for r in e], [r["mse"] for r in e], "-o", color=SERIES["ellipses"], lw=2, ms=5,
           label="ellipse SVG, minified")
    b.plot([r["gzip"] for r in e], [r["mse"] for r in e], "--o", color=SERIES["ellipses"], lw=2, ms=5,
           mfc="#fcfcfb", label="ellipse SVG, gzip")
    for name, label in (("jpeg", "JPEG"), ("webp", "WebP"), ("avif", "AVIF")):
        rows = res[name]
        b.plot([r["bytes"] for r in rows], [r["mse"] for r in rows], "-", color=SERIES[name], lw=2, label=label)
    b.axvline(res["source_bytes"], color=MUTED, lw=0.8, ls=":")
    b.annotate("source JPEG", (res["source_bytes"], b.get_ylim()[1]), xytext=(-4, -12),
               textcoords="offset points", ha="right", color=MUTED, fontsize=7.5)
    b.set_xlabel("file size (bytes)")
    b.set_ylabel("MSE against source")
    b.set_title("Rate-distortion", loc="left", fontsize=10)
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--dp", type=int, default=1, help="decimal places for coordinates")
    ap.add_argument("--out", type=Path, default=Path("out/rd/rd.png"))
    args = ap.parse_args(argv)

    target = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255
    runs = [r for r in args.runs if (r / "best.json").exists()]
    res = {"source_bytes": Path(args.image).stat().st_size,
           "ellipses": ellipses(target, runs, args.dp),
           "jpeg": codec(target, "JPEG", range(2, 96, 3), optimize=True),
           "webp": codec(target, "WEBP", range(0, 101, 4), method=6),
           "avif": codec(target, "AVIF", range(0, 101, 4), speed=0)}

    print(f"{'λ':>8} {'n':>5} {'MSE':>8} {'bytes':>6} {'gzip':>6} {'brotli':>6} {'B/shape':>7} "
          f"{'JPEG':>8} {'WebP':>8} {'AVIF':>8}  (codec MSE at the gzip size)")
    for r in res["ellipses"]:
        print(f"{r['lam']:8.1e} {r['n']:5d} {r['mse']:8.5f} {r['bytes']:6d} {r['gzip']:6d} {r['brotli']:6d} "
              f"{r['bytes'] / r['n']:7.1f} " + " ".join(f"{at(res[c], 'bytes', r['gzip']):8.5f}"
                                                      for c in ("jpeg", "webp", "avif")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    plot(res, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
