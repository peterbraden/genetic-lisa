"""Low-rate comparison with perceptual measures, including a packed binary
ellipse format estimated by `splatlisa/packed.py`.

For each run, bits are removed greedily from whichever field group costs the
least MSE per bit saved, from generous widths down to a floor; every state is
a (size, image) point. Points are scored by MSE, DSSIM (1 - SSIM) and LPIPS
(AlexNet), and each encoding is reduced to its lower envelope per measure.
Codecs are swept over quality and resolution (1 to 1/8, upscaled
bicubically), as in rd.py.

    uv run --with matplotlib --with lpips python scripts/lowrate.py \
        ../couch-db/_attachments/image/lisa.jpg out/rd/lam-* --out out/rd/lowrate.png
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from splatlisa.loss import ssim
from splatlisa.minify import to_min_svg
from splatlisa.packed import bits_per_ellipse, quantise, size, to_svg_natural
from splatlisa.score import rasterise

SERIES = {"ellipses": "#2a78d6", "jpeg": "#eb6834", "webp": "#1baf7a", "avif": "#eda100"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MEASURES = {"mse": "MSE", "dssim": "DSSIM (1 − SSIM)", "lpips": "LPIPS"}
CODECS = {"jpeg": ("JPEG", range(2, 96, 3), {"optimize": True}),
          "webp": ("WEBP", range(0, 101, 4), {"method": 6}),
          "avif": ("AVIF", range(0, 101, 4), {"speed": 0})}
SCALES = (1, 0.5, 0.25, 0.125)
START = {"pos": 10, "size": 8, "angle": 8, "colour": 8, "alpha": 8}
FLOOR = {"pos": 4, "size": 3, "angle": 2, "colour": 2, "alpha": 1}


class Scorer:
    def __init__(self, target: np.ndarray):
        import lpips

        self.target = target
        self.t = torch.from_numpy(target).permute(2, 0, 1).contiguous()
        self.net = lpips.LPIPS(net="alex", verbose=False)

    def mse(self, img: np.ndarray) -> float:
        return float(((img - self.target) ** 2).mean())

    def __call__(self, img: np.ndarray) -> dict:
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        with torch.no_grad():
            lp = float(self.net(x[None] * 2 - 1, self.t[None] * 2 - 1))
        return {"mse": self.mse(img), "dssim": 1 - float(ssim(x, self.t)), "lpips": lp}


def svg_rows(scorer: Scorer, runs: list[Path], w: int, h: int) -> list[dict]:
    rows = []
    for run in runs:
        params = json.loads((run / "best.json").read_text())
        svg = to_min_svg(params, w, h, 1)
        rows.append({"run": str(run), "n": len(params["mu"]), "bytes": len(gzip.compress(svg.encode(), 9, mtime=0)),
                     **scorer(rasterise(svg, w, h))})
    return rows


def packed_rows(scorer: Scorer, runs: list[Path], w: int, h: int) -> list[dict]:
    rows = []
    for run in runs:
        params = json.loads((run / "best.json").read_text())
        n = len(params["mu"])

        def image(bits):
            return rasterise(to_svg_natural(quantise(params, w, h, bits), w, h), w, h)

        bits = dict(START)
        while True:
            rows.append({"run": str(run), "n": n, "bits": dict(bits), "bytes": size(n, bits), **scorer(image(bits))})
            trials = []
            for g in bits:
                if bits[g] > FLOOR[g]:
                    b = {**bits, g: bits[g] - 1}
                    saved = bits_per_ellipse(bits) - bits_per_ellipse(b)
                    trials.append(((scorer.mse(image(b)) - rows[-1]["mse"]) / saved, b))
            if not trials:
                break
            bits = min(trials, key=lambda t: t[0])[1]
    return rows


def encode(img: Image.Image, fmt: str, q: int, opts: dict) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, quality=q, **opts)
    return buf.getvalue()


def codec_rows(scorer: Scorer, name: str, max_bytes: int) -> list[dict]:
    fmt, qualities, opts = CODECS[name]
    # From an array, so no colour profile from the source file is embedded.
    full = Image.fromarray((scorer.target * 255).round().astype(np.uint8))
    rows = []
    for scale in SCALES:
        src = full if scale == 1 else full.resize((round(full.width * scale), round(full.height * scale)),
                                                  Image.LANCZOS)
        for q in qualities:
            data = encode(src, fmt, q, opts)
            if len(data) > max_bytes:
                continue
            img = Image.open(io.BytesIO(data)).convert("RGB").resize(full.size, Image.BICUBIC)
            rows.append({"scale": scale, "quality": q, "bytes": len(data),
                         **scorer(np.asarray(img, dtype=np.float32) / 255)})
    return rows


def envelope(rows: list[dict], key: str) -> list[dict]:
    out = []
    for r in sorted(rows, key=lambda r: (r["bytes"], r[key])):
        if not out or r[key] < out[-1][key]:
            out.append(r)
    return out


def best_within(rows: list[dict], key: str, budget: int) -> dict | None:
    fits = [r for r in rows if r["bytes"] <= budget]
    return min(fits, key=lambda r: r[key]) if fits else None


def plot(res: dict, out: Path, max_bytes: int):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE)
    for ax, (key, label) in zip(axes, MEASURES.items()):
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        e = envelope(res["packed"], key)
        ax.plot([r["bytes"] for r in e], [r[key] for r in e], "-", color=SERIES["ellipses"], lw=2,
                label="ellipses, packed (estimate)")
        s = envelope(res["svg"], key)
        ax.plot([r["bytes"] for r in s], [r[key] for r in s], "--o", color=SERIES["ellipses"], lw=1.5, ms=4,
                mfc=SURFACE, label="ellipses, SVG minified + gzip")
        for name in CODECS:
            c = envelope(res[name], key)
            ax.plot([r["bytes"] for r in c], [r[key] for r in c], "-", color=SERIES[name], lw=2,
                    label=name.upper() if name != "webp" else "WebP")
        ax.set_xlim(100, max_bytes)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "25" else ""))
        ax.set_xlabel("file size (bytes)")
        ax.set_title(label + ", lower is better", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def compare(res: dict, scorer: Scorer, budgets: list[int], w: int, h: int, out: Path):
    """For each budget, the best image by LPIPS of each encoding within it."""
    full = Image.fromarray((scorer.target * 255).round().astype(np.uint8))
    label_h = 28
    sheet = Image.new("RGB", (w * 5, (h + label_h) * len(budgets)), "white")
    d = ImageDraw.Draw(sheet)
    for i, budget in enumerate(budgets):
        tiles = [(f"source; budget {budget} B", full)]
        r = best_within(res["packed"], "lpips", budget)
        params = json.loads((Path(r["run"]) / "best.json").read_text())
        img = rasterise(to_svg_natural(quantise(params, w, h, r["bits"]), w, h), w, h)
        tiles.append((f"{r['n']} ellipses packed {r['bytes']} B\nLPIPS {r['lpips']:.3f}",
                      Image.fromarray((img * 255).round().astype(np.uint8))))
        for name in CODECS:
            r = best_within(res[name], "lpips", budget)
            if r is None:
                tiles.append((f"{name.upper()}: none this small", Image.new("RGB", (w, h), "#dddddd")))
                continue
            fmt, _, opts = CODECS[name]
            src = full if r["scale"] == 1 else full.resize((round(w * r["scale"]), round(h * r["scale"])),
                                                           Image.LANCZOS)
            dec = Image.open(io.BytesIO(encode(src, fmt, r["quality"], opts))).convert("RGB").resize((w, h),
                                                                                                    Image.BICUBIC)
            tiles.append((f"{name.upper()} {r['bytes']} B ({r['scale']:g}x)\nLPIPS {r['lpips']:.3f}", dec))
        for j, (label, im) in enumerate(tiles):
            y = i * (h + label_h)
            sheet.paste(im, (j * w, y + label_h))
            d.multiline_text((j * w + 3, y + 2), label, fill="black", spacing=2)
    sheet.save(out)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--max-bytes", type=int, default=4000)
    ap.add_argument("--budgets", type=int, nargs="+", default=[250, 500, 1000, 2000])
    ap.add_argument("--out", type=Path, default=Path("out/rd/lowrate.png"))
    args = ap.parse_args(argv)

    target = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255
    h, w, _ = target.shape
    scorer = Scorer(target)
    runs = [r for r in args.runs if (r / "best.json").exists()]
    res = {"svg": svg_rows(scorer, runs, w, h), "packed": packed_rows(scorer, runs, w, h),
           **{name: codec_rows(scorer, name, args.max_bytes) for name in CODECS}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=1))

    encodings = ["packed", "svg", *CODECS]
    for key in MEASURES:
        print(f"\n{MEASURES[key]}: best within each budget")
        print(f"{'budget':>7} " + " ".join(f"{e:>8}" for e in encodings))
        for b in args.budgets:
            cells = [best_within(res[e], key, b) for e in encodings]
            print(f"{b:7d} " + " ".join(f"{c[key]:8.4f}" if c else f"{'-':>8}" for c in cells))
    print("\npacked, best LPIPS within each budget:")
    for b in args.budgets:
        r = best_within(res["packed"], "lpips", b)
        print(f"{b:7d} B: {r['n']} ellipses, {bits_per_ellipse(r['bits'])} bits each {r['bits']}, {r['bytes']} B")
    plot(res, args.out, args.max_bytes)
    compare(res, scorer, args.budgets, w, h, args.out.with_name(args.out.stem + "-compare.png"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
