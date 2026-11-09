"""Error against processing time for several runs, one curve each.

Every snapshot is rasterised by resvg at the source's size, so all curves
share one renderer. Splat runs need snapshot_every == log_every (for the
times); genetic runs are directories written by run_genetic.py, whose own
log is also converted to MSE (sum of squared 0-255 differences).

    uv run --with matplotlib python scripts/curves.py ../couch-db/_attachments/image/lisa.jpg \
        defaults=out/curves/defaults genetic=out/genetic --out out/curves.png
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from splatlisa.score import SHAPE, rasterise

FITTEST = re.compile(r"^(\S+ \S+) UTC New fittest: \[F:\S+ - ([\d.]+) \((\d+) shap")
SLEEP = re.compile(r"^(\S+ \S+ [+-]\d{4}) Sleep\s+Entering Sleep.* (\d+) secs")


def sleeps(path: Path) -> list[tuple[float, float]]:
    """(start, seconds) of each system sleep in `pmset -g log` output."""
    out = []
    for line in path.read_text().splitlines() if path.exists() else []:
        if m := SLEEP.match(line):
            out.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S %z").timestamp(), float(m.group(2))))
    return out


def awake(wall: float, slept: list[tuple[float, float]]) -> float:
    """Seconds since the epoch, less the time asleep before `wall`."""
    return wall - sum(min(d, wall - s) for s, d in slept if s < wall)


def resvg_mse(svg: str, target: np.ndarray) -> float:
    h, w, _ = target.shape
    return float(((rasterise(svg, w, h) - target) ** 2).mean())


def count(svg: str) -> int:
    return len(SHAPE.findall(svg)) - len(re.findall(r'<rect width="\d+" height="\d+"', svg))


def splat_curve(run: Path, target: np.ndarray) -> dict:
    info = json.loads((run / "run.json").read_text())
    times = {r["it"]: r["t"] for r in info["log"]}
    t, mse, n = [], [], []
    for f in sorted((run / "frames").glob("*.svg")):
        it = int(f.stem)
        if it in times:
            svg = f.read_text()
            t.append(times[it]), mse.append(resvg_mse(svg, target)), n.append(count(svg))
    svg = (run / "best.svg").read_text()
    t.append(info["result"]["seconds"]), mse.append(resvg_mse(svg, target)), n.append(count(svg))
    return {"t": t, "mse": mse, "n": n}


def genetic_curve(run: Path, target: np.ndarray) -> dict:
    h, w, _ = target.shape
    lines = (run / "genetic.log").read_text().splitlines()
    start = float(lines[0].split()[-1])
    # The program's timestamps are wall-clock; remove system sleep (pmset log).
    slept = sleeps(run / "sleep.log")
    t0 = awake(start, slept)
    log = {"t": [], "mse": [], "n": []}
    for line in lines:
        if m := FITTEST.match(line):
            stamp = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc).timestamp()
            log["t"].append(awake(stamp, slept) - t0)
            log["mse"].append(float(m.group(2)) / (255**2 * 3 * w * h))
            log["n"].append(int(m.group(3)))
    snaps = {"t": [], "mse": [], "n": []}
    for f in sorted((run / "frames").glob("*.svg")):
        svg = f.read_text()
        snaps["t"].append(awake(start + int(f.stem), slept) - t0), snaps["mse"].append(resvg_mse(svg, target)), snaps["n"].append(count(svg))
    return {"log": log, "snapshots": snaps}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("runs", nargs="+", help="label=directory")
    ap.add_argument("--out", default="out/curves.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.5))
    summary = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        run = Path(path)
        if (run / "genetic.log").exists():
            g = genetic_curve(run, target)
            c = g["snapshots"]
            line, = ax.plot(c["t"], c["mse"], label=label, color="black", lw=2)
            ax.plot(g["log"]["t"], g["log"]["mse"], color="black", lw=0.6, ls=":", label=f"{label} (own renderer)")
        else:
            c = splat_curve(run, target)
            line, = ax.plot(c["t"], c["mse"], label=label)
        bx.plot(c["t"], c["n"], color=line.get_color(), lw=2 if line.get_color() == "black" else 1.5, label=label)
        summary[label] = {"t": c["t"], "mse": c["mse"], "n": c["n"]}
    for a in (ax, bx):
        a.set_xscale("log")
        a.set_xlabel("processing time (s)")
        a.grid(True, which="both", alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylabel("MSE (resvg, colours in [0, 1])")
    ax.set_title("Error against time")
    bx.set_ylabel("shapes")
    bx.set_title("Shape count against time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
