"""Run the evolutionary Rust implementation for a fixed time and record its
progress: the "New fittest" log (timestamped) and a copy of best.svg every
`--every` seconds, named by elapsed seconds.

    cargo build --release --manifest-path ../rust/Cargo.toml
    uv run python scripts/run_genetic.py ../rust/target/release/lisa \
        ../couch-db/_attachments/image/lisa.jpg out/genetic --seconds 7200
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary")
    ap.add_argument("image", help="JPEG source")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=3600)
    ap.add_argument("--every", type=float, default=15)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    binary, image = Path(args.binary).resolve(), Path(args.image).resolve()
    log = open(out / "genetic.log", "w")
    start = time.time()
    log.write(f"# start {start}\n")
    log.flush()
    # The program writes best.svg, best.json and best.png to its working directory.
    proc = subprocess.Popen([str(binary), "-i", str(image)], cwd=out, stdout=log, stderr=subprocess.STDOUT)
    try:
        while (elapsed := time.time() - start) < args.seconds and proc.poll() is None:
            time.sleep(args.every)
            best = out / "best.svg"
            if best.exists():
                shutil.copy(best, out / "frames" / f"{int(time.time() - start):06d}.svg")
    finally:
        proc.terminate()
        proc.wait()
        log.close()


if __name__ == "__main__":
    main()
