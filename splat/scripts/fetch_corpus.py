"""Download the corpus listed in corpus.tsv from Wikimedia Commons.

Images are fetched as 512 px-wide thumbnails into corpus/ (git-ignored),
then validated and re-encoded as PNG.
"""

import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
UA = "splatlisa-research/0.1 (https://github.com/peterbraden/genetic-lisa)"
API = "https://commons.wikimedia.org/w/api.php"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def thumb_url(title: str, width: int) -> str:
    q = urllib.parse.urlencode({"action": "query", "titles": title, "prop": "imageinfo",
                                "iiprop": "url", "iiurlwidth": width, "format": "json"})
    pages = json.loads(get(f"{API}?{q}"))["query"]["pages"]
    page = next(iter(pages.values()))
    if "imageinfo" not in page:
        raise LookupError(f"not found: {title}")
    return page["imageinfo"][0]["thumburl"]


def main(width: int = 512):
    out = ROOT / "corpus"
    out.mkdir(exist_ok=True)
    for line in (ROOT / "corpus.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        split, slug, title = line.split("\t")
        dest = out / f"{slug}.png"
        if dest.exists():
            continue
        im = Image.open(io.BytesIO(get(thumb_url(title, width))))
        im.convert("RGB").save(dest)
        print(f"{split}\t{slug}\t{im.size[0]}x{im.size[1]}")
        time.sleep(1)


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
