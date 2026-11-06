import json

import numpy as np
from PIL import Image

from splatlisa.score import score
from splatlisa.train import Config, train


def test_training_improves_on_flat_background_and_writes_consistent_svg(tmp_path):
    # Two coloured discs on grey: a handful of ellipses should fit it well.
    h, w = 32, 40
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    img = np.full((h, w, 3), 0.5, dtype=np.float32)
    img[(xx - 12) ** 2 + (yy - 12) ** 2 < 49] = [0.9, 0.2, 0.1]
    img[((xx - 28) / 9) ** 2 + ((yy - 20) / 5) ** 2 < 1] = [0.1, 0.3, 0.8]
    src = tmp_path / "src.png"
    Image.fromarray((img * 255).astype(np.uint8)).save(src)

    lam = 1e-4
    cfg = Config(image=str(src), out=str(tmp_path / "out"), iters=300, lam=lam, device="cpu",
                 densify_from=50, densify_every=50, log_every=1000, snapshot_every=1000)
    result = train(cfg)

    flat = float(((img - img.reshape(-1, 3).mean(0)) ** 2).mean())
    assert result["mse"] < 0.2 * flat
    assert 1 <= result["n"] <= 12
    # The SVG, rasterised independently, scores as training reported.
    s = score(str(tmp_path / "out" / "best.svg"), str(src), lam, background="#000000")
    assert s["n"] == result["n"]
    assert abs(s["mse"] - result["mse"]) < 0.1 * result["mse"] + 1e-4
    run = json.loads((tmp_path / "out" / "run.json").read_text())
    assert run["result"]["fitness"] <= run["result"]["before_prune"]["fitness"]


def test_glaze_layers_are_added_during_training(tmp_path):
    # A textured image under a broad tint: a case for translucent layers.
    rng = np.random.default_rng(0)
    h, w = 32, 40
    img = 0.3 + 0.4 * rng.random((h, w, 3)).astype(np.float32)
    img[:, : w // 2] = 0.6 * img[:, : w // 2] + 0.4 * np.array([0.9, 0.8, 0.1], dtype=np.float32)
    src = tmp_path / "src.png"
    Image.fromarray((img * 255).astype(np.uint8)).save(src)
    cfg = Config(image=str(src), out=str(tmp_path / "out"), iters=200, lam=1e-5, device="cpu",
                 densify_from=50, densify_every=50, log_every=1000, snapshot_every=0, glaze_k=2)
    result = train(cfg)
    events = [r["event"] for r in json.loads((tmp_path / "out" / "run.json").read_text())["log"]]
    assert any("glaze+" in e and "glaze+0" not in e for e in events)
    s = score(str(tmp_path / "out" / "best.svg"), str(src), 1e-5, background="#000000")
    assert abs(s["mse"] - result["mse"]) < 0.1 * result["mse"] + 1e-4
