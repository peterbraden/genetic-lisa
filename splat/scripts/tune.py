"""Hyperparameter search over the training constants with Optuna (TPE).

The objective is the mean, over the tuning split of corpus.tsv, of
fitness(trial) / fitness(defaults) at a reduced resolution, so that each
image counts equally regardless of how hard it is. `lam` is held fixed: it
defines the goal rather than the method.

    # Fitness of the defaults on each tuning image (run once)
    python scripts/tune.py baseline
    # Start one search worker; run several in parallel against the same study
    python scripts/tune.py search --trials 25
    # Best trials so far
    python scripts/tune.py best
    # Compare defaults and a parameter set on another split, at another size
    python scripts/tune.py evaluate --params tuning/best.json --split held --max-size 0
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

import optuna
import torch

from splatlisa.train import Config, train

ROOT = Path(__file__).resolve().parent.parent
TUNING = ROOT / "tuning"
STORAGE = f"sqlite:///{TUNING / 'study.db'}"
STUDY = "splatlisa"
PROXY_SIZE = 128
LAM = 1e-5


def images(split: str) -> list[Path]:
    out = []
    for line in (ROOT / "corpus.tsv").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            s, slug, _ = line.split("\t")
            if s == split:
                out.append(ROOT / "corpus" / f"{slug}.png")
    return out


def run(image: Path, params: dict, max_size: int | None) -> dict:
    with tempfile.TemporaryDirectory() as out:
        cfg = Config(image=str(image), out=out, lam=LAM, max_size=max_size or None,
                     quiet=True, snapshot_every=0, **params)
        return train(cfg)


def suggest(trial: optuna.Trial) -> dict:
    f, i, c = trial.suggest_float, trial.suggest_int, trial.suggest_categorical
    return {
        "tau_start": f("tau_start", 0.75, 4.0, log=True),
        "tau_end": f("tau_end", 0.25, 1.0, log=True),
        "anneal_until": f("anneal_until", 0.3, 0.8),
        "ramp_from": f("ramp_from", 0.6, 0.95),
        "densify_until": f("densify_until", 0.4, 0.85),
        "densify_every": i("densify_every", 50, 200, step=25),
        "ssim_weight": f("ssim_weight", 0.0, 0.5),
        "opacity_penalty": f("opacity_penalty", 0.1, 10.0, log=True),
        "lr_mu": f("lr_mu", 3e-4, 6e-3, log=True),
        "lr_scale": f("lr_scale", 0.005, 0.08, log=True),
        "lr_theta": f("lr_theta", 0.005, 0.08, log=True),
        "lr_col": f("lr_col", 0.01, 0.1, log=True),
        "lr_op": f("lr_op", 0.01, 0.1, log=True),
        "grad_ratio": f("grad_ratio", 1.0, 12.0, log=True),
        "size_threshold": f("size_threshold", 0.005, 0.08, log=True),
        "max_growth": f("max_growth", 0.02, 0.3),
        "max_prune": f("max_prune", 0.02, 0.3),
        "spawn_k": i("spawn_k", 2, 24),
        "spawn_max": f("spawn_max", 0.03, 0.25),
        "spawn_opacity": f("spawn_opacity", 0.4, 1.0),
        "relocate_k": c("relocate_k", [0, 2, 4]),
        "reorder_moves": c("reorder_moves", [0, 3]),
    }


def baseline_path(size: int) -> Path:
    return TUNING / f"baseline-{size}.json"


def cmd_baseline(args):
    TUNING.mkdir(exist_ok=True)
    fit = {p.stem: run(p, {}, PROXY_SIZE)["fitness"] for p in images("tune")}
    baseline_path(PROXY_SIZE).write_text(json.dumps(fit, indent=1))
    print(json.dumps(fit, indent=1))


def cmd_search(args):
    torch.set_num_threads(args.threads)
    base = json.loads(baseline_path(PROXY_SIZE).read_text())
    paths = images("tune")

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        ratios = []
        for step, p in enumerate(paths):
            ratios.append(run(p, params, PROXY_SIZE)["fitness"] / base[p.stem])
            trial.report(sum(ratios) / len(ratios), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        for p, r in zip(paths, ratios):
            trial.set_user_attr(p.stem, r)
        return sum(ratios) / len(ratios)

    study = optuna.create_study(
        study_name=STUDY, storage=STORAGE, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=16),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=args.trials)


def cmd_best(args):
    study = optuna.load_study(study_name=STUDY, storage=STORAGE)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    print(f"{len(done)} complete, {pruned} pruned")
    for t in sorted(done, key=lambda t: t.value)[: args.top]:
        per = " ".join(f"{k}={v:.3f}" for k, v in t.user_attrs.items())
        print(f"#{t.number:3d} {t.value:.4f}  {per}")
    best = study.best_trial
    (TUNING / "best.json").write_text(json.dumps(best.params, indent=1))
    print(json.dumps(best.params, indent=1))


def cmd_evaluate(args):
    torch.set_num_threads(args.threads)
    params = json.loads(Path(args.params).read_text())
    ratios = []
    for p in images(args.split):
        a = run(p, {}, args.max_size)
        b = run(p, params, args.max_size)
        ratios.append(b["fitness"] / a["fitness"])
        print(f"{p.stem:14s} default n={a['n']:4d} fit={a['fitness']:.5f}   "
              f"tuned n={b['n']:4d} fit={b['fitness']:.5f}   ratio={ratios[-1]:.3f}", flush=True)
    print(f"geometric mean ratio {math.exp(sum(map(math.log, ratios)) / len(ratios)):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline")
    s = sub.add_parser("search")
    s.add_argument("--trials", type=int, default=25)
    s.add_argument("--threads", type=int, default=2)
    b = sub.add_parser("best")
    b.add_argument("--top", type=int, default=10)
    e = sub.add_parser("evaluate")
    e.add_argument("--params", required=True)
    e.add_argument("--split", default="held")
    e.add_argument("--max-size", type=int, default=PROXY_SIZE)
    e.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    {"baseline": cmd_baseline, "search": cmd_search, "best": cmd_best, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
