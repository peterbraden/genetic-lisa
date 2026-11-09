# Splat-Lisa

Approximates an image with translucent SVG ellipses, minimising

    fitness = MSE(SVG raster, source) + λ·N

where colours are in [0, 1] and `N` is the ellipse count. `λ` is the MSE
reduction an ellipse must buy to justify its existence.

Unlike the earlier evolutionary versions, this is directed rather than a
random walk, borrowing from 3D Gaussian Splatting:

- Differentiable rendering. Ellipses are composited in painter's order with a
  soft (sigmoid) edge whose width anneals from 1 px to 0.3 px; the final 22%
  of training uses a one-pixel linear ramp, which is the box-filtered coverage
  an anti-aliasing SVG rasteriser produces. Adam updates every parameter of
  every ellipse on every step. Loss is `0.82·L1 + 0.18·(1 − SSIM)` plus a
  small opacity penalty.
- Adaptive density control every 200 steps:
  - prune: remove ellipses whose exact leave-one-out MSE saving is below `λ`,
    or whose opacity is negligible;
  - clone / split: ellipses with high accumulated positional gradient are
    duplicated (small) or halved along the major axis (large), children
    taking the parent's place in the stack;
  - spawn: new ellipses are placed where an opaque patch of the local mean
    colour would save the most squared error, provided that saving exceeds `λ`;
  - relocate the least valuable ellipses to better sites (after 3DGS-MCMC),
    and move ellipses to the top or bottom of the stack when that exactly
    reduces the error;
  - optional glaze (`--glaze-k`): translucent layers over large boxes, with
    opacity and colour fitted by least squares to the residual over the
    current image. With opacity 1 this is the spawn rule; where the image
    already has the right structure it yields a faint wash.
- Final greedy pruning against the exact fitness on the hard-edged render.
- Culled rasterisation: each ellipse is evaluated only within its bounding
  box, and each pixel composites a short, ordered list (at most ~15 deep on
  the Mona Lisa), as 3DGS does per tile. Together with a separable SSIM
  window this makes a step about 8x cheaper than dense evaluation.

The training render is verified against resvg, an independent SVG
implementation (`tests/test_render.py`).

## Usage

```
# Train; writes out/best.svg, best.png, best.json, run.json and frames/
uv run splatlisa ../couch-db/_attachments/image/lisa.jpg --lam 1e-5 --out out
# Score any SVGs (including the older results) with the same fitness
uv run splatlisa-score ../couch-db/_attachments/image/lisa.jpg out/best.svg ../images/36m.svg
# Tests
uv run pytest
```

All options: `uv run splatlisa --help`. Constants are scale-free where
possible: positions, sizes and the position learning rate are fractions of
the image's longest side, the clone/split threshold is relative to the median
positional gradient, and the opacity penalty is in multiples of `λ`. The
defaults come from the search described below.

## Corpus and tuning

`corpus.tsv` lists thirteen public-domain paintings on Wikimedia Commons,
divided into a tuning split and a held-out split (which includes the Mona
Lisa). The images are not committed.

```
# Download into corpus/
uv run python scripts/fetch_corpus.py
# Hyperparameter search (Optuna TPE) at 128 px; state in tuning/ (not committed)
uv run python scripts/tune.py baseline
uv run python scripts/tune.py search --trials 20   # run several workers in parallel
uv run python scripts/tune.py best
# Check a parameter set against the defaults on the held-out split
uv run python scripts/tune.py evaluate --params tuning/best.json --split held
# Shape statistics (size, opacity, value, stacking depth) of any SVGs
uv run python scripts/shapes.py ../couch-db/_attachments/image/lisa.jpg ../images/36m.svg
# Does adding one large translucent ellipse to a result, then fine-tuning, help?
uv run python scripts/escape.py ../couch-db/_attachments/image/lisa.jpg ../images/splat-99.svg --lam 1e-5
```

The search (80 trials, 22 constants, λ = 1e-5, 128 px; objective the mean
ratio of fitness to that of the previous defaults) found trial 78 at 0.909 on
the tuning split. On the held-out split the geometric-mean ratio was 0.970:
five of seven paintings improved by 2-9% and two were unchanged, with 20-30%
fewer ellipses. Much of the tuning-split gain came from the Mondrian, which has
no counterpart among the held-out images. Principal changes: a sharper final
edge, a clone/split threshold of 9.3 times the median gradient (far fewer
clones and splits), longer and less frequent densification, and relocation and
reordering enabled. Note that `tuning/baseline-128.json` refers to the
previous defaults.

## Results

Mona Lisa, 200x311, 2500 steps. MSE measured by resvg. Times are from the
original dense renderer on Apple MPS; with culling, a λ = 1e-5 run takes about
2.5 minutes on CPU.

| λ      | ellipses | MSE     | time  | file                        |
|--------|----------|---------|-------|-----------------------------|
| 1e-4   | 38       | 0.00430 | 2 min | `images/splat-38.svg`       |
| 3e-5   | 54       | 0.00332 | 3 min | `images/splat-54.svg`       |
| 1e-5   | 99       | 0.00240 | 4 min | `images/splat-99.svg`       |
| 3e-6   | 168      | 0.00183 | 6 min | `images/splat-168.svg`      |

For comparison, the best earlier evolutionary result, `images/36m.svg`, uses
140 shapes (circles, polygons and rectangles) for an MSE of 0.00272.

## Observations

- Fitness plateaus once spawning finds no site worth `λ`; the ellipse count
  then settles at an equilibrium set by `λ`, not by the step budget.
- Soft edges flatter the MSE: a blurred approximation suits a sfumato
  painting. Training on the anti-aliased edge removes the discrepancy between
  training and SVG MSE.
- Results vary by about ±3% in fitness between runs differing only in
  inconsequential details (floating-point order, initial shape count). Any
  proposed improvement must be compared across several such perturbations.
- Relocation and reordering alone showed no gain on that basis (four runs
  each at λ = 1e-5, previous defaults); the search nonetheless enabled both,
  and their contribution within the tuned set has not been isolated.
- On the Mona Lisa at λ = 1e-5 (two runs each): previous defaults 0.00344,
  splitting disabled 0.00330, tuned defaults 0.00328 with 87 ellipses.
- The evolutionary result is a deep stack of large faint shapes (6.4 per
  pixel, median opacity 0.39); ours are nearly a mosaic (1.2-1.8 per pixel).
  Glazing produces layering (3-5 per pixel) and lower MSE but more ellipses,
  with no gain in fitness; adding one large translucent ellipse to a finished
  result and fine-tuning gives none either.
- With culling, MPS offers no advantage over CPU at this image size; sorting
  and scattering dominate. CPU is the default and is deterministic.
