# Splat-Lisa

Approximates an image with translucent SVG ellipses, minimising

    fitness = MSE(SVG raster, source) + λ·N

where colours are in [0, 1] and `N` is the ellipse count. `λ` is the MSE
reduction an ellipse must buy to justify its existence.

Unlike the earlier evolutionary versions, this is directed rather than a
random walk, borrowing from 3D Gaussian Splatting:

- Differentiable rendering. Ellipses are composited in painter's order with a
  soft (sigmoid) edge whose width anneals from 2 px to 0.5 px; the final 20%
  of training uses a one-pixel linear ramp, which is the box-filtered coverage
  an anti-aliasing SVG rasteriser produces. Adam updates every parameter of
  every ellipse on every step. Loss is `0.8·L1 + 0.2·(1 − SSIM)` plus a small
  opacity penalty.
- Adaptive density control every 100 steps:
  - prune: remove ellipses whose exact leave-one-out MSE saving is below `λ`,
    or whose opacity is negligible;
  - clone / split: ellipses with high accumulated positional gradient are
    duplicated (small) or halved along the major axis (large), children
    taking the parent's place in the stack;
  - spawn: new ellipses are placed where an opaque patch of the local mean
    colour would save the most squared error, provided that saving exceeds `λ`.
- Final greedy pruning against the exact fitness on the hard-edged render.

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

All options: `uv run splatlisa --help`.

## Results

Mona Lisa, 200x311, 2500 steps on Apple MPS. MSE measured by resvg.

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
- Rendering is dense, `O(N·H·W)` per step; tile culling would be the next
  optimisation for larger images.
