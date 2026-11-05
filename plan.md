# Splat-Lisa: directed ellipse fitting

Goal: approximate a source image as an SVG of translucent ellipses, minimising
`fitness = MSE(render, source) + λ·N`, where `N` is the ellipse count.

The earlier implementations performed a random walk (mutate, keep if fitter).
This implementation borrows from 3D Gaussian Splatting (Kerbl et al. 2023) and
its 2D descendants (GaussianImage, Image-GS, 3DGS-MCMC):

1. Differentiable rendering. Every ellipse parameter receives a gradient on
   every step, so all shapes improve simultaneously instead of one mutation
   per generation.
2. Adaptive density control: periodic, directed structural edits.
   - clone: small ellipse with large accumulated positional gradient
     (under-reconstruction) is duplicated.
   - split: large ellipse with large positional gradient (over-reconstruction)
     is replaced by two smaller ones sampled within it.
   - spawn (Image-GS): new ellipse placed at the peak of the blurred residual
     error, coloured from the source.
   - prune: remove ellipses with negligible opacity, or whose exact
     leave-one-out MSE increase is below λ (they do not earn their keep).
3. Reparameterisation as in 3DGS: log-scales, sigmoid opacity and colour,
   Adam with per-group learning rates; optimiser state carried across
   structural edits.
4. Loss: L1 + 0.2·D-SSIM (3DGS) plus a small L1 opacity penalty that drives
   superfluous ellipses towards zero opacity, where pruning removes them.

## Known (verified in the repository)

- Source image: `couch-db/_attachments/image/lisa.jpg`, 200x311 RGB.
- Prior fitness: squared pixel difference with a per-shape surcharge of 1% of
  the error (multiplicative). This implementation uses an additive `λ·N`,
  which gives each ellipse an absolute MSE threshold to beat.

## Assumptions (to be checked)

- SVG ellipses have hard edges, unlike Gaussians. Training uses a sigmoid
  edge of width `τ` pixels, annealed towards ~0.5 px so the soft render
  converges on what an SVG renderer (with anti-aliasing) produces. The final
  fitness is measured with a hard, supersampled rasteriser of the SVG
  semantics.
- Compositing is SVG painter's order ("over" operator, sRGB, no gamma
  correction), with a solid background rectangle. Order is fixed; new shapes
  from clone/split are inserted adjacent to their parent, spawned shapes on top.
- Dense rendering (`N × H × W`) is affordable at this resolution; tile-based
  culling is unnecessary for now.

## Layout

```
splat/pyproject.toml
splat/splatlisa/render.py   # parameters, soft/hard rendering, leave-one-out cost
splat/splatlisa/model.py    # parameters plus optimiser state across structural edits
splat/splatlisa/density.py  # clone, split, spawn, prune
splat/splatlisa/loss.py     # L1, SSIM
splat/splatlisa/svg.py      # SVG export
splat/splatlisa/train.py    # CLI training loop
splat/splatlisa/score.py    # fitness of any SVG, via resvg
splat/tests/                # compositing, leave-one-out, density rules, SVG parity
```

## Findings

- Rendering parity with resvg is confirmed by test; training MSE matches the
  SVG MSE to within 1% once the final ramp-edge phase is used.
- The soft sigmoid edge understated MSE by about 15% relative to the SVG;
  the ramp phase removes the gap.
- After roughly 1000 steps spawning ceases (no site saves more than `λ`) and
  `N` reaches an equilibrium governed by `λ`. Results are in `splat/README.md`.

## Status

- [x] Renderer and leave-one-out with tests
- [x] Density control with tests
- [x] Training loop, SVG export, scoring tool
- [x] λ sweep on the Mona Lisa; results recorded

## Possible next steps

- Relocation (3DGS-MCMC): move the least valuable ellipses to high-error
  sites instead of prune-then-spawn, to escape local minima at fixed `N`.
- Tile culling to reduce the dense `N·H·W` cost.
- Reordering: the painting order is fixed at insertion; swaps could be tested
  with the same exact leave-one-out machinery.
