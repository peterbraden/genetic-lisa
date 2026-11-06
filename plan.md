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

## Second iteration

Known:
- Culled rendering matches dense rendering (images and gradients, by test)
  and, with separable SSIM, a step is ~8x cheaper on CPU (77 ms vs 611 ms at
  168 ellipses). A full run takes ~2.5 min rather than ~4.
- Run-to-run noise is about ±3% in fitness.
- Relocation and top/bottom reordering, both exact in their cost estimates
  and tested, did not improve fitness beyond that noise (4 runs each).
  Disabled by default.

Assumptions not yet tested:
- Relocation may help only if applied late, once spawning has ceased; the
  trial applied it throughout densification.
- Given the noise, several short restarts with the best kept (a population,
  as in the earlier evolutionary versions) may be the cheapest gain.

## Third iteration: large translucent shapes

Hypothesis (from the user): at low error the fitness surface is rugged, and
escaping its local minima needs large translucent shapes affecting broad
areas, which the evolutionary search produced and which splitting may
prevent.

Known (`scripts/shapes.py`, Mona Lisa 200x311):
- The evolutionary `36m.svg` (140 shapes) is a deep stack of large faint
  layers: median area 1.6% of the canvas, median opacity 0.39, 6.4 shapes
  over the mean pixel, 10 shapes over 10% of the canvas.
- Our results are nearly a mosaic: splat-168 has median area 0.35%, median
  opacity 0.54, 1.7 shapes per pixel, one shape over 10%.
- Our results nonetheless have lower MSE for the shape count, so translucency
  is not by itself better under MSE.

Assumptions:
- Spawning, not splitting, is the main obstacle: spawn proposes near-opaque
  patches no wider than 10% of the image, and clone/split respond to local
  positional gradient, which a broad low-amplitude error barely produces.
- A "glaze" spawn rule, scoring a region by A·|mean(target − current)|²,
  would propose such shapes.

Tests:
1. Shape statistics (done, above).
2. Escape test (`scripts/escape.py`): add one large faint ellipse to a
   finished result and fine-tune, against null perturbations. Improvement
   beyond the null spread indicates a trap of the hypothesised kind.
3. Ablations: no split; spawn only.
4. Glaze spawn rule, then repeat 1 and 2.

Results (λ = 1e-5, two perturbations each, resvg fitness):
- Escape test: on splat-99 and splat-168, 18 added glazes against 6 null
  perturbations gave no gain (best -0.6%); fine-tuning fades the added
  layer. A finished mosaic is not improved by one added layer.
- Very low λ (1e-6, 3e-7, 4000 steps): 384 and 465 shapes, MSE ~0.0013,
  nearly identical, so limited by the growth rate within the densification
  window rather than by λ. Shapes became smaller and more opaque (median
  opacity 0.68), not translucent.
- Ablations, fitness mean of two: defaults 0.00344; no split 0.00330 (both
  runs below every default run); spawn only 0.00338 with ~87 shapes; faint
  spawns (opacity 0.2) 0.00340; glaze 2/step 0.00344, 4/step 0.00350.
- Glaze runs are layered (3-5 shapes per pixel against 1.3) with lower MSE
  but more shapes, on the same count/error curve rather than below it.
- The defaults now give 114-147 shapes, more than the 99 before the
  scale-free change (grad_ratio = 4); the change was not neutral.

Tuning (Optuna TPE, 80 trials, λ = 1e-5, 128 px):
- Best, trial 78: 0.909 on the tuning split; 0.970 geometric mean on the
  held-out split (5 of 7 better by 2-9%, 2 unchanged), with 20-30% fewer
  ellipses. Adopted, rounded, as the defaults.
- On the Mona Lisa the rounded defaults give 0.00326 and 0.00330 (86-87
  ellipses), against 0.00343-0.00344 for the previous defaults.

Not known:
- Whether the tuned constants suit other values of λ; a future objective
  should average over several.
- The contribution of relocation and reordering within the tuned set.
- Whether layering pays at very low λ once the step budget no longer limits
  the count (densification window and growth rate would need raising).

## Possible next steps

- Restarts or a small population over the stochasticity identified above.
- Relocation restricted to the equilibrium phase.
- Larger source images, where culling matters more.
