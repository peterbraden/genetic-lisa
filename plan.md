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

## Fourth iteration: progressive resolution and animation

Goal (from the user): train coarse to fine, expanding the working
resolution in several steps as an adaptive rule; and animate the snapshots.

Known:
- Step cost scales with pixels × overlap: the 960x1431 Mona Lisa takes
  ~2.4 s a step against ~0.06 s for 200x311.
- λ is defined on mean squared error, so its meaning does not change with
  resolution; spawn thresholds (λ·3·h·w) and position learning rates
  (fractions of the longest side) are already scale-free.
- Ellipse parameters scale exactly: centres and semi-axes by f. With 4x4
  supersampling at the coarse level and 2x2 at twice the size, the sample
  points coincide, so the hard render is preserved exactly.
- Adam is invariant to the rescaling if the position moments are divided
  by f (first) and f² (second) and the position learning rate multiplied by
  f.

Design:
- A pyramid of targets, halving from the full size down to `min_size`
  (off by default).
- At a density step below full size, expand one level when the fitness
  improved by less than `expand_tol` (relative) since the previous density
  step: the error the current resolution can address is exhausted, and what
  remains is detail it cannot represent. Expansion is forced, one level per
  density step, from `expand_by` of the run, so that full-size
  densification follows.
- On expansion: rescale parameters and optimiser state, reset gradient
  statistics, skip the rest of that density step.
- Snapshots are written in full-size coordinates, so frames from every
  level compose one animation.

Assumptions:
- Relative fitness plateau is an adequate proxy for "error is
  homogeneous" at the current level. An alternative is the best spawn gain
  falling below λ.
- Edge softness, in pixels, becomes relatively sharper as resolution rises;
  this suits coarse-to-fine and is left as is.

Status: implemented (`--min-size`, `Model.rescale`), with tests that the
rescaled hard coverage is exact and that Adam steps scale with the image.
Not yet measured.

## Error against time

Goal (from the user): compare all approaches by error against processing
time, with the evolutionary version as the baseline; both the initial slope
and the plateau matter.

Known:
- The Rust implementation builds with crate versions pinned to its era
  (Cargo.lock, not committed: image 0.18, rand 0.3, jpeg-decoder 0.1,
  backtrace 0.3.69). It is single-threaded, has no shape penalty, and its
  fitness is the summed squared 0-255 difference, so MSE = F / (255²·3·w·h).
- `scripts/run_genetic.py` records its log and snapshots; `scripts/curves.py`
  scores every snapshot of every run with resvg and plots MSE and shape
  count against time on log axes.

- The machine sleeps when idle, and wall-clock time includes the sleep: the
  first set of curves (and the 2-hour large Mona Lisa run) contained stalls
  of up to 8 minutes, identical across concurrent processes. Training now
  times with `time.perf_counter` (which excludes sleep on macOS), and timing
  runs are made under `caffeinate -i`.

Assumptions:
- Wall-clock time on one machine is the fair measure; the splat runs use 4
  threads, and one single-threaded run is included for comparison.
- Two hours suffice to show the evolutionary plateau. Unknown: the original
  36m.svg may have taken far longer.

Results (Mona Lisa 200x311, resvg MSE; genetic times corrected for sleep):
- Up to ~5 s the splat runs and the evolutionary search descend alike;
  thereafter the splat runs are faster: at 100 s ~0.0025 against ~0.0033.
- Progressive resolution (from 78 px) has much the steepest start: MSE 0.004
  at ~5 s, when the others are near 0.015; it ends at 0.0026 with 84 shapes
  in 83 s, slightly above the defaults (0.0025, 84 shapes, 145 s).
- λ = 1e-5 runs plateau at 0.0023-0.0025 by 100-150 s. The evolutionary
  search reaches 0.0022 after 2360 s awake with 507 shapes, still falling.
  λ = 3e-7 reaches 0.0011 at 486 s with 379 shapes: below the evolutionary
  curve at every time after ~20 s, with fewer shapes.
- The evolutionary result scores 0.0019 on its own renderer against 0.0022
  on resvg: without anti-aliasing it partly fits its own aliasing.
- One thread is as fast as four at this size. Glazing is slower (342 s) and
  no better.

## Fifth iteration: extended run at full resolution

Goal (from the user): the large Mona Lisa (960x1431) has too few shapes
(82 at λ = 1e-5); run far longer with many more.

Known:
- No result so far shows a plateau: over the last two thirds (in log time)
  every run's MSE falls as a power law (slopes -0.16 to -0.49) with the
  shape count still rising; part of the splat runs' final fall is the edge
  ramp, a one-off.
- Prune, relocate, reorder and the final greedy prune held dense (N, H, W)
  tensors: ~450 MB each at 82 shapes, over 5 GB at 1000 shapes at full
  size, on an 8 GB machine. They now use sparse (shape, pixel) pairs within
  each footprint (`soft_pairs`, `hard_pairs`, `removal_cost_pairs`,
  `move_cost_pairs`), equal to the dense versions by test; a test fails if
  density control or the final prune evaluates dense coverage.
- At full size a training step takes ~1.5 s with 82 or 2000 shapes; the
  sparse costs take 0.3-0.6 s.

Run: λ = 1e-7, 20000 steps, progressive from 240 px with all expansions by
25% of the run, snapshots every 250 steps, under `caffeinate -ims`
(`out/mona-long`).

Assumptions:
- The tuned constants (λ = 1e-5, 128 px, 2500 steps) remain adequate at
  λ = 1e-7 and 8x the steps; the schedule stretches with `iters`.
- Shape count will be limited by λ rather than by growth: 84 density steps.

Results (12.6 h, not the estimated 6-7):
- 1090 shapes, MSE 0.00439 at 960x1431, against 82 shapes and 0.00591.
  Visually far better: features, hands and landscape are recognisable.
- The 240 px phase took 9 min to reach 383 shapes; at expansion the full-size
  MSE was 0.00466. The remaining 12 h added ~700 shapes for 6% lower MSE and
  modest visible detail. Shape growth was still 14 spawns per density step
  at the end (not λ-limited).
- Full-size MSE is dominated by fine texture (craquelure, grain): the source
  downsampled and upsampled again scores 0.00431 from 120 px, 0.00397 from
  240 px, 0.00355 from 480 px. 1090 ellipses score worse than a perfect
  120 px image, so MSE at this size barely measures what the shapes achieve.
- Full-size cost: ~1.9 s per training step (benchmark 1.5 s) and roughly
  4-5 min per density step, ~4 h of the total. Not yet profiled.

## Possible next steps

- Restarts or a small population over the stochasticity identified above.
- Relocation restricted to the equilibrium phase.
- Larger source images, where culling matters more.

## Compression investigation

Ellipses as very lossy image compression (SVG and an estimated packed
binary format against JPEG, WebP and AVIF, by MSE, DSSIM and LPIPS):
`splat/investigations/compression.md`.
