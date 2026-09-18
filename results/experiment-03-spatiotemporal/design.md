# Experiment 3 design, fixed before neural trials

## Experiment 2 collapse diagnosis

The saved Experiment 2 onset and steady injections were reconstructed in the
metadata-selected order LPLC2 L/R, LC10a L/R. Each vector has 460 entries.
Intro, New Game menu, Oak dialogue, and bedroom share onset hash
`d3777aff4d9c80dd7b7fad2f75c8fae1e2e1fcdf0ecdb20de86d66ce5d53999c`
and steady hash
`d97869e217b46f2ae5818022fc70de7073c92a8da3fc744eda94137d9865098d`.
Their pairwise Euclidean distance is exactly zero in both phases. Title's hashes
are different and its distances from each other screen are 17.1581 (onset) and
13.2665 (steady). Left onset has min 0, max 0.8, mean 0.39826, SD 0.40000,
229 nonzero; left steady has min 0, max 0.8, mean 0.23478, SD 0.36428,
135 nonzero. Right onset has min 0, max 0.8, mean 0.40174, SD 0.40000,
231 nonzero; right steady has min 0, max 0.8, mean 0.24348, SD 0.36810,
140 nonzero.

The cause is confirmed in the upstream implementation: the largest dark
component was reduced to `(dx, sqrt(area))`; `FeatureDetectors` uses `dx` only
to select side, and all five large component sizes saturate both active
channels at the 0.8 cap. Thus spatial averaging to a single object,
left/right aggregation, and detector saturation cause the equality. The
framebuffers and extracted component geometries differ, so the equality is
neither an identical-source-image issue nor an implementation bug in spike
aggregation. The biological projection neurons received exactly equal
voltage sequences in four conditions. Upstream does not provide a pixelwise
retinotopic map for these visual projection cells.

## Upstream reuse and new mapping

[`Eyes`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/eyes.py)
provides luminance plus absolute temporal change, but takes a 1-D panorama.
[`FeatureDetectors`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/eyes.py)
selects LPLC2 and LC10a by MaleCNS metadata and supplies their biological
interpretation, but maps an object only to one side. `FlyBrain.step(inject=...)`
and `reset(seed)` are reused, as is Experiment 1's exact DN spike aggregation
and distance analysis. [`brain.py`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/brain.py)
exposes neuron type, side, 3-D soma positions, and photoreceptor azimuth.
Photoreceptor azimuth has no elevation; visual projection neurons have no
documented receptive-field coordinates. Soma position is not assumed to be a
visual receptive field. `reservoir.Trace` is not used because exact spike
counts are needed.

Experiment 3 adds a **declared surrogate projection map**: split an 18x20
spatial grid at the horizontal midpoint, and give each metadata-selected
LPLC2/LC10a neuron a fixed grid-cell sample within its side. Grid cells are
assigned in row-major order, evenly spanning the 18x10 half-grid. This
preserves explicit x/y source coordinates and many independent injected
values, but does **not** claim true biological retinotopy or local adjacency
among target neurons. The cell ordering is stable only for this MaleCNS build.
LC10a receives a low-gain dark-object/contrast proxy; LPLC2 receives absolute
change as a motion/looming-like proxy. Neither is semantic opponent detection
or true looming. Gains are fixed before neural trials and kept below the
upstream 0.8 voltage cap. No LC4 threat or LPLC1 projectile stimulation is
invented from unsupported game labels.

The 18x20 grid is an exact 8x8 mean pooling of the 144x160 framebuffer: no
interpolation and no vertical averaging across the whole image. Ten captured
frames, four PyBoy ticks apart, are presented for 20 MaleCNS steps each,
preserving the prior 200-step trial length. The initial frame is the same
conceptual checkpoint as Experiments 1/2; bedroom starts from the canonical
state. A short fixed right-button interval is applied only to the bedroom
sequence; all trials reuse the one captured sequence, so emulator timing is
not a trial-level stochastic variable. Static control repeats each sequence's
first frame for all ten windows. No-vision means no external injection.

This map is a test of whether the MaleCNS downstream circuit can retain a
spatially distributed, temporally changing direct projection-neuron signal.
It cannot establish that these artificial cell-to-pixel assignments match a
fly's anatomical visual field. Distinct encoder outputs are a required
preflight gate; a collapse will be reported rather than silently ignored.

Implementation detail fixed before the CUDA trial: source voltages were rounded to
16 uniform levels from 0 to 0.8 (0.05 increments), allowing FlyBrain's
scalar-per-group `inject` API to drive several cells at once. The exact
quantized 460-cell vectors are retained in `trials.json`. This quantization
was not chosen from condition labels. Preflight found five unique condition
sequences; Title and New Game menu had no framebuffer change during the
36-elapsed-game-frame observation window. The static control is therefore identical
to their dynamic condition, which is an expected limitation of the capture.
