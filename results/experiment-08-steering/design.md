# Experiment 8 — preregistered baseline-relative DNa02 steering

Written before Experiment 8 calibration or test outcomes were generated.
Experiments 1–7 remain unchanged.

## Fixed scope

The primary controller emits only LEFT, RIGHT, or no action. DNg100, MDN, and
DNp01 are diagnostic records and cannot affect actions. The unchanged Experiment
3 18×20 encoder drives the same LC10a/LPLC2 MaleCNS populations. The controller
receives only DNa02 L/R window rates and a frozen pretrial baseline. It has no
pixels, encoder statistics, labels, game state, RAM, reward, trained parameters,
route, or action targets.

## Baseline and cadence

Each condition resets MaleCNS with its declared seed, then runs ten 10-step
no-injection baseline windows (2.0 simulated seconds). Per-trial DNa02 L/R means,
sample variances, L−R mean, and L−R sample variance are frozen before active
vision. Baseline does not adapt during the primary period. Each decision contains
10 MaleCNS steps (200 ms), then holds one direction for 8 PyBoy frames and releases
for 4. The primary horizon is 60 decisions. A one-decision cooldown follows every
action. Encoder state resets immediately before active vision.

## Calibration and threshold freezing

Calibration seeds are 401–420; each receives left-half dark, right-half dark,
uniform neutral, and no-vision stimuli for 30 active windows after its own fixed
baseline. Seeds 401–410 are the inner selection set. Seeds 411–420 are untouched
validation. Candidate multipliers are [1.0, 1.5, 2.0]. The per-trial threshold is
`multiplier × max(baseline L−R SD, pooled calibration SD floor)`. The SD floor is
the pooled within-seed residual SD from selection-set baseline windows and prevents
zero thresholds caused by 200 ms spike-count quantization.

For each multiplier, the fixed objective is mean correct lateral action fraction
(LEFT under left stimulus plus RIGHT under right stimulus)/2 minus mean incorrect
lateral fraction, subject to mean neutral/no-vision action fraction ≤ 10%. Select
the largest objective; ties select the larger multiplier. If none satisfy the
constraint, select the lowest neutral/no-vision action fraction, then the largest
objective, then the larger multiplier. Polarity is L evidence→LEFT only if the
selection-set mean `(left stimulus L−R) − (right stimulus L−R)` is positive;
otherwise it is reversed. Validation must show opposite-signed mean signals and a
two-sided paired seed sign-permutation p < 0.05 to satisfy the lateral prerequisite.
Thresholds and polarity are saved to `thresholds.json` before test seeds run.

## Primary and controls

Primary seeds 421–440, never used for selection. Each bedroom condition reloads
`states/bedroom.state`: live, no vision, frozen initial frame, deterministic fixed
spatial shuffle per seed, and horizontally mirrored live vision. A mapping-swap
intervention on seeds 421–430 uses normal live neural activity but reverses only
the controller output. Title animation is a separately declared dynamic natural
condition. It starts from a fresh emulator boot followed by 1,500 frames and is
tested with live and no vision for all primary seeds. No screen classifier selects
an environment or controller.

Matched seeds compare live/no-vision, live/frozen, live/shuffled, and normal/mirror.
Action statistic: total variation distance with 9,999 seed-stratified condition
label swaps. Continuous statistic: absolute paired difference of seed-mean steering
signals with 9,999 paired sign flips. Both were chosen before results. The lateral
assay uses the paired difference between each seed's left-stimulus and negated
right-stimulus mean signal with an exact-enumerated/random sign test.

Closed-loop feedback reports matched-decision frame-hash divergence, steering
signal absolute divergence, action disagreement, and their time courses for live
versus frozen. Natural encoder lateral energy is diagnostic only: sums of the
declared projection-vector entries assigned to left and right halves. It never
enters the controller.

PASS requires validated controlled lateral steering, significant live versus no
vision steering or actions, a predictable significant/substantial mirror change,
and live/frozen feedback exceeding Experiment 7's action TV 0.005 and trajectory
measures. WEAK requires controlled steering and natural continuous modulation but
sparse actions or weak feedback. Otherwise FAIL. Navigation is not evaluated.
