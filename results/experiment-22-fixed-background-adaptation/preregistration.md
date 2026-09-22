# Experiment 22 — Fixed adapting background, pre-adaptation and tonic excitation (preregistration)

Written after the R0 reproduction and the luminance-scale check below, **before any E22
fixed-reference calibration or held-out evaluation**. **Prohibited:** gameplay, closed loop,
controller actions, DOWN, Pokémon RAM / coordinates / map or room IDs / dialogue or battle
state, reward, RL, behaviour cloning, classifier labels, per-neuron / per-subtype /
per-direction / per-stimulus parameters, manual preferred-direction assignment, held-out
tuning, invented Tm3/CT1 geometry. Experiments 1–21 and their artifacts are not modified;
`column_motion.py`, `tonic_disinhibition.py` and all earlier runners are imported unchanged.

## Base (verified)

Branch `worktree-bridge-cse_014eULVFzQa6eL4wV7d1hV9L`, HEAD `74bf6f6` (E21 commit, pushed to
the branch); `origin/main` = `88522ee` (E20 merge). E21 is in HEAD's ancestry but not yet
on `origin/main`; HEAD is a clean fast-forward of `origin/main`. Working tree clean apart
from the new E22 module. Canonical base = `74bf6f6`.

Re-read from E21 artifacts: E21 frozen model FAIL / E (held-out T4 +0.0258). Exploratory
A1 (D1 linear, G = 16, beta = 2, r0_exc = 1, r0_inh = 0, E_inh = −0.2, tau 4/12,
sigma = 0.2204, M = 0.1424): calibration T4 +0.2497 (4/4), all-8 +0.1265; held-out T4
+0.1879 (4/4), all-8 +0.0950; **T4 ON > OFF in 0/4 subtypes, T5 OFF > ON in 0/4**; held-out
geometry residual −0.03 (50 perms); temporal-flat T4 −0.0215. E19 `OPPONENT_SIGNED`
oracle: held-out T4 +0.3427, all-8 +0.3562.

## Motivation (verified in code)

`column_motion.contrast_signal` defines contrast as `L − mean_t L` **per stimulus sequence**,
and `column_motion.low_pass` initialises each filter at the first frame. Under a tonic
excitatory rate `r = max(0, r0 + beta s / sigma)` and an output rectified at the tonic
resting potential, the sign of the *between-bar* baseline therefore depends on stimulus
polarity: for an OFF bar (dark bar on a bright field) every column the bar crosses sits
slightly *above* its own sequence mean whenever the bar is elsewhere, holding Mi1 above
tonic; for an ON bar it sits slightly *below*. This is a candidate cause of E21's inverted
ON/OFF preference. E22 asks whether the E21 directional signal survives when contrast,
tonic excitation, filter state and membrane baseline are defined relative to one fixed
adapting background.

## Luminance scale (verified, not assumed)

The retina decodes sRGB to **linear** light before column sampling. Display gray 0.5 reads
back as linear **0.2122** (this is E19's `ctrl_gray`), and E19 full-contrast bars sit on a
black (ON) or white (OFF) field — there is no common gray background in the E19 bank.
E22 therefore specifies backgrounds **in linear light**. Primary background 0.5 (linear);
the rendered frame (uint8 truncation) reads back as **L0 = 0.49693**, and that measured
value is the reference. Bars are drawn at Weber contrast ±c around the background,
`L0 ± c·min(L0, 1 − L0)` (full contrast: 1.0 and 0.0 linear), so ON and OFF bars are
symmetric in the signal the model actually sees. Gratings: E19 sinusoid applied in linear
light around the background. Bar trajectories are E19's (`parametric_bar` mask).

## Reference conditions

| id | contrast signal | stimulus bank | initial state |
|---|---|---|---|
| R0 | `L − mean_t L` (E21 exactly) | E19 bank | filters at frame 0, membrane at tonic rest |
| R0g | `L − mean_t L` | gray-background bank | as R0 (separates background from reference) |
| R1 | `L − L0` | gray-background bank | filters at frame 0 (cold filters), membrane at tonic rest |
| R2 | `L − L0` | gray-background bank | **40 neutral L0 frames prepended** (filters and membrane settle; frames never scored) |
| R3 | as R2, tonic excitation removed (`r0_exc = 0`) | gray-background bank | as R2 |

R0 reproduction (done before this file): the E22 module reproduces E21 exploratory A1
calibration T4 +0.249659 and all-8 +0.126525 **bit-identically**.

**Direct test (no tuning):** R0, R0g, R1, R2 and R3 are evaluated at the unchanged E21 A1
parameters on calibration and held-out data. This answers the experimental question
independently of the calibration search below.

## Constants

`sigma = 0.2204`, `M = 0.1424`, chemistry and `E_inh = −0.2`, `tau_fast = 4`, `tau_slow = 12`
are inherited from E21 unchanged for every condition (so R0 → R2 changes only the reference,
background and initial state; gain changes enter only through the declared grid).
Adaptation length 40 frames (> 3 × tau_slow); sensitivity {30, 60} reported for the frozen
model only.

## Step-polarity audit (before any motion)

Full-field increment and decrement (Weber 0.5, step at frame 20) under R2. Required, per
presynaptic class, post-step mean filtered signal: Mi1 and Mi4 **> 0 for increments, < 0
for decrements**; Mi9, Tm1, Tm2, Tm9 **> 0 for decrements, < 0 for increments**; pre-step
signal exactly 0 (|s| < 1e−9) after adaptation; static L0 gives exactly zero contrast.
Also reported at the presynaptic-rate level for the frozen model, and for R0 (pre-step
artefact). **If the audit fails, E22 stops and the signal path is debugged.**

## Model hierarchy and calibration grid (calibration split only)

All E21 D1 linear-transfer single compartment; class-level tonic rates only.

* **A1** fixed background + tonic excitation, cold filters (`adapt = 0`), no tonic inhibition
* **A2** fixed background + pre-adaptation (`adapt = 40`) + tonic excitation
* **A3** A2 + tonic inhibition restored, **Mi9 and Mi4 separate**:
  `(r0_Mi9, r0_Mi4) ∈ {(0.5,0), (0,0.5), (0.5,0.5), (1,0), (0,1), (1,1)}`

Shared axes: `G ∈ {4, 16}`, `beta ∈ {1, 2}`, `r0_exc ∈ {0.5, 1, 2}`, background
`∈ {0.4, 0.5, 0.6}` (linear; bars at Weber ±c). 36 × 8 = **288 configurations**. No other
parameter is searched; temporal constants are not searched.

## Gates (calibration, applied in order, before any DSI ranking)

1. **ON/OFF gate:** each T4 subtype's mean response over all calibration ON moving bars
   exceeds its mean over OFF moving bars — **4/4 required**. (T5 OFF > ON reported,
   secondary.)
2. **Static gate:** (a) response to a static ON bar exceeds response to a static OFF bar in
   4/4 T4 subtypes (static = mid-trajectory frame held, E16 `frozen_mid`); (b) T4 mean DSI
   computed with every moving stimulus replaced by its `frozen_mid` version has
   |DSI| ≤ 0.05.
3. **Geometry gate:** 20 column-permutation nulls (seed 2209), calibration ON bank; residual
   = null mean / observed T4 mean DSI **< 0.40** with observed > 0.
4. **Temporal gate:** `tau_slow = tau_fast` reduces calibration T4 mean DSI by **≥ 40 %**.

Selection among gate survivors: highest calibration T4 mean DSI with 4/4 T4 positive; ties
within 0.005 go to the simpler level (A1 < A2 < A3), then the smaller total tonic rate. If
nothing survives all gates: freeze the best calibration T4 DSI among ON/OFF-gate survivors
(descriptive only); if none, the best overall (descriptive only). Exactly one configuration
is frozen in `selected-config.json` before held-out data are touched.

## Held-out (once)

Gray-background version of the E19 held-out bank: speeds 0.25/0.75/1.5/3/4, widths 12/26,
Weber contrasts 0.25/0.5/0.75, offsets ±28, two gratings, frozen/shuffle/reverse/gray
controls. Preferred direction per neuron = E18 anatomy prediction (unchanged). DSI =
(R_pref − R_null)/(R_pref + R_null + 1e−9); unit = neuron; 10 000-sample bootstrap CI per
subtype; Holm-corrected sign tests across the 8 subtypes.

## Primary success criteria (held-out)

1. 4/4 T4 subtypes ON > OFF; 2. T4 mean DSI ≥ +0.15; 3. 4/4 T4 subtypes positive;
4. geometry residual (200 perms, seed 22092026, speed axis) < 0.40 (< 0.30 reported as
strong); 5. temporal flattening reduces T4 mean DSI by ≥ 40 %; 6. stable under
pre-adaptation: |T4 mean DSI(adapted) − T4 mean DSI(cold)| ≤ 0.05 and both 4/4 ON > OFF.

Secondary (mechanistic lead): 4/4 ON > OFF, 4/4 positive, T4 mean DSI in [+0.08, +0.15),
geometry residual < 0.40 and temporal reduction ≥ 40 %.

## Categories and verdicts

* **A** strong DS with correct ON polarity (all primary criteria) → **PASS**
* **C** held-out T4 ≥ 0.15 and 4/4 positive, ON/OFF still wrong (< 4/4) → PARTIAL
* **B** ON/OFF correct 4/4, 4/4 positive, T4 ≥ 0.05, primary not met → PARTIAL
* **D** DS collapses under a fixed reference: R1, R2 at E21 parameters **and** the frozen
  model all give held-out T4 mean < 0.05 or fewer than 3/4 positive → FAIL (E21 lead was a
  sequence-mean artefact)
* **E** unstable / initialisation-dependent / uninterpretable (anything else, including a
  sign flip between cold and adapted) → FAIL

Precedence A, C, B, D, E. T5 is descriptive only (no CT1, no T5 retuning).

**Mechanistic attribution of the E21 lead** (E21 parameters, held-out): retention =
T4(R2) / T4(R0). ≥ 0.67 with 4/4 ON > OFF → *mostly real*; 0.33–0.67, or ≥ 0.67 with wrong
ON/OFF → *partly real*; < 0.33 → *mostly artefact* of sequence-mean centring × tonic
baseline × rectification.

**Gameplay readiness** (separate from scientific verdict): GO only for A; CONDITIONAL GO
for B with geometry and temporal dependence both satisfied; NO-GO otherwise. No gameplay
or generation experiment is started in E22 regardless.

## Held-out analyses for the frozen model

Reference comparison R0/R0g/R1/R2/R3; cold vs adapted (cold = `adapt = 0`, and a fully cold
membrane starting at E_leak); ablations B1 sequence-mean restored, B2 tonic excitation
removed, B3 cold start, B4 temporal flat, B5 geometry shuffle (200), B6 Mi9 removed, B7 Mi4
removed, B8 both removed, B9 frozen / gray / shuffle / reverse controls, B10 static ON/OFF
steps; inhibition decomposition (excitation only, +Mi9, +Mi4, +both); adaptation-length
{30, 40, 60}; oracle comparison (E21 exploratory A1, E21 frozen, E20 C2, E19
`OPPONENT_SIGNED` — benchmark only, never fitted to).

## Reproduction

```sh
python run_fixed_background_experiment.py calibrate
python run_fixed_background_experiment.py heldout
python analyze_fixed_background_experiment.py
```
