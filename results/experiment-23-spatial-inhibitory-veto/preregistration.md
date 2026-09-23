# Experiment 23 — Mi1–Mi4 spatial inhibitory veto in T4 (preregistration)

Written and frozen **before any E23 calibration-verification or held-out run of the
geometry manipulations**. Before this file only three things were run:

* the E22 Mi1 + Mi4 calibration value, read from E22's saved exploratory diagnostic;
* an anatomy-only inspection of Mi1/Mi4 centroids;
* the transforms' *geometric* effect (centroid offsets, weight totals, snap error).

No response of a transformed model has been computed. **Prohibited:** gameplay, closed
loop, controller actions, RAM, reward, RL, labels, per-subtype / per-neuron /
per-direction parameters, held-out tuning, invented CT1/Tm3 geometry, any use of the E19
oracle inside the model. E19–E22 code and artifacts are not modified, and no prior
held-out evaluation is rerun.

## Base (verified)

Branch `worktree-bridge-cse_014eULVFzQa6eL4wV7d1hV9L`, HEAD `09833da` (E22; contains E21
`74bf6f6`). Both are local/branch-only; `origin/main` = `88522ee`. Working tree clean.

## E22 findings re-read from artifacts

* **E22 R2 at E21 parameters (held-out):** T4 +0.1962, 4/4 positive, 4/4 ON>OFF,
  geometry residual −0.004.
* **Calibration diagnostic under R2:**

  | model | T4 |
  |---|---|
  | Mi1 + Mi4 | +0.1965 |
  | Mi1 + Mi9 | +0.0008 |
  | Mi1 alone | +0.0008 |
  | equal filters (4/4) | +0.1640 |
  | no filters | 0.0000 |

* **Tonic Mi9 0.5** suppressed DS (held-out +0.179 → +0.031).
* **E22 frozen model:** FAIL / E.

## Hypotheses

**H1** — T4 DS in the fixed-background adapted model comes from the relative *spatial
offset* of Mi4 inhibition with respect to Mi1 excitation. Reversing the offset reverses the
preferred direction; co-locating Mi4 with Mi1 collapses DS.

**H0** — Mi1/Mi4 geometry is correlated with the effect but not causal.

Anatomy (inspection, no responses): the |w|-weighted Mi4 centroid lies *ahead* of the Mi1
centroid along each neuron's E18 anatomy-predicted preferred direction. The anatomy-predicted
direction is derived from the fast-minus-slow arm offset, with Mi9 and Mi4 pooled. Numbers:

* median projection +1.4 … +1.8 px (column spacing 3.35 px);
* positive in 74–79 % of neurons;
* only ~54 % of T4 neurons have retina-mapped Mi4 input.

Preferred-direction motion therefore reaches Mi1 before Mi4 — the arrangement of a
Barlow–Levick null-side veto seen from the null direction.

## Frozen candidate (no search)

The E22 R2 model at the inherited E21 parameters, with Mi9 excluded:

| setting | value |
|---|---|
| reference | fixed; linear background 0.5 (L0 = 0.49693 measured); 40 neutral pre-adaptation frames, never scored |
| transfer | D1 linear, G = 16, beta = 2 |
| tonic rates | r0_exc = 1 (Mi1); Mi4 r0 = 0 (phasic inhibition) |
| E_inh | −0.2 |
| filters | tau_fast = 4 (Mi1), tau_slow = 12 (Mi4) |
| constants | sigma = 0.2204, M = 0.1424, chemistry inherited from E21 |
| inputs | Mi1 + Mi4 only |

Single passive compartment; output `max(V − V_rest, 0)`. No subtype, neuron or direction
coefficients.

## Geometry manipulations (anatomy only, frozen here)

Coordinates are screen pixels of the E5 retina columns. For each T4 neuron, c1 and c4 are
the |w|-weighted centroids of its Mi1 and Mi4 input columns, and d = c4 − c1. Every Mi4
input column p is rigidly translated and snapped to the nearest retina column of the same
eye not already used by that neuron. Heavier inputs are placed first; weights, input count
and row totals are preserved exactly.

* **G0 native** — unchanged.
* **G1 reversed** — p → p − 2d. The Mi1→Mi4 offset is negated; field shape is preserved.
* **G2 co-located** — p → p − d. The Mi4 centroid is moved onto the Mi1 centroid.
* **G3 geometry shuffle** — E22 column permutation of all partner classes. 200
  permutations on held-out, seed 23092026.

Mi1 geometry, all other subtypes, temporal filters, tonic rates, chemistry and stimuli
are unchanged. Neurons without mapped Mi4 input are unchanged by G1/G2. Known
limitation: snapping to the discrete lattice adds a median ~1.6 px position error per
input (geometric check done before this file); the co-location condition controls for it.

## Causal ablations

* **S1 Mi4 inhibitory → excitatory.** Mi4 weight signs are flipped, routing it to E_exc.
  Its rate transfer (r0 = 0), geometry and filter are unchanged.
* **S2** Mi4 removed.
* **S3** Mi1 removed. Response amplitude is reported alongside DSI.

## Temporal tests

* **T0** inherited filters (4 / 12).
* **T1** equal filters (4 / 4).
* **T2** no filtering (0 / 0).
* **T3** common-persistence sweep with Mi1 = Mi4 ∈ {2, 6, 12}.
* **T1-R** reversed geometry under equal filters (descriptive).

## Stimuli

**Calibration** (implementation verification only): the E22 calibration gray-background
bank (speeds 0.5 / 1 / 2).

**Held-out — a fresh bank never evaluated in any experiment:**

| axis | values |
|---|---|
| speed | 0.4, 1.25, 2.5, 3.5 px/frame |
| width | 10, 22 px |
| Weber contrast | 0.35, 0.65 |
| start offset | ±14 px |

Plus static frozen-mid bars, gray, frozen, shuffle, reverse and full-field steps. All
conditions use the linear-0.5 background.

The held-out bank is run once. Every condition above is evaluated in that single run.

## Metrics and statistics

* **DSI** is E19/E22's, with the E18 anatomy-predicted preferred direction per neuron.
* **Primary endpoint:** T4 mean DSI over all held-out moving ON bars.
* **Per subtype:** DSI, 10 000-sample bootstrap CI and signResp. Holm-corrected sign tests
  across the four subtypes use per-neuron pref − null for native, and null − pref for
  reversed.
* **ON>OFF:** mean over all moving ON bars against all moving OFF bars.
* **Amplitude:** mean (pref + null) response.
* **Per neuron (descriptive):** native vs reversed DSI — Pearson, Spearman, median change,
  and the fraction of native-positive responsive neurons that turn negative (all neurons
  and the Mi4-mapped subset). DSI vs offset magnitude and projection (Spearman).

## Criteria (held-out)

| id | criterion |
|---|---|
| N | native T4 mean ≥ +0.15, 4/4 positive, 4/4 ON>OFF |
| R | reversed T4 mean ≤ −0.10 **and** ≥ 3/4 subtypes negative (4/4 reported as strong) **and** reversed 4/4 ON>OFF (direction reverses, contrast polarity does not) |
| C | co-located \|T4 mean\| < 0.05 **or** ≥ 70 % reduction vs native |
| G | 200-perm geometry residual (null mean / observed) < 0.30 |
| I | Mi4-excitatory T4 mean ≤ 0.5 × native (≥ 50 % loss or reversal) |
| P | no-filter \|T4 mean\| < 0.05 or ≥ 70 % reduction |
| S | static frozen-mid \|T4 DSI\| ≤ 0.05 for native **and** reversed; static ON>OFF 4/4 (native) |

Equal filters (T1) are **not** required to fail.

## Categories

Evaluated in order:

* **E** — the candidate does not reproduce prospectively: native T4 < 0.10, or < 3/4
  positive, or ON>OFF < 4/4.
* **A** — strong spatial-veto confirmation: N, R, C, G, I, P, S all met → **PASS**.
* **A-partial** — every causal criterion (R, C, G, I, P, S) met, but native in
  [0.10, 0.15) → **PARTIAL** (mechanistic partial success).
* **B** — partial spatial-veto support: native geometry-dependent (G or C met); reversed
  ≤ 0.5 × native, but R not met.
* **C** — geometry necessary but not sufficient: G or C met; reversal does not reach
  ≤ 0.5 × native.
* **D** — Mi4 contributes (S2 loses ≥ 50 %) but spatial manipulations are not causal
  (neither G nor C, and R not met).
* Anything else → reported as **E** (uninterpretable).

## Gameplay progression

* **GO** — category A.
  * T4 readout frozen for gameplay.
  * T5 stays a separate, unresolved research track.
* **CONDITIONAL GO** — A-partial, or A except that R holds with exactly 3/4 subtypes.
  The caveat is stated explicitly.
* **NO-GO** — otherwise.

No gameplay, generation harness or Experiment 24 is started in E23.

## Amendments

Any amendment will be appended below with a timestamp, and will state whether held-out
data had already been viewed.

### Amendment 1 — 2026-09-23T03:19Z (held-out ALREADY viewed; verdict unchanged)

Two notes, neither changing any criterion, threshold, category, verdict or progression decision.

* **Category gap.** On held-out, every criterion was met except **C** (co-located |T4| = 0.064,
  a 66 % reduction against the 70 % threshold). The category ladder above has no rule for
  "all causal criteria met except co-location", so the case falls through to **E** and the
  progression rule gives **NO-GO**. Both are applied literally. An analyst assessment of
  spatial-veto support is reported separately in `verdict.json` and is labelled
  non-preregistered.
* **Post-hoc exploratory diagnostic.** `diagnose_spatial_veto_colocation.py` was run on the
  calibration split only, after the held-out run
  (`exploratory-colocation-diagnostic.json`). It tests whether the negative co-location
  residual comes from the filter asymmetry (it does not) or from lattice snapping (snapping
  alone costs ~17 % and quantises sub-column shifts). It is not used for any decision.
