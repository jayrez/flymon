# Experiment 28 — Joint refractory × APL operating-point search (preregistration)

Frozen before any E28 grid point was evaluated. Base: `origin/main` f43b618 (contains E27 b36e30d).
Branch: `experiment-28-mb-joint-operating-point`. No learning, conditioning, gameplay, rewards,
evolution or decoder optimisation is run. No RAM data enters the neural model.

## Question

Does a small, preregistered joint refractory × APL-gain grid contain one operating point that
simultaneously preserves KCg-d visual coding (R3), DAN dynamic range at a fixed amplitude (R4, PAM01
and PPL101 separately) and ≥ 10 % leverage of the frozen 418-edge plastic circuit on **each** MBON
(R5, MBON01 and MBON11 separately), while also passing R1, R2 and R6?

E27 found these properties in different places. At 20 ms + APL×5 there was visual selectivity but only
about 6–8 % leverage. At 60 ms there was ≥ 10 % leverage but no visual modulation. E27 froze nothing
validly. E28 asks whether the two regions overlap.

## Frozen elements (unchanged from E26/E27)

T4 pathway (sha256 87829806…), KCg-d identity (206 cells), PAM01, PPL101, MBON01, MBON11, the 418
audited plastic edges (hash 65dbc6264c04a2fa…), the connectome matrix `_W` (never mutated), the
plasticity architecture (zero plastic delta throughout), the visual stimuli, dt = 20 ms, and the noise
configuration. KC gain is fixed at 1.0 (not used in E28). No APL connections are added. APL gain scales
only the existing APL output columns, exactly as in E27 `OperatingPointBrain`.

## Anchor reproduction (before this grid)

The anchors are stock, 20 ms + APL×5 and 60 ms, run on the E27 calibration seeds 2801–2806 with the
merged code. They must reproduce the E27 sweep JSONs for KCg-d rate, the best visual contrast and the
zero-edge effect (see `anchor-reproduction.json`). If they differ materially, E28 stops.

## Refractory step resolution (prospective)

`refractory_steps = int(round(ms / 20))`. The suggested values 20/30/40/50/60 ms map to 1/2/2/2/3 steps.
Because of Python's banker's rounding, 30 and 50 ms both collapse onto 40 ms. They were therefore
replaced **before any result** by 80 and 100 ms.

| Requested ms | Steps | Effective ms | Ceiling Hz |
|---|---|---|---|
| 20 | 1 | 20 | 25.0 |
| 40 | 2 | 40 | 16.7 |
| 60 | 3 | 60 | 12.5 |
| 80 | 4 | 80 | 10.0 |
| 100 | 5 | 100 | 8.33 |

## Grid (exactly 25 points; never widened after results)

Refractory ∈ {20, 40, 60, 80, 100} ms × APL gain ∈ {3, 4, 5, 6, 8}. KC gain = 1, sensory_input = True.
Each point is evaluated exactly once, on calibration seeds only. See `grid-config.json`.

## Seeds (`seed-freeze.json`)

- Calibration: 2901–2906.
- Confirmation: 2911–2920.

The two sets are disjoint from each other and from all E26/E27 seeds. Confirmation seeds are never
used before the operating point is frozen.

## Assays

These are reused unchanged from E27: `baseline_assay`, `visual_assay` and `leverage_assay` (all KCg-d
driven at 0.3 V for 300 steps; baseline vs all 418 edges zeroed).

**New fixed-amplitude DAN assay (`dan_assay_fixed`).** For each channel (appetitive→PAM01,
aversive→PPL101) and each seed, the assay runs:

1. reset;
2. 200-step warm-up;
3. 100-step baseline;
4. a 10-step DAN stimulation window at a fixed amplitude of **0.3**.

The response is Δduty = stimulated duty − baseline duty, in percentage points. E27's adaptive
`rule_amplitude` is **not** used for R4.

## Criteria (persisted per point as a criteria object)

| ID | Rule |
|---|---|
| R1 | ceiling-normalised duty < 0.80 for KCg-d, MBON01, MBON11, PAM01, PPL101 |
| R2 | median target rate > 1 Hz and no target population silent (E27 rule) |
| R3 | E27 `passes_R3`: some stimulus changes KCg-d by \|rel\| ≥ 5 % vs both no-visual and gray, with the same sign in ≥ 5/6 seeds |
| R4_PAM | PAM01 mean Δduty ≥ 10 pp at amplitude 0.3, positive in ≥ 5/6 seeds |
| R4_PPL1 | PPL101 mean Δduty ≥ 10 pp at amplitude 0.3, positive in ≥ 5/6 seeds |
| R4 | R4_PAM and R4_PPL1 |
| R5_MBON01 | zeroing the 418 edges lowers MBON01 by ≥ 10 % (rel ≤ −0.10), lower in ≥ 5/6 seeds |
| R5_MBON11 | same rule for MBON11 |
| R5 | R5_MBON01 and R5_MBON11 (not pooled) |
| R6 | finite voltages, whole-brain at-ceiling fraction < 1 %, silent fraction < 50 % (E27 rule) |

## Selection (no interpolation)

1. Only points with all of R1–R6 passing are eligible.
2. Among those, choose the fewest effective refractory steps,
3. then the lowest APL gain,
4. then the lowest requested ms.

The frozen point is an exact grid point. If none passes, `frozen-operating-point.json` has
`"operating_point": null` and `"status": "NO PASSING OPERATING POINT"`. The best non-passing point is then
written separately to `best-nonpassing-candidate.json`. Its ranking is:

1. most sub-criteria passed;
2. then R3, then R5, then R4;
3. then fewest steps;
4. then lowest APL gain.

That point is **not** frozen.

## Confirmation (runs once)

The frozen point, together with stock, is run on seeds 2911–2920. If there is no passing point, the best
non-passing point is run instead, with `descriptive_confirmation_only = true`, and it cannot establish
an operating point. The point is robust only if it passes all criteria again on the confirmation seeds.

## Failure policy

If E28 fails, none of the following is done post hoc:

- add KCs, MBONs or DANs;
- increase plastic weights;
- lower the 10 % leverage threshold;
- alter visual injection;
- direct-drive KCg-d;
- widen the grid;
- use KC gain.

Learning experiments remain blocked until an operating point is frozen **and** confirmed.
