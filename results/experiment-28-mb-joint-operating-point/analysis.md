# Experiment 28 — Joint mushroom-body operating-point search

**Verdict: PARTIAL — TRADEOFF REMAINS.**

- No point on the preregistered 25-point refractory × APL grid passes R1–R6 together.
- `frozen-operating-point.json` therefore records `"operating_point": null` and
  `"status": "NO PASSING OPERATING POINT"`.
- The KCg-d visual-coding region (R3) and the per-MBON plastic-leverage region (R5) do **not** overlap.
- There is one improvement over E27. A regime that passes every criterion except R3 (80 ms + APL×3: R1,
  R2, R4_PAM, R4_PPL1, R5_MBON01, R5_MBON11, R6) now exists. It replicates on fresh confirmation seeds, but
  that confirmation is descriptive only.
- No learning was run. E26 Phases 3–6 should **not** be rerun.

Base: `origin/main` f43b618 (contains E27 b36e30d). Branch `experiment-28-mb-joint-operating-point`.
Preregistration: `preregistration.md`, hash-frozen in `preregistration-freeze.json` before any grid point.

## 1. E27 anchor reproduction (`anchor-reproduction.json`, `anchor-diagnosis.json`)

The three anchors were rerun with the merged code on the E27 calibration seeds 2801–2806:

| Anchor | KCg-d Hz (E28 / E27) | Best visual (E28 / E27) | Zero-edge, MBON01+11 (E28 / E27) | DAN 0.3 PAM/PPL pp (fixed / E27 grid) |
|---|---|---|---|---|
| stock | 48.52 / 48.52 | −0.02 % / −0.02 % | −0.33 % / −0.33 % | 0.0/0.0 / 0.0/0.0 |
| 20 ms + APL×5 | 1.37 / 1.37 | −10.28 % / −10.28 % | −7.65 % / −7.84 % | 12.8/10.6 / 12.8/10.6 |
| 60 ms | 3.32 / 3.32 | +4.31 % / +4.31 % | −10.80 % / −10.80 % | 10.0/9.3 / 10.0/9.3 |

Stock and 60 ms are bit-identical. At 20 ms + APL×5, baseline, visual and DAN results are bit-identical.
Leverage differs by 1–2 spikes in two seed/condition counts, and every pass/fail flag is unchanged.

Diagnosis: FlyBrain's float32 cuSPARSE SpMV (`_W @ spikes`) is not bit-deterministic on the Tesla P40.
Five of five probes gave run-to-run differences of ≤ 4.5e-8 on about 2,300 rows. Under the strong 0.3 V
all-KC drive, such an ULP difference can occasionally flip a single threshold crossing. Two checks rule
out a state leak across `reset()`:

- repeats of the assay within one process were identical;
- running the DAN assay first did not change leverage.

Judged not material, so E28 proceeded. Leverage values should nevertheless be read as carrying a few
tenths of a percentage point of floating-point reproducibility noise at strongly driven points.

## 2. Effective grid (`grid-config.json`)

| Requested ms | Steps | Effective ms | Firing ceiling | APL gains |
|---|---|---|---|---|
| 20 | 1 | 20 | 25.0 Hz | 3, 4, 5, 6, 8 |
| 40 | 2 | 40 | 16.7 Hz | 3, 4, 5, 6, 8 |
| 60 | 3 | 60 | 12.5 Hz | 3, 4, 5, 6, 8 |
| 80 | 4 | 80 | 10.0 Hz | 3, 4, 5, 6, 8 |
| 100 | 5 | 100 | 8.3 Hz | 3, 4, 5, 6, 8 |

The suggested values 20/30/40/50/60 ms map to 1/2/2/2/3 steps (`round(1.5) = round(2.5) = 2`), so 30 and
50 ms would duplicate 40 ms. They were replaced prospectively, before any result, by 80 and 100 ms. The
grid has 25 unique (steps, APL) points, and each ran exactly once, on calibration seeds 2901–2906 only.
KC gain was 1 and `sensory_input` was True throughout. Run time was about 252 s per point on the P40.

## 3. Joint grid (calibration seeds 2901–2906; `grid-results.json`, `candidate-ranking.json`)

Column definitions:

- **Visual Δ:** best KCg-d contrast vs no-visual (stimulus, consistent seeds).
- **PAM / PPL1 pp:** response to fixed amplitude 0.3, in percentage points of duty.
- **Leverage:** change in MBON firing when all 418 plastic edges are zeroed.
- **Gates** (in order): R1 R2 R3 R4_PAM R4_PPL1 R5_MBON01 R5_MBON11 R6.

| ms | st | APL | KCg-d Hz | Visual Δ | PAM pp | PPL1 pp | MBON01 | MBON11 | Gates |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 1 | 3 | 15.59 | −1.85 % (drifting_vertical, 5/6) | +0.0 | +0.7 | −1.15 % | −1.49 % | ✓✓✗✗✗✗✗✓ |
| 20 | 1 | 4 | 4.82 | +15.22 % (vertical, 6/6) | +11.6 | +12.5 | −5.33 % | −6.78 % | ✓✓✗✓✓✗✗✓ |
| 20 | 1 | 5 | 1.36 | +10.89 % (drifting_vertical, 3/6) | +10.9 | +9.0 | −6.40 % | −6.71 % | ✓✓✗✓✗✗✗✓ |
| 20 | 1 | 6 | 0.73 | +11.38 % (pokemon_walk, 4/6) | +15.0 | +9.7 | −3.86 % | −7.11 % | ✓✓✓✓✗✗✗✓ |
| 20 | 1 | 8 | 0.37 | −3.05 % (vertical, 4/6) | +13.6 | +14.1 | −8.24 % | −11.25 % | ✓✓✗✓✓✗✓✓ |
| 40 | 2 | 3 | 4.75 | +2.31 % (pokemon_walk, 3/6) | +9.6 | +10.0 | −3.57 % | +2.11 % | ✓✓✗✗✓✗✗✓ |
| 40 | 2 | 4 | 3.55 | −6.17 % (pokemon_walk, 5/6) | +8.8 | +8.8 | −5.06 % | −3.65 % | ✓✓✗✗✗✗✗✓ |
| 40 | 2 | 5 | 2.22 | −6.33 % (pokemon_walk, 4/6) | +10.9 | +8.4 | −2.81 % | +0.79 % | ✓✓✗✓✗✗✗✓ |
| 40 | 2 | 6 | 1.05 | +9.08 % (pokemon_bedroom, 5/6) | +12.8 | +8.8 | −2.66 % | −1.62 % | ✓✓✓✓✗✗✗✓ |
| 40 | 2 | 8 | 0.59 | −8.53 % (bedroom_after_walk, 5/6) | +12.9 | +12.2 | −3.13 % | −1.27 % | ✓✓✓✓✓✗✗✓ |
| 60 | 3 | 3 | 2.79 | +5.81 % (horizontal, 6/6) | +13.4 | +12.4 | −11.68 % | −7.59 % | ✓✓✗✓✓✓✗✓ |
| 60 | 3 | 4 | 2.26 | +2.61 % (pokemon_walk, 3/6) | +11.3 | +14.2 | −13.68 % | −7.92 % | ✓✓✗✓✓✓✗✓ |
| 60 | 3 | 5 | 1.85 | −10.23 % (horizontal, 4/6) | +12.4 | +12.2 | −15.36 % | −7.26 % | ✓✓✗✓✓✓✗✓ |
| 60 | 3 | 6 | 1.32 | +5.99 % (drifting_vertical, 3/6) | +11.6 | +11.9 | −6.56 % | −5.20 % | ✓✓✗✓✓✗✗✓ |
| 60 | 3 | 8 | 0.67 | +46.73 % (drifting_vertical, 6/6) | +14.5 | +11.9 | −7.06 % | −5.17 % | ✓✓✓✓✓✗✗✓ |
| 80 | 4 | 3 | 2.52 | −3.48 % (drifting_vertical, 4/6) | +12.2 | +10.0 | −14.89 % | −11.11 % | ✓✓✗✓✓✓✓✓ |
| 80 | 4 | 4 | 2.05 | +5.66 % (pokemon_walk, 5/6) | +11.1 | +13.2 | −12.67 % | −11.94 % | ✓✓✗✓✓✓✓✓ |
| 80 | 4 | 5 | 1.69 | −6.45 % (bedroom_after_walk, 4/6) | +10.9 | +11.3 | −11.74 % | −7.44 % | ✓✓✗✓✓✓✗✓ |
| 80 | 4 | 6 | 1.25 | +10.38 % (bedroom_after_walk, 4/6) | +14.0 | +12.9 | −10.86 % | −7.59 % | ✓✓✗✓✓✓✗✓ |
| 80 | 4 | 8 | 0.80 | −8.75 % (pokemon_bedroom, 4/6) | +13.2 | +10.1 | −7.97 % | −3.87 % | ✓✓✗✓✓✗✗✓ |
| 100 | 5 | 3 | 2.34 | −2.23 % (pokemon_bedroom, 6/6) | +11.3 | +11.0 | −15.75 % | −13.55 % | ✓✓✗✓✓✓✓✓ |
| 100 | 5 | 4 | 2.04 | −3.37 % (pokemon_bedroom, 5/6) | +10.4 | +11.3 | −13.67 % | −8.68 % | ✓✓✗✓✓✓✗✓ |
| 100 | 5 | 5 | 1.67 | +5.27 % (horizontal, 5/6) | +9.2 | +8.2 | −15.85 % | −13.04 % | ✓✓✗✗✗✓✓✓ |
| 100 | 5 | 6 | 1.35 | +3.68 % (bedroom_after_walk, 4/6) | +8.9 | +7.7 | −16.81 % | −11.48 % | ✓✓✗✗✗✓✓✓ |
| 100 | 5 | 8 | 0.90 | −17.86 % (horizontal, 5/6) | +10.0 | +9.1 | −7.17 % | −7.28 % | ✓✓✗✓✗✗✗✓ |

How the visual column relates to R3:

- The visual column shows the best condition against no-visual only.
- R3 additionally requires |rel| ≥ 5 % against gray, with the same sign, in ≥ 5/6 seeds.
- Rows with a large no-visual Δ can therefore still fail R3. Examples: at 20/4 the gray contrast is only +2.0 % (3/6); 100/8 is
  consistent in only 4/6 seeds against gray; 60/5 is consistent in only 4/6 seeds against no-visual.

Other observations:

- **R1, R2, R6:** all 25 points pass. The highest target ceiling-normalised duty on the grid is 0.648, the
  whole-brain at-ceiling fraction is ≤ 0.11 %, and the silent fraction is ≤ 5.6 %.
- **Saturation:** every grid point leaves it; the stock E26/E27 regime sits at 0.97 ceiling duty and
  48.5 Hz.

## 4. Criteria regions (`tradeoff-analysis.json`)

| Criterion | Points passing (of 25) |
|---|---|
| R1 / R2 / R6 | 25 / 25 / 25 |
| R3 (KCg-d visual) | **4**: 20/6, 40/6, 40/8, 60/8 |
| R4_PAM | 20 |
| R4_PPL1 | 16 |
| R4 (both DANs) | 15 |
| R5_MBON01 | 11 (every 60–100 ms point with APL ≤ 6 except 60/6) |
| R5_MBON11 | 6: 20/8, 80/3, 80/4, 100/3, 100/5, 100/6 |
| R5 (both MBONs) | **5**: 80/3, 80/4, 100/3, 100/5, 100/6 |
| R3 ∧ R5 | **0** |
| R3 ∧ R4 | 2: 40/8, 60/8 |
| R4 ∧ R5 | 3: 80/3, 80/4, 100/3 |
| All pass | **0** |

**Primary question: do the visual-selectivity region and the plastic-leverage region overlap in joint
refractory × APL space? No.**

- **R3 region:** passes only at high APL gain (≥ 6) and short-to-medium refractory (≤ 60 ms).
- **R5 region:** passes only at long refractory (≥ 80 ms) with low-to-moderate APL (≤ 6).
- **Separation:** no R5 cell touches an R3 cell, even diagonally. The closest pair, 60/8 (R3) and
  100/6 (R5), is two grid steps apart.
- **MBON11 is the binding leverage constraint.** It reaches −10 % only at 80–100 ms, at the isolated
  point 20/8 (−11.25 %) and nowhere in the R3 region. MBON01 is more permissive: it passes throughout
  60 ms with APL ≤ 5.

Response-surface figures are in `captures/experiment-28/`. Figures 01–06 are heatmaps (refractory rows ×
APL columns, with passing cells outlined in green):

- 01: visual modulation;
- 02: PAM01 headroom;
- 03: PPL101 headroom;
- 04: MBON01 leverage;
- 05: MBON11 leverage;
- 06: criteria count.

## 5. Frozen operating point

**None.** `frozen-operating-point.json` contains `{"status": "NO PASSING OPERATING POINT", "operating_point": null}`.
No point was interpolated.

## 6. Best non-passing point (`best-nonpassing-candidate.json`; descriptive, not frozen)

**80 ms (4 steps) + APL×3**, `ref80ms|apl3|kcg1|sens1`. It passes 7/8 sub-criteria and fails only R3.
80/4 and 100/3 also pass 7/8 (again failing only R3). The preregistered rule breaks the tie in favour of
80/3: fewest steps first, then lowest APL gain.

## 7. Confirmation (`confirmation-results.json`, seeds 2911–2920, run once; `descriptive_confirmation_only = true`)

| Metric | Stock (same seeds) | 80 ms + APL×3 |
|---|---|---|
| KCg-d / MBON01 / MBON11 / PAM01 / PPL101 rate | 48.6 Hz each | 2.51 / 2.83 / 3.07 / 2.83 / 2.99 Hz |
| Max target ceiling-normalised duty | 0.972 | 0.307 |
| Best visual Δ vs no-visual | +0.01 % | +1.44 % (pokemon_walk) |
| PAM01 at 0.3 | +0.0 pp | +13.2 pp (10/10 seeds positive) |
| PPL101 at 0.3 | +0.0 pp | +11.6 pp (10/10) |
| MBON01 zero-edge | −0.36 % | −15.15 % (10/10 lower) |
| MBON11 zero-edge | −0.32 % | −10.56 % (9/10 lower) |

`robust = false`, because the point is descriptive and R3 fails. The seed-count thresholds are absolute
(≥ 5 seeds, as in E27), which makes them weaker on 10 seeds. The observed counts are 9–10/10, so this does
not change any conclusion.

Everything except visual coding replicates on fresh seeds:

- non-saturated rates;
- DAN dynamic range at a fixed amplitude in both DANs;
- ≥ 10 % leverage on both MBONs.

## 8. Tradeoff analysis (descriptive Spearman ρ across the 25 points; not used for selection)

| Pair | ρ |
|---|---|
| visual \|Δ\| vs MBON01 leverage | −0.28 |
| visual \|Δ\| vs MBON11 leverage | −0.30 |
| visual \|Δ\| vs KCg-d baseline rate | −0.48 |
| visual \|Δ\| vs APL gain | +0.59 |
| MBON01 leverage vs refractory steps | +0.76 |
| MBON11 leverage vs refractory steps | +0.62 |
| PAM01 headroom vs refractory steps | −0.22 |
| PPL101 headroom vs refractory steps | −0.01 |
| PAM01 headroom vs APL gain | +0.42 |
| PPL101 headroom vs APL gain | +0.05 |

The two control parameters act on different properties:

- **APL gain drives visual selectivity.** Sparser KCs (KCg-d rate ρ = −0.94 with APL) give larger
  relative visual contrasts.
- **Refractory length drives plastic leverage.** Leverage has ρ = −0.15 with APL and +0.67 with steps
  (weaker MBON).

Visual \|Δ\| and leverage are only weakly anti-correlated (ρ ≈ −0.3). The trade-off is therefore not a
single smooth frontier that both criteria could meet in the middle. Leverage varies smoothly with
refractory, but R3 is patchy and non-monotonic. For example, at 60 ms:

- APL×8 gives +46.7 % (6/6 seeds);
- APL×6 gives +6.0 % (3/6);
- APL×5 gives −10.2 % (4/6).

The grid gives no evidence that raising APL gain at long refractory recovers visual coding: 80/8 and 100/8
fail R3 and also lose leverage. In this grid the two regions are **separated**: disjoint and never
adjacent. There is no sharp cliff and no overlap.

Figures:

- 07–08: visual \|Δ\| vs MBON01 and vs MBON11 leverage.
- 09: DAN headroom across the grid.
- 10: stock vs E27-best vs E28-best. E27-best is 20/5 at 4/8 on the E28 seeds; E28-best is 80/3 at 7/8.
- 11: confirmation.
- 12: gate summary.
- 13–16 (supplementary): KCg-d rate, ceiling duty, visual-consistency surfaces, anchor reproduction.

## 9. Criteria (best point, 80 ms + APL×3)

| Criterion | Calibration (2901–2906) | Confirmation (2911–2920, descriptive) |
|---|---|---|
| R1 | pass | pass |
| R2 | pass | pass |
| R3 | **fail** (best −3.48 %, 4/6) | **fail** (+1.44 %) |
| R4 PAM | pass (+12.2 pp, 6/6) | pass (+13.2 pp, 10/10) |
| R4 PPL1 | pass (+10.0 pp, 6/6) | pass (+11.6 pp, 10/10) |
| R5 MBON01 | pass (−14.89 %, 6/6) | pass (−15.15 %, 10/10) |
| R5 MBON11 | pass (−11.11 %, 5/6) | pass (−10.56 %, 9/10) |
| R6 | pass | pass |

## 10. Required answers

1. **What commit was E28 based on?** `origin/main` f43b6180410ee46c3fc852ad5806af6abe0991df (the merge of
   PR #12, containing E27 b36e30d).
2. **Did the E27 anchor points reproduce?** Yes. Stock and 60 ms are bit-identical. 20 ms + APL×5 is
   identical except for ±1–2 spikes in two leverage counts (−7.65 % vs −7.84 %, same flags). This was
   traced to non-deterministic float32 cuSPARSE SpMV; it is not material.
3. **What requested refractory values were used?** 20, 40, 60, 80 and 100 ms.
4. **What effective refractory-step values did they map to?** 1, 2, 3, 4 and 5 steps (20–100 ms
   effective).
5. **Were any requested values duplicates?** Of the suggested values, 30 and 50 ms would both have mapped
   to 2 steps (the same as 40 ms). They were replaced prospectively, so the used grid has no duplicates.
6. **How many unique joint operating points were run?** 25.
7. **Did every preregistered point run exactly once?** Yes (checked by `stage_select` and by
   `test_grid_complete_exactly_once_calibration_only`).
8. **What fixed DAN amplitude was used?** 0.3, for both channels. The adaptive `rule_amplitude` was not
   used.
9. **Does PAM01 pass R4 anywhere?** Yes, at 20/25 points.
10. **Does PPL101 pass R4 anywhere?** Yes, at 16/25 points (15 pass both DANs).
11. **Does KCg-d pass R3 anywhere?** Yes, at 4 points: 20/6, 40/6, 40/8 and 60/8.
12. **Does MBON01 pass R5 anywhere?** Yes, at 11 points (60–100 ms).
13. **Does MBON11 pass R5 anywhere?** Yes, at 6 points: 20/8, 80/3, 80/4, 100/3, 100/5 and 100/6.
14. **Do the R3 and R5 regions overlap?** No. They are disjoint and never adjacent (the minimum distance
    is 2 grid steps).
15. **Does any point pass R1–R6 simultaneously?** No.
16. **What is the least-invasive all-pass point?** None exists. The frozen operating point is an explicit
    null.
17. **Does it pass fresh confirmation?** Not applicable. The descriptive confirmation of the best
    non-passing point reproduced 7/8 sub-criteria and failed R3 again.
18. **If none pass, what is the best non-passing point?** 80 ms + APL×3. It fails only R3.
19. **Is the visual/leverage tradeoff continuous or sharply separated?** Separated rather than a sharp
    cliff. Leverage rises smoothly with refractory length, and visual selectivity rises (patchily) with
    APL gain. The regions pass through different corners of the grid and do not meet. Visual \|Δ\| and
    leverage are only weakly anti-correlated (ρ ≈ −0.3).
20. **Should E26 Phases 3–6 now be rerun?** No. Nothing was frozen, R3 fails at every leverage-sufficient
    point, and conditioning would be uninterpretable.

## 11. Interpretation and next experiment

This improves on E27, where no point passed R4 at a meaningful amplitude together with R5. E28 identifies
a reproducible non-saturated regime (80 ms + APL×3) in which:

- both DANs have fixed-amplitude dynamic range;
- the audited 418 KCg-d edges carry ≥ 10 % of **each** MBON's output.

However, in that regime the KCg-d population no longer carries a reliable visual signal. The KCg-d cells
that carry visual information are the ones sparsened by strong APL feedback, and in that regime the
audited edges carry < 10 % of the MBON output.

Per the preregistered failure policy, nothing was broadened post hoc: no KCs, MBONs or DANs were added,
plastic weights were not increased, the 10 % threshold was not lowered, visual injection was not altered,
KCg-d was not driven directly, and the grid was not widened. The question this raises is whether the
audited 418-edge KCg-d → MBON01/MBON11 circuit is too narrow to carry both properties, for example because
visual information reaches other KC classes or compartments. That needs a **separate audit and
preregistered experiment** covering the broader edge set. It must not be a modification of this one.
Learning remains blocked.
