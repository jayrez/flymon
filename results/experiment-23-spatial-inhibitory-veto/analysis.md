# Experiment 23 — Mi1–Mi4 spatial inhibitory veto in T4

**Scientific verdict: FAIL (preregistered). Mechanistic category: E, reached by fall-through.
Gameplay progression: NO-GO (preregistered rule).**

Of the seven preregistered held-out criteria, six were met: native strength, reversal,
geometry shuffle, Mi4 inhibition, temporal persistence and static controls. Only
co-location missed. Co-locating Mi4 with Mi1 reduced |T4 DSI| by 66 % against a 70 %
threshold, and overshot to −0.064. The category ladder had no rule for "everything met
except co-location", so the result falls through to E.

**Analyst assessment (not preregistered): the spatial inhibitory veto is partially
supported, close to strongly.**

* Reversing where Mi4 sits relative to Mi1 reverses the preferred direction in all four T4
  subtypes (+0.185 → −0.170) and leaves ON polarity intact.
* Making Mi4 excitatory, removing Mi4 or removing temporal persistence abolishes direction
  selectivity.
* Differential fast/slow timing is not needed.

Preregistration: `preregistration.md`, frozen before any transformed model was run, with
one post-held-out amendment that changes nothing. Base: HEAD `09833da` (E22; E21 `74bf6f6`),
`origin/main` `88522ee`.

## 1. Candidate, reproduction and transforms

* **Frozen candidate.** The E22 R2 model at E21 parameters, with Mi1 + Mi4 inputs only:
  * fixed linear-0.5 background, 40 pre-adaptation frames;
  * G = 16, beta = 2, r0_exc = 1, Mi4 phasic (r0 = 0), E_inh = −0.2;
  * filters 4 / 12; E21 constants.
* **Calibration reproduction.** T4 +0.1965 (4/4 positive, 4/4 ON), matching E22's Mi1 + Mi4
  diagnostic.
* **Transforms (anatomy only).** Each T4 neuron's Mi4 inputs are rigidly translated by −2d
  (reversal) or −d (co-location), where d = c4 − c1 in screen pixels. Inputs are then snapped
  to unused retina columns of the same eye.
  * Weights, input counts and row totals are preserved exactly (tested).
  * Median snap error is 1.5–1.7 px; one column is 3.35 px.

Native geometry (anatomy, held-out independent):

| subtype | T4 with mapped Mi4 | median \|d\| | median projection on predicted preferred direction | fraction > 0 | angle R |
|---|---|---|---|---|---|
| T4a | 930 / 1680 | 2.68 px | +1.37 px | 0.79 | 0.54 |
| T4b | 913 / 1687 | 3.00 px | +1.55 px | 0.75 | 0.53 |
| T4c | 973 / 1773 | 2.85 px | +1.42 px | 0.74 | 0.61 |
| T4d | 900 / 1705 | 3.34 px | +1.79 px | 0.75 | 0.78 |

Mi4 sits *ahead* of Mi1 along the preferred direction. Preferred motion therefore excites
Mi1 before it reaches Mi4, while null motion meets Mi4 inhibition first. That is a
Barlow–Levick veto arrangement.

After reversal the median projection is −1.14 … −1.47 px; after co-location it is −0.24 … +0.04 px.

## 2. Held-out results (fresh bank, run once)

The held-out bank had never been evaluated before: speeds 0.4 / 1.25 / 2.5 / 3.5, widths
10 / 22, Weber contrasts 0.35 / 0.65, offsets ±14.

| condition | T4 mean DSI | + / − subtypes | ON>OFF | amplitude | criterion |
|---|---|---|---|---|---|
| native | **+0.1850** | 4 / 0 | 4/4 | 0.0071 | N ✓ |
| reversed Mi4 | **−0.1698** | 0 / 4 | 4/4 | 0.0071 | R ✓ (4/4) |
| co-located Mi4 | −0.0636 | 0 / 4 | 4/4 | 0.0069 | C ✗ (66 % reduction) |
| geometry shuffle (200) | null −0.0011 ± 0.0047; observed +0.1912 (speed axis) | | | | G ✓ (residual −0.006, p = 0.005) |
| Mi4 → excitatory | +0.0046 | 3 / 1 | 4/4 | 0.0211 | I ✓ |
| Mi4 removed | +0.0025 | 3 / 1 | 4/4 | 0.0169 | — |
| Mi1 removed | 0.0000 | — | 0/4 (no response) | **0.0000** | — |
| T1 equal filters (4/4) | +0.1559 | 4 / 0 | 4/4 | 0.0067 | not required to fail |
| T2 no filtering | −0.0000 | — | 4/4 | 0.0060 | P ✓ |
| T3 common tau 2 / 6 / 12 | +0.1545 / +0.1527 / +0.1458 | 4 / 0 each | 4/4 | 0.006–0.007 | — |
| reversed + equal filters | −0.1537 | 0 / 4 | 4/4 | 0.0068 | — |
| + Mi9 phasic (E22 strong) | +0.1850 | 4 / 0 | 4/4 | 0.0071 | Mi9 contributes nothing |
| static frozen-mid (native / reversed) | +0.0022 / −0.0078 | | 4/4 | | S ✓ |

Per subtype (bootstrap 95 % CI; Holm sign test in the expected direction):

| subtype | native DSI | CI | signResp | reversed DSI | CI | signResp |
|---|---|---|---|---|---|---|
| T4a | +0.212 | [+0.188, +0.236] | 0.72 | −0.191 | [−0.216, −0.165] | 0.27 |
| T4b | +0.194 | [+0.168, +0.220] | 0.74 | −0.168 | [−0.193, −0.144] | 0.35 |
| T4c | +0.195 | [+0.170, +0.220] | 0.72 | −0.155 | [−0.180, −0.131] | 0.33 |
| T4d | +0.140 | [+0.117, +0.163] | 0.68 | −0.165 | [−0.190, −0.141] | 0.33 |

All are significant in the predicted direction (Holm p ≤ 4e−24 native, ≤ 1e−20 reversed).
For reversed, signResp is the fraction with pref > null, so 0.27–0.35 means most responsive
neurons now prefer the anatomical null direction.

## 3. Per-neuron reversal

Native vs reversed DSI (6,845 T4 neurons):

* Pearson −0.50, Spearman −0.40.
* 67 % of native-positive neurons turn negative. In the Mi4-mapped subset the figure is
  75 % (per subtype 68–79 %).
* The median change is 0 because ~46 % of neurons have no mapped Mi4 input. Those are
  untouched by the transform and have DSI ≈ 0 (mean −0.006 … +0.008).
* Neurons with mapped Mi4: native +0.27 … +0.39 → reversed −0.29 … −0.34.

DSI tracks the anatomical offset. Spearman of native DSI against the Mi1→Mi4 projection on
the preferred direction is +0.62 / +0.75 / +0.69 / +0.65 (T4a–d). Against offset magnitude
it is only +0.12 … +0.26, so it is the *direction* of the offset that matters, not its size.

## 4. Why co-location missed

Exploratory; calibration only; run after held-out; `exploratory-colocation-diagnostic.json`.

* The co-location residual is the same with equal filters (−0.064) or a common tau of 12
  (−0.062). It is **not** caused by the Mi1/Mi4 filter asymmetry.
* The DSI-versus-shift curve is step-like:

  | fraction of offset removed | T4 DSI |
  |---|---|
  | 0 | +0.197 |
  | 0.5 | +0.183 |
  | 1.0 | −0.061 |
  | 1.5 | −0.137 |
  | 2.0 | −0.180 |

  The offsets are sub-column (median ~0.8–1.0 column), so half-shifts mostly snap back to
  the same columns.
* A snap round-trip (shift by +d then −d) back to the native position already costs 17 %
  (+0.164).

So the discrete retina lattice cannot place Mi4 exactly on Mi1. The co-location condition
tests "move Mi4 by one quantised offset" more than true co-location. The residual also shows
that the centroid alone does not fully determine DSI. Field shape and snapping around it
matter.

## 5. Oracle comparison (benchmark only)

| model | held-out T4 mean DSI | ON>OFF | geometry residual | temporal flat | Mi4 reversal |
|---|---|---|---|---|---|
| E19 `OPPONENT_SIGNED`, E23 bank | +0.343 | — | — | — | **+0.349 (no reversal)** |
| E22 strong (R2, E22 bank) | +0.196 | 4/4 | −0.004 | −18 % | — |
| E22 frozen (E22 bank) | +0.031 | 4/4 | 0.14 | −99 % | — |
| **E23 native (E23 bank)** | **+0.185** (54 % of oracle) | 4/4 | −0.006 | −16 % | **−0.170 (reverses)** |

Per-neuron correlation of native E23 with the oracle is 0.54 / 0.65 / 0.64 / 0.59 (T4a–d).

The oracle does not reverse when Mi4 is reversed. Its slow arm pools Mi9 and Mi4, and Mi9
dominates. So E23's causal signature is **Mi4-specific** and differs from the E19 oracle's
Mi9-weighted fast/slow correlator, even though the two agree on which direction neurons
prefer.

## 6. Answers

1. **Does the frozen candidate reproduce strong T4 DS prospectively?** Yes: +0.185 on a
   fresh held-out bank.
2. **Are all four T4 subtypes positive?** Yes, 4/4.
3. **Is ON specificity preserved?** Yes, 4/4 native, reversed and co-located.
4. **Does reversing the Mi4 offset reverse the preferred direction?** Yes: −0.170.
5. **How many subtypes reverse sign?** All four.
6. **Does co-location collapse DS?** Not to the preregistered standard: −0.064, a 66 %
   reduction, with a negative overshoot and lattice limits (§4).
7. **Does geometry shuffle collapse DS?** Yes, residual ≈ 0.
8. **Is Mi4 inhibition specifically needed?** Yes.
9. **What happens if Mi4 becomes excitatory?** DS goes to +0.005 while response amplitude
   triples.
10. **What happens if Mi4 is removed?** +0.003.
11. **What happens if Mi1 is removed?** No response at all (amplitude 0), so DSI is undefined
    and reported as 0.
12. **Is differential fast/slow timing necessary?** No. Equal filters give +0.156, 84 %.
13. **Is temporal persistence necessary?** Yes. With no filtering DSI is 0.
14. **Does common temporal filtering preserve DS?** Yes, across tau 2–12 (+0.146 … +0.155).
15. **Does DSI scale with the spatial offset?** With its projection on the preferred
    direction, strongly (Spearman +0.62 … +0.75). With its magnitude, only weakly.
16. **Do individual neurons reverse sign?** Yes: 67 % of all native-positive neurons, 75 %
    of those with Mi4 input; r = −0.50.
17. **Are static responses correct?** Yes. Static DSI is +0.002 / −0.008, T4 is ON ≫ OFF,
    and gray gives 0.
18. **How close is E23 to the E19 oracle?** 54 % of its T4 DSI, with per-neuron r ≈ 0.6.
    The oracle lacks E23's Mi4 reversal signature.
19. **Is a spatial inhibitory veto supported?** Partially, close to strongly. Reversal,
    sign and persistence tests all behave as a veto predicts. The co-location test missed
    its threshold, and centroid position does not fully determine DSI.
20. **Is T4 strong enough to freeze for gameplay?** Not under the preregistered rule
    (NO-GO); see §7.

## 7. Gameplay-readiness implication

**Preregistered decision: NO-GO.** The progression rule gave GO only for category A, and
CONDITIONAL GO only for A-partial or "A with 3/4 reversal". The co-location miss therefore
yields NO-GO.

**Discrepancy to flag.** The task specification defines CONDITIONAL GO as "causal geometry
support is strong but one threshold is marginal". This outcome fits that definition exactly:
one criterion missed by 4 percentage points, and every other causal test passed decisively.
My preregistration encoded a narrower rule. I have applied the preregistered rule and have
not switched to the task-level reading after seeing results. Whether to treat E23 as a
CONDITIONAL GO is a decision for the project owner.

If that decision is made:

* **Frozen for gameplay:** the E23 native T4 readout — fixed-background adapted Mi1 + Mi4
  single compartment, E21 parameters.
* **Still unresolved:** T5 (no direction selectivity; CT1 geometry unknown), a clean
  co-location null, and the residual lattice-snapping sensitivity.

No gameplay, generation harness or Experiment 24 was started.

## 8. Runtime

* Calibration verification: 95 s.
* Held-out: 464 s in total.
  * conditions: 229 s
  * static controls: 54 s
  * 200-permutation geometry null: 122 s (20 workers)
  * oracle: 25 s
* Analysis: about 1 min.
* Exploratory diagnostic: about 2 min.

CPU only, deterministic.

## Files

Results (this directory):

* `preregistration.md` (plus Amendment 1)
* `geometry-metrics.json`, `transform-mappings.json`
* `calibration-verification.json`
* `heldout-results.json` — all conditions, static controls, geometry null, oracle
* `statistics.json` — subtype statistics and per-neuron reversal
* `verdict.json`
* `heldout-per-neuron.npz`, `traces.npz`
* `exploratory-colocation-diagnostic.json`

Figures: `captures/experiment-23/01–15*.svg`.

Code:

* `flymon/spatial_veto.py`
* `run_spatial_veto_experiment.py`
* `analyze_spatial_veto_experiment.py`
* `diagnose_spatial_veto_colocation.py` (exploratory)
* `test_spatial_veto.py` (21 tests)
