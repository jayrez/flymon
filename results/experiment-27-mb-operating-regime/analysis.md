# Experiment 27 — Mushroom-body operating regime

**Verdict: PARTIAL.**

* **Saturation can be broken:** any refractory period ≥ 20 ms does it (R1, R2, R6).
* **Visual responsiveness can be restored:** refractory 20 ms plus 5× existing APL output
  (R3; KCg-d changes +7 % to +13 % by condition on confirmation seeds).
* **No preregistered operating point also gives R4 DAN headroom or R5 plastic leverage.**
* **The two goals conflict:** every point where zeroing the 418 audited edges changes MBON
  output ≥ 10 % (R5) has no visual modulation (R3 fails), and vice versa.

**No operating point was frozen.** E26 Phases 3–6 should **not** be rerun unchanged yet.

Base: branch `experiment-27-mb-operating-regime`, from
`origin/experiment-26-plasticity-validation` `336f06b` (contains `af29393`). PRs #9, #10 and
#11 are open. The preregistration was written after the stock reproduction and before any
intervention.

## Stock reproduction (E26 seeds 2701–2706)

* **Bit-identical to E26:**
  * KCg-d visual totals for every stimulus;
  * zeroing all 418 edges gives −0.34 %;
  * KC, MBON01/11, PAM01, PPL101, APL and DPM duty 0.971–0.973 (48.6 Hz), reached ≥ 90 % at
    steps 17–18.
* **Minor difference:** controller-DN duty differs by 1 spike in 36,000 cell-steps from E26's
  separate diagnostic run. The wrapper is verified bit-identical to `FlyBrain.step`, so this
  is GPU run-to-run floating-point variation.
* **Assay correction, fixed before any intervention:** the DAN baseline now follows a 200-step
  warm-up. A 50-step window had included the post-reset transient and showed spurious stock
  headroom.

## Refractory audit

* **Parameter:** `FlyBrain(refractory=seconds)`, converted to
  `refractory_steps = round(refractory/dt)`. It is global and 0 in stock.
* **Mechanism:** each step, `v = 0` for any neuron whose last spike was ≤ k steps ago, applied
  before the threshold check. So k steps caps duty at 1/(k+1).
* **Reset:** `reset()` clears `last_spike`; no state leaks (tested).
* **R1 definition:** because of the cap, R1 is scored on duty normalised to the refractory
  ceiling (preregistered).

## Refractory sweep (calibration seeds 2801–2806)

Visual Δ is KCg-d versus no visual, best condition, with seed consistency. DAN headroom is at
the rule amplitude. The last two columns are the effect of zeroing all 418 edges.

| refractory | KCg-d rate (ceiling-normalised duty) | KCg-d visual Δ | PAM01 headroom | PPL101 headroom | MBON01 zero-edge | MBON11 zero-edge | criteria passed |
|---|---|---|---|---|---|---|---|
| 0 (stock) | 48.5 Hz (0.97) | −0.02 % (1/6) | 0.0 pp | 0.0 pp | −0.34 % | −0.31 % | R2 |
| 20 ms | 16.8 Hz (0.67) | −0.34 % (5/6) | +3.4 | +2.9 | −6.8 % | −5.3 % | R1 R2 R6 |
| 40 ms | 7.3 Hz (0.44) | −1.7 % (6/6) | +4.7 | +2.7 | −10.0 % | −9.4 % | R1 R2 R6 |
| 60 ms | 3.3 Hz (0.27) | +4.3 % (4/6) | +7.0 | +7.6 | −11.5 % | −10.2 % | R1 R2 **R5** R6 |
| 80 ms | 2.7 Hz (0.27) | −3.5 % (4/6) | +4.9 | +5.2 | −12.3 % | −7.6 % | R1 R2 R6 |
| 100 ms | 2.4 Hz (0.28) | +0.9 % (4/6) | +4.5 | +4.7 | −4.3 % | −3.7 % | R1 R2 R6 |

At 40 ms the target MBON pair changes −9.7 %, just under the gate.

**Stop rule:** refractory fails, so the APL stage runs. Its base is the smallest refractory
passing R1 and R6: 20 ms.

## APL output gain (existing APL edges only; base 20 ms)

| APL gain | KCg-d rate | visual Δ | PAM / PPL headroom | target zero-edge | criteria passed |
|---|---|---|---|---|---|
| 1.5 | 16.3 Hz | +0.07 % | −0.4 / +1.7 pp | −1.4 % | R1 R2 R6 |
| 2 | 16.2 Hz | +0.08 % | +1.0 / +2.3 pp | −1.4 % | R1 R2 R6 |
| 3 | 15.5 Hz | +1.1 % | +0.5 / +0.7 pp | −2.5 % | R1 R2 R6 |
| 5 | 1.4 Hz | **−10.3 % (5/6)** | +3.4 / +2.2 pp | −7.8 % | R1 R2 **R3** R6 |

**Stop rule:** APL fails, so the shared-KC-gain stage runs.

## Shared KC input gain (all 4,064 KCs, one parameter; base 20 ms)

This is an engineered correction.

| KC gain | KCg-d rate | visual Δ | PAM / PPL headroom | target zero-edge | criteria passed |
|---|---|---|---|---|---|
| 0.75 | 15.9 Hz | +0.2 % | +0.7 / +1.6 pp | −5.3 % | R1 R2 R6 |
| 0.5 | 8.9 Hz | −0.9 % | +0.3 / +1.8 pp | −2.8 % | R1 R2 R6 |
| 0.35 | 3.9 Hz | +0.9 % | +3.9 / +3.4 pp | −10.5 % | R1 R2 **R5** R6 |
| 0.25 | 1.2 Hz | −0.9 % | +7.8 / +5.2 pp | −13.3 % | R1 R2 **R5** R6 |

**No family produced a point passing R1–R6.**

## Best-ranked point and confirmation (non-passing, descriptive)

**Best-ranked point:** refractory 20 ms + APL gain 5. Ranking is lexicographic (R1 and R2,
then R3, R4, R5, least invasive). It is frozen only for a descriptive confirmation; no point
passes.

**Confirmation seeds 2811–2820, run once:**

| population | rate |
|---|---|
| KCg-d | 1.3 Hz (ceiling-normalised duty 0.05) |
| MBON01 | 9.7 Hz |
| PAM01 | 10.7 Hz |
| PPL101 | 12.4 Hz |

| criterion | confirmation result |
|---|---|
| R3 visual | **holds** — bedroom-after-walk +12.9 % (9/10 seeds, d_z 0.57); bedroom +10.0 %; vertical +7.9 %; horizontal +6.7 %; drifting +7.6 %; walk +2.7 %; 36–60 % of KCg-d cells modulated |
| R4 DAN headroom (rule amplitude 0.1) | +3.8 / +4.0 pp → **fails as preregistered** |
| R5 zeroing edges | MBON01 −6.2 %, MBON11 −6.9 % (10/10 seeds lower) → **fails** |
| R1, R2, R6 | hold |

The rule amplitude of 0.1 is degenerate here. The amplitude grid shows +8 pp at 0.2, +12–13 pp
at 0.3 and up to +25 pp at 0.8.

**Stock on the same seeds:** saturated (48.6 Hz); visual effect ≤ 0.02 %; headroom 0; zeroing
edges −0.31 %.

## Mechanistic interpretation

Sources: `mechanistic-diagnostics.json`; figures 08–12.

* **Stock is a self-sustaining high-activity attractor** of the recurrent central-brain / KC
  network (KC → KC: 1.15 M ACh synapses) with no refractoriness. It covers ~7,600 neurons,
  including every KC, MBON, DAN, APL and DPM cell.
* **The olfactory loop is not the cause.** FlyBrain's documented `sensory_input=False`
  (non-selectable diagnostic) leaves the mushroom body at 97 % duty. The runaway olfactory
  receptor loop it removes is not the cause of the mushroom-body saturation.
* **Refractory breaks the lock but not the tonic state.** It breaks the one-spike-per-step
  lock, but KC activity remains tonic and input-insensitive at 20–40 ms. At ≥ 60 ms, rates
  fall to 2–3 Hz, but visual modulation stays within ±4 %.
* **Strong APL feedback (5×) sparsens KCs** (~1.3 Hz) and makes KCg-d visually selective. But
  MBON and DAN activity is then carried mostly by inputs other than KCg-d. The 418 KCg-d edges
  are ~6–8 % of target-MBON output: short of the gate, though ~20× better than stock.
* **Reducing KC gain** raises the KCg-d share of MBON drive (R5 passes at ≤ 0.35), but it does
  not restore visual selectivity.
* **Only 206 of 4,064 KCs are visually driven.** They are also a small fraction of MBON01/11
  input, so making them *both* stimulus-selective *and* dominant over MBON output is not
  achievable with these one-parameter corrections.

## Criteria (best-ranked point, calibration → confirmation)

| criterion | result |
|---|---|
| R1 avoid saturation | pass → pass (ceiling-normalised duty 0.05) |
| R2 avoid silence | pass → pass |
| R3 visual modulation | pass → pass |
| R4 DAN headroom (rule amplitude) | fail → fail (+3 to +4 pp; +12 to +25 pp at amplitudes ≥ 0.3) |
| R5 plastic leverage | fail → fail (−7.8 % / −6.6 %) |
| R6 stability | pass → pass |

## Required answers

1. **Does stock saturation reproduce?** Yes, bit-identically.
2. **What refractory support existed in FlyBrain?** A global `refractory` parameter (seconds),
   0 by default, which clamps v = 0 for k steps after a spike.
3. **Which refractory values were tested?** 0, 20, 40, 60, 80 and 100 ms.
4. **Where does gross KC/MB saturation break?** At 20 ms (normalised duty 0.67; whole-brain
   fraction at the ceiling 0.16 %).
5. **Does KCg-d regain visual modulation?** Not with refractory alone (≤ 4.3 %). Yes with
   20 ms + APL ×5.
6. **How large is the modulation?** +7 % to +13 % on confirmation seeds, with 36–60 % of cells
   modulated and d_z ≈ 0.3–0.6.
7. **Does PAM01 regain reinforcement headroom?** Partly. +3.4 to +7.8 pp at the rule amplitude
   (R4 fails); +12 pp or more at amplitude ≥ 0.3.
8. **PPL101?** The same pattern: +2 to +7.6 pp at the rule amplitude; +13 pp or more at
   ≥ 0.3.
9. **Does MBON01 leave saturation?** Yes (2–18 Hz depending on the point).
10. **MBON11?** Yes.
11. **Does zeroing the 418 edges now cause ≥ 10 %?** Only at refractory 60 ms (−10.8 %) and KC
    gain ≤ 0.35 (−10.5 %, −13.3 %). None of those points has visual modulation.
12. **How large is the effect at the visually responsive point?** −6.6 % to −7.8 %.
13. **Was APL intervention required?** Refractory failed, so APL was tested. It gave the only
    visually responsive point, but did not pass.
14. **Was global gain/baseline correction required?** It was tested (preregistered order) and
    did not pass.
15. **What is the least invasive passing configuration?** **None passed.** The best-ranked
    non-passing point is refractory 20 ms + APL ×5.
16. **Does it generalise to confirmation seeds?** Its pattern does: R1–R3 and R6 hold; R4 and
    R5 fail.
17. **Does the frozen 418-edge set remain unchanged?** Yes (hash `65dbc6264c04a2fa` in every
    run; T4 hash unchanged).
18. **Are the dynamics numerically stable?** Yes: finite voltages; ≤ 0.16 % of neurons at the
    ceiling; ≤ 5.5 % silent at every non-stock point.
19. **Can E26 Phases 3–6 now be rerun meaningfully?** Not unchanged. No point gives visual
    drive and ≥ 10 % plastic leverage together.
20. **What next?** See below.

## Next experiment

Do not rerun E26 Phases 3–6 unchanged, and do not run conditioning.

A follow-up should be separately preregistered. Two directions, with the E26 criteria kept as
they are (R5 = −10 %):

1. **Two-parameter operating points.** A small joint grid of refractory 20–60 ms × APL gain
   3–8, to look for the overlap of R3 and R5 that one-parameter families lack.
2. **A pre-registered fixed DAN amplitude** (e.g. 0.3) for R4. The current rule is degenerate
   outside saturation.

Alternatively, a separately audited question: whether a broader but still audit-derived
KC → MBON01/11 set (e.g. all γ-lobe KCs) is justified. That would be a new circuit, not a
post-hoc change here.

## Runtime

| stage | time |
|---|---|
| stock reproduction | ~5 min × 2 (before / after the assay fix) |
| refractory sweep | ~40 min |
| APL sweep | ~25 min |
| KC-gain sweep | ~25 min |
| confirmation (best + stock, 10 seeds) | ~20 min |
| `sensory_input` diagnostic | ~6 min |
| analysis | < 1 min |
| tests | < 1 min (CPU), ~21 s (GPU) |

## Files

`preregistration.md`, `repository-metadata.json`, `stock-reproduction.json`,
`refractory-sweep.json`, `apl-sweep.json`, `kcgain-sweep.json`, `candidate-ranking.json`,
`frozen-operating-point.json` (non-passing; descriptive only), `confirmation-results.json`,
`diagnostic-sensory-input-off.json`, `mechanistic-diagnostics.json`.

Figures: `captures/experiment-27/01–12*.svg`.

Code:

* `run_mb_operating_regime.py`, `analyze_mb_operating_regime.py`, `test_mb_operating_regime.py`
* `run_mb_plasticity.py` — harness now accepts `refractory` / `sensory_input`; defaults
  unchanged
