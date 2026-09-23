# Experiment 22 — Fixed adapting background, pre-adaptation and tonic excitation

**Scientific verdict: FAIL. Mechanistic category: E (preregistered default). Gameplay
readiness: NO-GO.**
**Attribution of the E21 lead (preregistered rule): mostly real.** Its direction
selectivity survives a fixed reference (R2/R0 retention 1.04, held-out). Its inverted ON/OFF
preference was an artefact of sequence-mean centring. But under the fixed reference the
direction selectivity no longer depends on the fast/slow filter asymmetry, so it fails the
preregistered temporal gate.

Preregistration: `preregistration.md` (frozen before any fixed-reference calibration). Base:
branch `worktree-bridge-cse_014eULVFzQa6eL4wV7d1hV9L`, HEAD `74bf6f6` (E21), `origin/main`
`88522ee`. No gameplay, RAM, reward, RL, labels or per-subtype parameters.
`column_motion.py` and every earlier module and artifact are unchanged.

## 1. What was done

* **R0 reproduction.** The new module (`flymon/adapted_reference.py`) reproduces the E21
  exploratory A1 exactly. Calibration T4 was +0.249659 and all-8 +0.126525. Held-out T4 was
  +0.187944. All three are bit-identical to the saved E21 artifacts.
* **Luminance scale.** The retina decodes sRGB to linear light. Display gray 0.5 reads
  back as linear **0.2122**, and E19 bars sit on black (ON) or white (OFF) fields. There is
  no common background in the E19 bank. E22 therefore sets the background in **linear**
  light. The primary background is 0.5, measured as **L0 = 0.49693** after uint8
  quantisation. Bars sit at Weber ±c with E19 trajectories.
* **Step-polarity audit.** It passed for R1 and for R2 at L0 ∈ {0.4, 0.5, 0.6}. Increments
  drive Mi1/Mi4 positive. Decrements drive Mi9/Tm1/Tm2/Tm9 positive. Pre-step signal is
  exactly 0 after adaptation, and a static L0 frame gives zero contrast. Under R0 the same
  ON step produces a spurious **pre-step** Mi1 signal of −0.166, because the zero point is
  set by the future.
* **Calibration funnel.** 288 configurations across three levels. A1 is fixed background +
  tonic excitation with cold filters. A2 adds 40-frame pre-adaptation. A3 is A2 plus tonic
  Mi9 and Mi4 set separately. The grid covered G, beta, r0_exc and L0. Gates ran in order
  before any ranking: ON/OFF → static → geometry (20 perms) → temporal (≥ 40 %).
* **Held-out run, once,** with the frozen model. The held-out analyses also include the
  preregistered reference comparison (R0/R0g/R1/R2/R3) at the unchanged E21 parameters and
  at the frozen parameters, initial-condition tests, ablations B1–B10, inhibition
  decomposition, a 200-permutation geometry null and the oracle comparison.

## 2. Calibration gate outcomes

| outcome | n |
|---|---|
| rejected at ON/OFF gate | 8 (all A3) |
| rejected at static gate | 0 |
| rejected at geometry gate | 7 |
| rejected at temporal gate | **236** (all 72 A1/A2, 164 A3) |
| passed all gates | 37 (all A3; **none with 4/4 T4 positive**) |

With the fixed reference, 280/288 configurations prefer ON in all four T4 subtypes (E21: 0/4).
Every A1/A2 configuration fails the temporal gate. At the E21 parameters, R1 calibration T4 is
+0.192 (4/4), but temporal flattening removes only 16 %.

**Frozen:** `A3|fixed|L0.5|ad40|G16|b2|r0e2|mi90.5|mi40`, calibration T4 +0.052 (3/4), geometry
residual 0.06, temporal drop 0.64. *Clarification:* the preregistration did not specify what to
do when gate survivors exist but none has 4/4 positive. The runner froze the best gate
survivor, and this was fixed in code before any held-out data were touched.

## 3. Reference-condition comparison (held-out)

At the **unchanged E21 A1 parameters** (the direct test):

| cond | T4 mean DSI | T4 positive | T4 ON>OFF | T5 OFF>ON | geometry residual (50) | temporal-flat T4 | flat reduction |
|---|---|---|---|---|---|---|---|
| R0 seq-mean, E19 bank | +0.1879 | 4/4 | **0/4** | 0/4 | −0.03 | −0.0215 | 111 % |
| R0g seq-mean, gray bank | +0.1911 | 4/4 | **0/4** | 0/4 | −0.04 | −0.0261 | 114 % |
| R1 fixed, cold filters | +0.1887 | 4/4 | **4/4** | 4/4 | −0.01 | +0.1524 | 19 % |
| R2 fixed, adapted | **+0.1962** | 4/4 | **4/4** | 4/4 | −0.004 | +0.1609 | 18 % |
| R3 = R2 without tonic excitation | +0.0027 | 3/4 | 4/4 | 4/4 | 0.42 | +0.0062 | — |

At the frozen parameters: R0 +0.050 (0/4 ON), R0g +0.038 (0/4 ON), R1 +0.037, R2 +0.031,
R3 +0.006. Under all three fixed-reference conditions T4 prefers ON 4/4 and 3/4 subtypes are
positive.

R0g versus R0 shows that the gray background alone changes nothing. R1/R2 versus R0g show
that the **reference alone** fixes ON/OFF and leaves the DSI magnitude intact.

## 4. Frozen model (primary, held-out)

| subtype | DSI | 95 % CI | signResp | ON>OFF | Holm p (sign test) |
|---|---|---|---|---|---|
| T4a | −0.0354 | [−0.049, −0.022] | 0.31 | yes | 4e−31 (wrong sign) |
| T4b | +0.0951 | [+0.079, +0.111] | 0.76 | yes | 4e−63 |
| T4c | +0.0373 | [+0.022, +0.053] | 0.63 | yes | 3e−16 |
| T4d | +0.0276 | [+0.015, +0.041] | 0.56 | yes | 6e−5 |

T4 mean +0.0311 (3/4), all-8 +0.0166. Criteria:

| criterion | result | met |
|---|---|---|
| 4/4 ON>OFF | 4/4 | ✓ |
| T4 mean ≥ 0.15 | 0.031 | ✗ |
| 4/4 positive | 3/4 | ✗ |
| geometry residual < 0.40 (200 perms) | 0.14 (z 10.8, p 0.005) | ✓ |
| temporal-flat reduction ≥ 40 % | 99 % | ✓ |
| stable under pre-adaptation | adapted − cold = −0.006, both 4/4 ON | ✓ |

The model is not A. It is not C (DS is weak and ON/OFF is correct). It is not B (3/4
positive, T4 < 0.05). It is not D, because R1/R2 at the E21 parameters stay at ~0.19. That
leaves **E** by the preregistered rule. The secondary mechanistic-lead criteria are also not
met.

**T5 (descriptive, no CT1, no retuning):** T5 mean +0.002. T5a −0.069, T5b +0.062,
T5c +0.013, T5d +0.002. T5 is OFF>ON 4/4 under every fixed-reference condition (0/4 under R0).

## 5. Initial conditions

Frozen model: adapted-40 +0.0311, adapted-30 +0.0311, adapted-60 +0.0311, cold filters
+0.0373, cold filters and cold membrane +0.0373. All are 4/4 ON>OFF.

At E21 parameters: R1 cold +0.1887 versus R2 adapted +0.1962.

Pre-adaptation changes DSI by < 0.01. Neither the sign nor the ON/OFF result depends on
initialisation. The adapted rest equals the tonic rest exactly (tested). The membrane is
effectively instantaneous at these conductances, so a cold membrane makes no difference.

## 6. Ablations (frozen model, held-out T4 mean DSI)

| ablation | T4 | pos | ON>OFF |
|---|---|---|---|
| frozen | +0.0311 | 3/4 | 4/4 |
| B1 sequence-mean restored | +0.0378 | 4/4 | **0/4** |
| B2 tonic excitation removed | +0.0058 | 3/4 | 4/4 |
| B3 cold start | +0.0373 | 3/4 | 4/4 |
| B4 temporal flat | +0.0002 | 2/4 | 4/4 |
| B5 geometry shuffle (200) | null +0.0044 ± 0.0024, residual 0.14 | | |
| B6 Mi9 removed | **+0.1788** | 4/4 | 4/4 |
| B7 Mi4 removed | +0.0112 | 3/4 | 4/4 |
| B8 Mi9 and Mi4 removed | +0.0020 | 3/4 | 4/4 |
| B9 gray / frozen / shuffle / reverse | T4 output 0 / ~0 / ~0 / ~0; static frozen-mid T4 DSI −0.015 | | |
| B10 full-field steps | T4 ON ≫ OFF, T5 OFF ≫ ON (all 8) | | |

Inhibition decomposition (held-out T4):

| composition | T4 |
|---|---|
| excitation only | +0.002 |
| + Mi9 | +0.011 |
| + Mi4 | **+0.179** (4/4) |
| + both (frozen) | +0.031 |

Tonic-rate decomposition:

| r0 Mi9 / Mi4 | T4 |
|---|---|
| 0 / 0 | +0.179 |
| 0 / 0.5 | +0.234 (3/4 ON) |
| 0 / 1 | +0.164 |
| 0.5 / 0 | +0.031 |
| 1 / 0 | +0.014 |
| 1 / 1 | +0.012 |

**Mi4 carries the directional signal and tonic Mi9 suppresses it.** The temporal gate
favoured tonic Mi9 because Mi9 is what makes the residual DS depend on tau_slow.

## 7. Why the fixed-reference DS survives temporal flattening

Exploratory, calibration only: `exploratory-temporal-diagnostic.json`. At E21 parameters
under R2:

* Mi1 alone gives +0.001. Mi1+Mi9 gives +0.001. Mi1+Mi4 gives +0.197, all of it.
* Removing all filtering (tau = 0) gives 0.000.
* Equal filters (tau_slow = tau_fast = 4) give +0.164, keeping 83 %.
* An instantaneous membrane changes nothing.

So the fixed-reference DS needs Mi4's spatial offset and a persistence time course on its
input. It does not need Mi4 to be *slower* than Mi1. The best reading is a spatially offset
inhibitory veto (Barlow–Levick-like) acting on a tonically depolarised, rectified output.
Under the sequence-mean reference the same parameters instead depended entirely on the
fast/slow asymmetry (flattening: 111 % loss). **The E21 lead's apparent timing dependence was
itself partly a property of the sequence-mean baseline.**

## 8. Oracle comparison (benchmark only)

| model | held-out T4 mean DSI | T4 ON>OFF | geometry residual | temporal-flat effect |
|---|---|---|---|---|
| E19 `OPPONENT_SIGNED` (E19 bank / gray bank) | +0.343 / +0.342 | — | — | — |
| E20 C2 frozen | +0.013 | 4/4 | 0.77 | — |
| E21 frozen | +0.026 | 0/4 | — | — |
| E21 exploratory A1 (= R0) | +0.188 | 0/4 | −0.03 | −111 % → −0.02 |
| **E22 E21-params R2** | **+0.196** | **4/4** | −0.004 | −18 % |
| E22 frozen | +0.031 | 4/4 | 0.14 | −99 % |

Per-neuron correlation of the E22 frozen model with the oracle on the gray bank: T4a −0.04,
T4b 0.63, T4c 0.46, T4d 0.34. The model is not a rescaled oracle (tested). R2 at E21
parameters reaches 57 % of the oracle T4 DSI with correct polarity. That is the closest any
non-oracle model has come.

## 9. Answers to the preregistered questions

1. **Does the E21 lead survive a fixed reference?** Yes. Held-out T4 +0.196 (R2) versus
   +0.188 (R0), 4/4 positive.
2. **Is ON/OFF restored?** Yes. T4 ON>OFF 4/4 and T5 OFF>ON 4/4 under R1, R2 and R3, versus
   0/4 under R0 and R0g.
3. **Was the ON/OFF inversion caused by sequence-mean centring?** Yes. R0g (gray bank,
   sequence mean) keeps it inverted, and R1 (same bank, fixed reference) fixes it. B1
   re-inverts the frozen model.
4. **Does the step-polarity audit pass?** Yes, for all fixed-reference conditions. R0 shows
   spurious pre-step drive.
5. **Does pre-adaptation matter?** Barely (< 0.01 DSI). It does not change sign or polarity.
6. **Is tonic excitation required?** Yes. R3 +0.003, B2 +0.006.
7. **Is geometry required?** Yes. Residuals are near 0 at E21 parameters and 0.14 for the
   frozen model (200 perms, z 10.8).
8. **Is the fast/slow temporal asymmetry required?** Not for the strong fixed-reference
   signal: flattening removes only 18–19 %. Temporal filtering itself is required (tau = 0
   gives 0, exploratory).
9. **Which inhibitory input carries DS?** Mi4. Mi9 alone contributes ~0, and tonic Mi9
   suppresses DS.
10. **Did any configuration pass all gates with ≥ 0.15?** No. The best gate survivor reached
    calibration +0.052 (3/4).
11. **Frozen held-out result?** +0.031 (3/4), ON 4/4. The primary criteria are not met.
12. **Category?** E by the preregistered precedence. D is excluded because the E21-parameter
    fixed-reference DS persists.
13. **Was the E21 signal caused by geometry and timing in a meaningful contrast reference,
    or by sequence-mean × tonic × rectification?** Its **directional magnitude** is
    geometry-dependent and survives a meaningful reference: *mostly real*, retention 1.04.
    Its **ON/OFF inversion** and its **fast/slow timing dependence** were products of the
    sequence-mean reference.
14. **Is the result stable across L0?** Yes. Calibration results for L0 0.4, 0.5 and 0.6
    differ by ≤ 0.008 for every configuration.
15. **T5?** Polarity is correct under a fixed reference, and DS is near zero (+0.002).
    T5-specific mechanisms are still needed.
16. **Controls?** Gray, frozen, shuffle and reverse T4 output is ≈ 0. Static frozen-mid DSI
    is −0.015.
17. **Holm significance?** Frozen T4b, T4c and T4d are significantly positive. T4a is
    significantly negative.
18. **Generalisation axes (frozen)?** Speed +0.031, width +0.049, contrast +0.027, position
    +0.048. Gratings are not scored for T4 by the E19 metric.
19. **Oracle gap?** Frozen 9 % of the oracle. E21-params R2 57 %.
20. **Did any step use held-out data for selection?** No. Selection used calibration only,
    and the held-out run was performed once.

## 10. Mechanistic conclusion

**Mostly real, with qualifications.** The strong T4 directional signal does not come from
sequence-mean centring: it is reproduced with a fixed adapting background, correct ON/OFF
polarity in all T4 and T5 subtypes, pre-adaptation and zero spurious baseline. What the
sequence-mean reference *did* cause was:

* the inverted ON/OFF preference, and
* the signal's dependence on the fast/slow filter asymmetry.

Under a meaningful reference, the directional signal comes from **tonic Mi1 excitation plus a
spatially offset Mi4 inhibitory input with a persistent time course**. The model needs no
opponent product, and it fails the preregistered fast/slow temporal gate. The preregistered
funnel therefore excluded it and froze a weak A3 model. That is why the formal verdict is
FAIL / E even though the question's main hypothesis ("survives a fixed reference") holds.

## 11. Gameplay-readiness implication

**NO-GO** by the preregistered rule: GO needs A, and CONDITIONAL GO needs B. No gameplay or
generation experiment was started.

Substantively, the E21-parameter R2 model has correct polarity and geometry-dependent T4 DS
of ~0.19, but it has not been frozen through a preregistered gate it passes. A next
experiment should preregister a timing criterion that fits a spatial-offset veto: for
example tau = 0 abolition and arm-position swap, rather than tau_slow = tau_fast. It should
freeze the Mi1 + Mi4 + tonic-excitation model prospectively and address T5 separately. It
should not proceed to VP→DN before then.

## 12. Runtime

R0 reproduction 10 s. Calibration funnel 3,887 s (288 configurations, 20 CPU workers).
Held-out 2,055 s:

* reference conditions 901 s
* ablations 377 s
* inhibition decomposition 313 s
* 200-perm null 201 s

CPU only, deterministic.

## Files

`preregistration.md`, `run-metadata.json` (repo metadata, reference definitions, adaptation
settings), `polarity-audit.json`, `calibration-grid.json`, `gate-outcomes.json`,
`selected-config.json`, `heldout-results.json` (primary, reference conditions, initial
conditions, inhibition decomposition, oracle comparison, axes), `ablation-results.json`,
`statistics.json`, `verdict.json`, `heldout-per-neuron.npz`, `traces.npz`,
`exploratory-temporal-diagnostic.json`. Figures: `captures/experiment-22/01–13*.svg`.

Code:

* `flymon/adapted_reference.py`
* `run_fixed_background_experiment.py`
* `analyze_fixed_background_experiment.py`
* `diagnose_fixed_background_temporal.py` (exploratory)
* `test_fixed_background.py` (28 tests)
