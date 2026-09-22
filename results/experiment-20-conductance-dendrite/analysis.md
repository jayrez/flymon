# Experiment 20 — Conductance / compartmental dendritic motion readout

**Verdict: FAIL. Mechanistic category: E** — the Experiment-19 `OPPONENT_SIGNED` oracle
is **not** reproduced by this minimal biophysical implementation.

The headline is not merely that the model missed a threshold. It is that the small
directional bias the model *does* produce shows **none of the dependencies a genuine
spatially ordered, sign-dependent opponent mechanism must have**.

## Result at a glance

| model | held-out mean DSI | positive subtypes | coherent | notes |
|---|---:|---:|---:|---|
| C0 single compartment | **+0.0067** | 6/8 | 2/8 | sign-preserving conductances only |
| C1 two compartments | **+0.0057** | 6/8 | 2/8 | compartmentalisation adds nothing |
| C2 two compartments + output threshold | **+0.0246** | 5/8 | 2/8 | selected |
| **E19 `OPPONENT_SIGNED` oracle** | **+0.3562** | **8/8** | **8/8** | computational control |

The selected model reaches **6.9 %** of the oracle, against a preregistered threshold of
+0.15 (needing ≥7/8 positive). Calibration already recorded
`met_preregistered_threshold: false`; the held-out run confirms it. Two configurations
were within 0.01 of the selected one, so this is not a knife-edge parameter choice.

## Why this is category E rather than D

The preregistration lists both **D** ("conductance dynamics improve selectivity but miss
the threshold") and **E** ("the oracle cannot be reproduced by this minimal biophysical
implementation") but specifies **no precedence rule between them**. The first version of
the analysis script used an *unpreregistered* heuristic — `C2 > C0` improvement ⇒ D —
which would have returned **D**. That heuristic is superseded here, and the discrepancy
is recorded explicitly in `gate-status.json`
(`naive_improvement_only_category`, `category_rule_note`). No preregistered threshold or
primary endpoint was changed.

The reason: "conductance dynamics improve selectivity" is only a meaningful claim if the
improvement depends on the things the hypothesised mechanism is made of. It does not.

| ablation | held-out mean DSI | relative change | mechanism implication |
|---|---:|---:|---|
| selected model (observed) | +0.0246 | — | — |
| **arm swap** (exchange which arm feeds which compartment) | +0.0241 | **−2.0 %** | **spatial ordering is irrelevant** |
| **inhibitory reversal neutralised** (`E_I → E_L`) | +0.0230 | **−6.5 %** | hyperpolarising inhibition barely contributes |
| **synaptic sign destroyed** (all input → `g_E`) | +0.0191 | **−22.3 %** | sign carries little of the effect |
| temporal flattening (`τ_slow = τ_fast`) | +0.0094 | −61.6 % | timing does matter |
| coupling removed (`g_c = 0`) | +0.0024 | −90.2 % | coupling does matter |

**The arm-swap result is the decisive one.** A delay-and-compare or compartmental
opponent mechanism is *defined* by which arm occupies which position: exchanging them
should abolish or invert the preference. Changing it by 2 % means the model is not
computing anything spatially ordered.

Two further results point the same way:

* **Geometry shuffle leaves 77 % of the effect intact.** Over 200 permutations the null
  mean is **+0.0144** against an observed **+0.0187** (speed subset; sd 0.00137,
  z = 3.13, p = 0.005). The p-value is "significant" only because the null is extremely
  tight; the *effect size* barely moves when the true optic-column geometry is destroyed.
  For contrast, Experiment 19's geometry null was +0.0055 against an observed −0.1053.
* **Sign and inhibition — the two ingredients E19 identified as essential — are nearly
  inert here** (−22 % and −6.5 %).

So the ~90 % coupling dependence and ~62 % timing dependence should be read carefully:
they show that coupling and temporal filtering generate the weak nonlinear response, but
**coupling dependence alone is not evidence that the opponent computation was
implemented**. A generic coupled-RC-plus-threshold cascade produces a small directional
bias for any spatially smooth drifting stimulus; that is what this appears to be.

## Per-subtype held-out results

| subtype | conductance DSI | 95 % CI | signResp | p (Holm) | oracle DSI | oracle signResp |
|---|---:|---:|---:|---:|---:|---:|
| T4a | **−0.1059** | [−0.121, −0.090] | 0.25 | 9e−40 | +0.3529 | 0.83 |
| T4b | +0.1078 | [+0.090, +0.126] | 0.72 | 3.6e−30 | +0.3512 | 0.82 |
| T4c | −0.0045 | [−0.020, +0.011] | 0.49 | 0.77 (n.s.) | +0.3401 | 0.81 |
| T4d | +0.0534 | [+0.036, +0.071] | 0.61 | 3.1e−08 | +0.3266 | 0.80 |
| T5a | **−0.1360** | [−0.152, −0.121] | 0.20 | 2e−61 | +0.3165 | 0.79 |
| T5b | +0.1853 | [+0.171, +0.201] | 0.88 | 3.4e−104 | +0.4124 | 0.91 |
| T5c | +0.0312 | [+0.015, +0.048] | 0.58 | 1.9e−05 | +0.4148 | 0.91 |
| T5d | +0.0652 | [+0.046, +0.085] | 0.60 | 1.5e−07 | +0.3354 | 0.85 |

Five subtypes positive, **two significantly inverted** (T4a, T5a), one null. The oracle
is uniformly positive across all eight with far higher sign consistency. Note that
T4a and T5a — the two inverted subtypes — are inverted in the *conductance* model but
not in the oracle, so this is not a property of the geometry.

## Oracle comparison

* **Amplitude**: 0.0246 vs 0.3562 — a **14.5×** shortfall (ratio 0.069).
* **Per-neuron DSI correlation**: Pearson −0.341 … +0.450 (mean **+0.159**),
  Spearman −0.480 … +0.446, normalised RMSE 1.005 … 1.588. Weak, and *negative* for T4a.
* **Temporal cross-correlation** of preferred−null traces: peak 0.413 … 0.914
  (mean **+0.693**), but the lags are scattered — **0, −4, −11, −13, +50 frames**. Four
  subtypes align at zero lag while others are displaced by a fifth of the sequence, so
  the temporal resemblance is not a consistent mechanistic match.

## Controls and generalisation

frozen **0.0** · gray **0.0** · shuffle 4.4e−05 vs dynamic 4.1e−04 (**−89 %**) ·
reverse 5.7e−04 (≳ dynamic, as expected — reversal is still motion). The model is
genuinely motion-driven and silent on static input.

| axis | mean DSI | positive |
|---|---:|---:|
| speed | +0.0187 | 4/8 |
| width | +0.0330 | 6/8 |
| contrast | +0.0279 | 6/8 |
| position | +0.0332 | 6/8 |

Uniformly small everywhere; no regime rescues it.

## ON/OFF specificity — the one clear success

| | T4a | T4b | T4c | T4d | T5a | T5b | T5c | T5d |
|---|---|---|---|---|---|---|---|---|
| favours | ON | ON | ON | ON | OFF | OFF | OFF | OFF |

T4 prefers ON motion by ~40–60× (3.2e−04 vs 8e−06) and T5 prefers OFF by ~13× (8.0e−04
vs 5.7e−05), with **no subtype-specific coefficients**. The conductance mapping — weight
sign selects the channel, reversal potential supplies the polarity — reproduces the
pathway asymmetry cleanly (T4 fast-excitatory/slow-inhibitory; T5 both arms excitatory)
and does so more strongly than Experiment 19. Pathway polarity is therefore *not* what
is missing; directional ordering is.

## Answers to the required questions

1. **Sign-preserving single-compartment conductance integration → DS?** Barely: +0.0067.
2. **Does compartmentalisation improve it?** **No** — C1 (+0.0057) ≈ C0 (+0.0067).
3. **Is coupling necessary?** For the weak effect that exists, yes (−90 % without it) —
   but see the caveat above; this is not evidence of the opponent computation.
4. **Is an output nonlinearity necessary?** It is the only thing that helps (C2 +0.0246,
   ~4× over C0/C1), yet still 14.5× short of the oracle.
5. **Threshold reached?** **No** (+0.0246 vs +0.15).
6. **Coherent subtypes?** 2/8 coherent; 5/8 merely positive (need ≥7/8).
7. **T4 signs correct?** No — T4a inverted, T4c null.
8. **T5 signs correct?** No — T5a inverted.
9. **ON/OFF preserved?** **Yes**, strongly.
10. **Generalises across speeds?** No (+0.0187, 4/8).
11. **Across contrasts?** No (+0.0279, 6/8).
12. **Geometry shuffle collapses it?** **No** — 77 % survives.
13. **Temporal flattening collapses it?** Substantially (−62 %).
14. **Sign destruction collapses it?** **No** (−22 %).
15. **Inhibitory-reversal neutralisation collapses it?** **No** (−6.5 %).
16. **Arm swap reverses/weakens it?** **No** (−2 %) — the decisive negative.
17. **Similarity to the oracle?** Weak: amplitude 6.9 %, per-neuron Pearson +0.159,
    temporal peak xcorr +0.693 at inconsistent lags.
18. **Is the oracle reproducible without coding the opponent equation?** **Not by this
    model.** (A regression test asserts the module contains no opponent expression and
    that its output is not a rescaled copy of the oracle.)
19. **What mechanism is actually necessary?** Still unresolved. Conductance sign and
    inhibitory reversal — necessary in E19's algebraic framing — are nearly inert once
    embedded in this passive two-compartment membrane. What the oracle needs, and what
    this model lacks, is an interaction whose outcome depends on **which arm is where**.
20. **Justified to proceed to VP→DN?** **No.**

## Conclusion

A minimal sign-preserving two-compartment conductance model does not reproduce the
Experiment-19 signed-opponent computation. Although compartment coupling and the output
nonlinearity can generate a small directional bias, the effect is far below the
preregistered threshold and lacks the expected dependence on synaptic sign, inhibitory
reversal, and spatial arm ordering. The E19 oracle therefore remains a computational
constraint whose biological implementation is unresolved.

The one positive transfer is pathway polarity: routing MaleCNS weight sign into
excitatory/inhibitory channels with distinct reversal potentials yields clean, strong
ON/OFF specificity across all eight subtypes with no per-subtype tuning.

## Limitations

The membrane model is deliberately minimal: passive, two compartments, symmetric
coupling, no active conductances, no location-dependent synaptic placement within a
compartment. **Compartment assignment was a stipulated model hypothesis**, since
MaleCNS v1.0 carries no subcellular annotation for T4/T5 — and the arm-swap ablation
shows this particular architecture is insensitive to it, which is itself the finding.
`τ_fast/τ_slow` were fixed to the oracle's values to isolate the readout rule rather
than re-searching temporal parameters. flyvis remains unresolved (F3 from E19).

## Recommended next experiment (proposed, not implemented)

The missing ingredient is **structural asymmetry that makes arm order matter**. A
symmetric passive cable cannot supply it. Candidate mechanisms for a follow-up, keeping
E19 `OPPONENT_SIGNED` as the oracle and the same geometry/stimuli/metrics:

* **location-dependent inhibition** — inhibition placed *between* the excitatory input
  and the readout site (on-path shunting), which is directional by construction;
* **asymmetric / non-reciprocal coupling** between compartments;
* **nonlinear dendritic subunits** with a saturating local nonlinearity before summation;
* **active conductances** (regenerative or voltage-dependent) that amplify one temporal
  order over the other;
* **multi-compartment cable** with graded electrotonic distance along the arm axis.

The diagnostic to preregister for that experiment is the arm-swap ablation itself: any
candidate that does not change substantially under arm swap should be rejected early,
before its DSI is even examined.

## Reproduction

```sh
python run_conductance_dendrite_experiment.py calibrate
python run_conductance_dendrite_experiment.py heldout
python analyze_conductance_dendrite_experiment.py
```
