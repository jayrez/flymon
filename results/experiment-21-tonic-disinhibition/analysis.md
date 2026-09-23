# Experiment 21 — Tonic shunting inhibition and release from inhibition in T4

**Overall verdict: FAIL. Mechanistic category: E** (no configuration passes the mechanistic
gates). **T4 verdict: FAIL. T5 verdict: does not transfer.**

Release from tonic inhibition **does occur** in this reconstruction, and it is even
directional — but it is far too small to open an input-resistance window during excitation,
because Mi1 excitation adds 5–20× more conductance than Mi9 release removes. Tonic inhibition
therefore does not explain T4 direction selectivity here. An unexpected, unpreregistered
observation (tonic *excitation* without tonic inhibition gives T4 DSI ≈ 0.19–0.25) is reported
separately as a lead, together with the reason it is not a credible T4 mechanism: it inverts
ON/OFF polarity.

## Base and reproduction

`origin/main` = `88522ee` (PR #6); E20 `e60eaf3` and E19 `859e1dc` in ancestry; branch
fast-forwarded (no rewrite). **D0 reproduces E20 exactly**: calibration mean DSI
0.031202248 vs recorded 0.031202248; held-out C0 +0.0067 and C2 +0.0246 (as recorded).

## Motivation, verified

E20 (`conductance_dendrite.py:95`) used `max(s, 0)` on a mean-centred contrast signal, so
inhibition started at zero and could not be released; measured E20 inhibition peaked at
0.087 × g_L. E21 gives every presynaptic class a tonic rate that the stimulus moves up or down,
routes conductance by MaleCNS weight sign, and keeps every conductance ≥ 0.

## The mechanism screen (calibration only; 180 configurations)

| gate | outcome |
|---|---|
| **1 — inhibitory release** | failed by the 18 zero-baseline configs (by construction); **passed by all 162 others** (Mi9 ≤ 0.9 × baseline for ≥ 3 frames in 95–98 % of driven pairs) |
| **2 — R_in window overlapping excitation** | **failed by all 162** — median overlap **exactly 0.00** in every configuration |
| 3 — inhibition clamp causal | never reached |

R_in does rise (median peak up to +134 %), but never during excitation. The Gate-2 diagnostic
(`gate2-diagnostic.json`) shows why, inside the excitation window:

| configuration | Mi9 conductance removed | Mi1 added | Mi4 added | removed ÷ added | R_in in window |
|---|---:|---:|---:|---:|---:|
| frozen D1, preferred | 9.01 | 122.1 | +10.2 | **0.073** | −74 % |
| frozen D1, null | 6.19 | 123.3 | +12.2 | 0.047 | −75 % |
| D2 G16 β0.5 r0 2, preferred | 6.47 | 30.5 | +2.5 | **0.212** | −44 % |
| D2 G16 β0.5 r0 2, null | 4.24 | 30.8 | +3.0 | 0.127 | −47 % |
| D2 G4 β1 r0 1, preferred | 2.15 | 15.3 | +1.3 | **0.140** | −74 % |

Two facts coexist. **Release is directional**: Mi9 removes ~1.45× more conductance in the
preferred than the null direction, peaking 0–3 frames before excitation — the timing the
hypothesis requires. **But it is overwhelmed**: excitation adds 5–20× more conductance than
release removes (ratio > 1 in ≤ 5 % of pairs), and Mi4 — ON-driven and anatomically near Mi1 —
adds inhibition at the same moment. Two structural reasons: the MaleCNS weight onto T4 from Mi1
(median Σ|w| ≈ 0.07–0.08) is 2–5× that of Mi9 (0.017–0.035), and release is capped at the tonic
baseline (it can only remove what is there) while excitation grows with the stimulus.

The held-out run reproduces the same picture: release fraction 0.985, overlap 0.00, R_in peak
+170 %.

## Frozen model and held-out results

No configuration passed the gates, so the preregistered fallback froze the best calibration
T4-DSI configuration, `D1_UNIFORM_TONIC | G 16 | β 2 | r0 1 (exc and inh) | E_I −0.2`, for
**descriptive** evaluation. Category E is fixed by the preregistered rule regardless of its DSI.

| model | T4 mean DSI | T4 positive | all-8 mean DSI | all-8 positive | notes |
|---|---:|---:|---:|---:|---|
| E20 C0 (D0, reproduced) | +0.0054 | — | +0.0067 | 6/8 | zero-baseline conductance |
| E20 C2 (D0, reproduced) | +0.0127 | — | +0.0246 | 5/8 | E20 selected |
| **E21 frozen (D1)** | **+0.0258** | **3/4** | +0.0139 | 5/8 | gate-rejected fallback |
| E21 best D2 (tonic inhibition) | +0.0164 | 3/4 | +0.0062 | 5/8 | the hypothesis itself |
| E21 best D3 (bounded) | +0.0151 | 3/4 | +0.0055 | 5/8 | bounded transfer |
| **E19 `OPPONENT_SIGNED` oracle** | **+0.3427** | **4/4** | **+0.3562** | **8/8** | benchmark |

D2 — tonic inhibition alone — reaches +0.016, barely above E20's zero-baseline model
(calibration +0.0118 vs +0.0101). The frozen model is 7.5 % of the oracle on T4.

### T4 subtypes (held-out, frozen)

| subtype | DSI | 95 % CI | signResp | p (Holm) |
|---|---:|---:|---:|---:|
| T4a | −0.0042 | [−0.011, +0.002] | 0.51 | 0.56 (n.s.) |
| T4b | +0.0513 | [+0.045, +0.057] | 0.77 | 5e−69 |
| T4c | +0.0298 | [+0.023, +0.036] | 0.69 | 2e−34 |
| T4d | +0.0264 | [+0.020, +0.033] | 0.64 | 3e−20 |

No T4 subtype is significantly inverted, but T4a is null and none approaches 0.15.

### T5 subtypes (descriptive)

| subtype | DSI | 95 % CI | signResp | p (Holm) |
|---|---:|---:|---:|---:|
| T5a | **−0.0945** | [−0.103, −0.086] | 0.16 | 5e−104 |
| T5b | +0.0814 | [+0.074, +0.089] | 0.83 | 7e−107 |
| T5c | +0.0218 | [+0.014, +0.030] | 0.61 | 2e−12 |
| T5d | −0.0006 | [−0.011, +0.010] | 0.53 | 0.18 |

T5 mean +0.002. Expected: every resolved T5 input (Tm1, Tm2, Tm9) is excitatory and CT1 is
unplaced, so there is no inhibitory class to release. No rescue parameters were added.

## Ablations (held-out, frozen configuration)

| ablation | T4 mean DSI | T4 positive | change vs frozen |
|---|---:|---:|---|
| frozen | +0.0258 | 3/4 | — |
| **A1 no tonic inhibition** | **+0.1879** | **4/4** | **+628 %** — see exploratory section |
| A2 no release (inhibition may rise, not fall) | +0.0230 | 4/4 | −11 % |
| **A3 inhibition clamped at baseline** | +0.0027 | 3/4 | −90 % |
| A4 inhibitory reversal → E_L | +0.0238 | 3/4 | −8 % |
| A5 temporal flat | +0.0080 | 3/4 | −69 % |
| A7 remove Mi1 | +0.0428 | 3/4 | +66 % |
| A8 remove Mi9 | +0.0174 | 2/4 | −33 % |
| A9 remove Mi4 | +0.0129 | 3/4 | −50 % |
| A6 geometry shuffle (200 perm.) | null +0.0045 ± 0.0015 | — | observed +0.0242, z 13.3, p 0.005, **residual 0.19** |
| A10 frozen / gray | **0.0 / 0.0** | — | shuffle 0.018, reverse 0.031 |

Read together: **preventing release (A2) costs only 11 %**, so release is not what carries the
frozen model's weak DS. Clamping *all* inhibition (A3, −90 %) does matter, which means the
inhibitory classes shape the response — but through their stimulus-evoked *increases* (Mi4
especially, −50 % when removed), not through release. Removing Mi1 *raises* DSI, because pure
disinhibition-driven depolarisation from a hyperpolarised rest is more directional than the
excitation-dominated response. The weak effect does depend on true geometry (residual 0.19) and
on timing (−69 %).

## ON/OFF specificity — lost in the frozen model

All four T4 subtypes respond **more to OFF** than ON motion (e.g. T4a 0.031 vs 0.053) and all
four T5 **more to ON** (T5a 0.0095 vs 0.0065) — the reverse of E19 and E20, both of which
preserved correct polarity. The cause is the tonic *excitatory* baseline: with a mean-centred
contrast signal, an OFF bar on a bright background keeps Mi1 slightly above its tonic rate for
most of the sequence, depolarising the cell, while an ON bar on a dark background holds it below.

## Exploratory, not preregistered: tonic excitation without tonic inhibition

A1 produced a configuration — tonic excitation `r0_exc = 1`, no tonic inhibition — that was
**never a calibrated candidate** (the D1 grid tied `r0_exc = r0_inh`) and **fails Gate 1 by
construction**. It is therefore not eligible for selection and does not change the verdict.
Characterised in `exploratory-a1-tonic-excitation.json`:

| | calibration | held-out |
|---|---:|---:|
| T4 mean DSI | **+0.2497 (4/4)** | **+0.1879 (4/4)** |
| geometry residual (50 perm.) | — | **−0.03** (fully geometry-dependent) |
| temporal-flat T4 | — | −0.0215 (timing-dependent) |
| T4 prefers ON | **0 / 4** | **0 / 4** |
| T5 prefers OFF | 0 / 4 | 0 / 4 |

The directional signal is consistent across splits and depends on geometry and timing, so it
is not noise. But it **inverts pathway polarity completely** and involves no release from
inhibition, so it is not a biologically credible T4 mechanism in its current form. Its most
plausible reading is that a depolarised operating point lets the output rectifier act on both
sides of rest, and that the time-averaged readout on mean-centred contrast rewards the
background luminance offset. That is a hypothesis for the next experiment, not a result of this
one.

## Oracle comparison

T4: 0.0258 vs 0.3427 (7.5 %). Per-neuron DSI correlation with the oracle: Pearson −0.02 … +0.43
(T4a 0.004, T4b 0.43, T4c 0.22, T4d 0.26). Temporal cross-correlation of preferred − null
traces: 0.22 … 0.91 at lags scattered from −7 to +53 frames. The oracle remains far ahead.

## Required answers

1. **Does adding tonic inhibitory baseline materially change T4 DS?** Not usefully: D2 reaches
   +0.016 vs +0.010 for zero baseline, and in the frozen configuration removing tonic inhibition
   *raises* T4 DSI (+0.026 → +0.188).
2. **Does preferred-direction motion reduce Mi9 conductance below baseline?** **Yes** — in
   95–98 % of driven pairs, ~1.45× more than in the null direction.
3. **Does it create a measurable input-resistance window?** Yes, R_in rises (up to +134 %
   calibration, +170 % held-out) — but outside excitation.
4. **Does the window overlap fast excitation?** **No** — median overlap 0.00 in all 162
   release-capable configurations; R_in falls 44–75 % during excitation.
5. **Is dynamic disinhibition causally necessary?** **No** — preventing release changes DSI by
   only −11 %.
6. **Does clamping inhibition destroy DS?** Yes (−90 %), but via evoked inhibition, not release.
7. **Does preventing conductance decreases destroy DS?** **No** (−11 %).
8. **Is Mi9 specifically necessary?** Partly (−33 %, 2/4 positive), not as a release source.
9. **Mi4's role?** Larger than Mi9's (−50 % when removed): ON-evoked inhibition near Mi1 that
   adds conductance during excitation, working against a release-based window.
10. **Is fast excitation necessary?** No — removing Mi1 *raises* DSI to +0.043.
11. **Does temporal flattening destroy DS?** Largely (−69 %).
12. **Does true optic-column geometry remain causal?** Yes — residual 0.19, z 13.3.
13. **Does the model outperform E20?** Marginally on T4 (+0.026 vs +0.013), not on all-8
    (+0.014 vs +0.025).
14. **Does it approach E19 `OPPONENT_SIGNED`?** No (7.5 % on T4).
15. **Are all four T4 subtypes positive?** No — 3/4 (T4a null).
16. **Any T4 subtype inverted?** None significantly; T4a is null.
17. **Is ON specificity preserved?** **No** — all T4 prefer OFF in the frozen model.
18. **What happens to T5?** Mean +0.002, T5a strongly inverted — no transfer.
19. **Does T5 need a distinct mechanism?** Yes; with no resolved inhibitory input it cannot use
    this one at all.
20. **Justified to move toward VP→DN?** **No.**

## Conclusion

Release from tonic inhibition is **insufficient** to explain T4 direction selectivity in this
reconstruction. The ingredients are present — Mi9 is tonically active, is silenced by ON motion,
is silenced more in the preferred direction, and does so at the right time — but the resulting
reduction in conductance is an order of magnitude smaller than the excitatory conductance it
would need to amplify, so no input-resistance window ever coincides with excitation. The E19
oracle remains a computational constraint whose biological implementation is unresolved.

## Limitations

Single passive compartment by design. Presynaptic rates are an abstract linear/logistic
transfer of E18's filtered contrast signal, not measured Mi9/Mi4/Mi1 physiology; in particular
the mean-centred contrast definition gives every cell an implicit background-dependent offset,
which interacts with tonic rates (the ON/OFF inversion). Only ~57 % of T4 neurons have a
column-resolved Mi9 input; conductance scale is normalised by a single anatomical constant.
Held-out ablations were run on a gate-rejected fallback configuration, so they describe that
configuration rather than a validated mechanism. flyvis remains unresolved (F3, unchanged).

## Recommendation

Do **not** proceed to VP→DN. Two follow-ups, both prospective with fresh held-out data:

* **Experiment 22 (recommended): operating point and polarity.** Test the exploratory lead
  properly — preregister tonic *excitation* as a candidate — but first replace the mean-centred
  contrast with a signal referenced to a declared adapting background, and require ON/OFF
  specificity as a gate *before* DSI. If the lead survives with correct polarity, it is a
  genuine mechanism; if polarity only comes right when DS disappears, it was an artifact.
* A **T5-specific** experiment remains necessary regardless, since T5 has no resolved inhibitory
  input and CT1 is unplaced.

## Reproduction

```sh
python run_tonic_disinhibition_experiment.py calibrate
python run_tonic_disinhibition_experiment.py heldout
python analyze_tonic_disinhibition_experiment.py
```
