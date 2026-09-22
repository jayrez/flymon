# Experiment 19 — Minimal nonlinearity for T4/T5 motion readout

**Verdict: FAIL. Mechanistic category: F** — only an antisymmetric opponent
(difference-of-products) computation on **signed** arm drives reads out the
Experiment-18 geometry. Neither an extended temporal regime, nor time-resolved
analysis, nor coincidence, shunting or a generic dendritic subunit succeeds.

Experiment 19 was designed to try hard *not* to reach this conclusion: it tested the
two "boring" explanations first (H0 temporal range, H1 hidden transient) and gave three
distinct nonlinear mechanism classes a fair, preregistered shot. All five failed; the
one architecture that works is a control, not a selectable biological model.

## H0 — was Experiment 18's temporal grid simply too short? **No.**

`N1_LEXT` extends `tau_slow` from E18's edge value 8 out to 24:

| tau_slow (tau_fast = 4) | 5 | 8 | 12 | 16 | 24 |
|---|---:|---:|---:|---:|---:|
| calibration mean \|DSI\| | 0.062 | 0.070 | 0.078 | 0.085 | **0.093** |
| coherent subtypes | 2/8 | 2/8 | 2/8 | 2/8 | **2/8** |

|DSI| rises monotonically but only to 0.093 (threshold 0.15) and **coherence never
moves off 2/8**. Held-out `N1_LEXT` gives mean DSI +0.019, 2/8. **E18's negative
result was not a temporal-range artifact.** H0 rejected.

## H1 — did mean-rate averaging hide a transient? **No.**

Three predetermined temporal metrics on the frozen selected model:

| metric | held-out mean DSI | mean \|DSI\| | coherent |
|---|---:|---:|---:|
| **A** whole-sequence mean (E18's endpoint) | −0.1053 | 0.181 | 1/8 |
| **C** motion window (frames 10–59) | −0.1238 | 0.174 | 3/8 |
| **D** anatomy-local event window (±6 frames) | −0.0929 | 0.188 | 2/8 |

All three agree, and all three are *negative* — i.e. anti-anatomical. Restricting to
the stimulus-driven window, or to each neuron's own predicted receptive-field crossing
time, changes nothing. **H1 rejected.** (The window for D comes from the rendered
stimulus trajectory and the neuron's anatomical centre, never from its response; a
regression test asserts the window ignores response peaks outside it.)

## The selected biological mechanism fails, and fails informatively

Calibration selected `N2_COINCIDENCE | tau 4/12 | identity gate` with
`met_preregistered_threshold: false` (mean |DSI| 0.205 ≥ 0.15 but only **1/8**
coherent, far below the required 6/8). Held-out, metric A:

| subtype | DSI | 95 % CI | signResp | p (Holm) |
|---|---:|---:|---:|---:|
| T4a | **−0.2857** | [−0.311, −0.260] | 0.20 | 2.6e−81 |
| T4b | **−0.2432** | [−0.266, −0.221] | 0.26 | 2.9e−53 |
| T4c | **−0.2162** | [−0.239, −0.194] | 0.27 | 7.0e−51 |
| T4d | **−0.2702** | [−0.294, −0.246] | 0.21 | 6.7e−75 |
| T5a | −0.1286 | [−0.143, −0.115] | 0.23 | 9.8e−61 |
| T5b | +0.1870 | [+0.174, +0.201] | 0.88 | 3.1e−132 |
| T5c | +0.0559 | [+0.041, +0.071] | 0.61 | 8.5e−12 |
| T5d | +0.0585 | [+0.041, +0.076] | 0.60 | 4.8e−10 |

**All four T4 subtypes are strongly and uniformly inverted.** That is not noise — it
is a mechanistic signature. T4's slow arm is *inhibitory* (Mi9, Mi4 negative weights)
while T5's is *excitatory* (Tm9 positive). A coincidence rule built on rectified
**magnitudes** discards that sign, so for T4 it multiplies the fast drive by the
strength of *inhibition* and systematically prefers the wrong direction. Magnitude
rectification throws away exactly the information the computation needs.

## The decisive result: a 2 × 2 mechanism decomposition

Held-out mean DSI (coherent subtypes in brackets):

| | product only | antisymmetric opponent |
|---|---:|---:|
| **signed drives** | +0.0560 [4/8] | **+0.3562 [8/8]** |
| **magnitude drives** | −0.1036 [1/8] | +0.0359 [4/8] |

Reference points: E18's HR formulation +0.0816 [3/8]; extended linear +0.0189 [2/8].

**Both ingredients are necessary and neither is sufficient.** Multiplication alone on
signed drives gets 4/8 and a tenfold-smaller effect; opponency alone on magnitude
drives gets 4/8. Only their conjunction is coherent. So the answer to "does plain
multiplication suffice?" is **no** — the required operation is specifically an
*antisymmetric opponent* one, computed on inputs that retain their synaptic sign.

`OPPONENT_SIGNED` held-out, per subtype — all eight positive, all Holm p < 1e−70:

| subtype | T4a | T4b | T4c | T4d | T5a | T5b | T5c | T5d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DSI | +0.353 | +0.351 | +0.340 | +0.327 | +0.317 | +0.412 | +0.415 | +0.335 |
| signResp | 0.83 | 0.82 | 0.81 | 0.80 | 0.79 | 0.91 | 0.91 | 0.85 |

Mean DSI **+0.3562**, mean signResp **0.839**, minimum subtype DSI **+0.317**, all the
same sign. Note this *minimal* opponent product (1-frame delay) substantially
outperforms Experiment-18's HR formulation (+0.0816, 3/8) on the harder Experiment-19
held-out set: HR's additional long slow-arm filter costs it generalisation across the
wider speed range.

## Causal ablations (on the frozen selected model)

| manipulation | mean \|DSI\| | change |
|---|---:|---|
| selected model (observed) | 0.1807 | — |
| **temporal-flat** (tau_slow = tau_fast) | 0.0608 | **−66 %** |
| **interaction ablation** (remove the product term) | 0.0328 | **−82 %**, mean DSI → +0.001 |
| **geometry shuffle** (200 permutations) | null mean +0.0055 ± 0.0026 | observed −0.1053, **p = 0.005** |

The geometry null is tight (sd 0.0026) and the observed value sits ~42 SD away, so the
true optic-column geometry is unambiguously causal for what the model does compute —
it is the *readout rule*, not the geometry, that is wrong. Both the temporal split and
the interaction term are likewise causally necessary.

## Static and temporal controls

frozen **0.0** · gray **0.0** · shuffle 4.9e−05 vs dynamic 2.54e−04 (**−81 %**) ·
reverse 2.49e−04 (≈ dynamic, as expected — reversal is still motion). The model is
genuinely motion-driven; it simply assigns direction incorrectly.

## ON/OFF specificity — restored

| | T4a | T4b | T4c | T4d | T5a | T5b | T5c | T5d |
|---|---|---|---|---|---|---|---|---|
| favours | ON | ON | ON | ON | OFF | OFF | OFF | OFF |

**All four T4 subtypes prefer ON motion and all four T5 subtypes prefer OFF motion**, with
no subtype-specific coefficients anywhere. This is a genuine improvement over
Experiment 18, where ON/OFF specificity was absent or inverted, and it shows the
pathway polarity carried by the connectome plus literature-fixed cell-class signs is
correct even when the directional readout is not.

## Generalisation and coverage sensitivity

Held-out by axis (selected model): speed −0.1024 [1/8], width −0.0472 [2/8], contrast
−0.0504 [3/8], position −0.1160 [3/8] — consistently negative, never coherent.

Resolved-weight stratification (descriptive): ≥0.25 → −0.1053, ≥0.40 → −0.1055,
≥0.60 → −0.1194. **Restricting to neurons whose input field is better resolved does not
clean up the direction preference**, so Experiment-18's ~42 % coverage limitation is
not the explanation for the failure. Tm3/CT1 coordinates were not invented.

## Scale analysis

Median adjacent-column spacing is 3.368 px, so displacement per frame relative to the
median T5 fast–slow offset (0.64 columns) is 0.12 at speed 0.25 up to 1.86 at speed 4,
passing through ≈0.93 at speed 2. The held-out set therefore brackets the matched
regime on both sides, and the failure is not a speed-mismatch artifact. (For context,
the Experiment-16 stimuli ran at ~7.4× the offset per frame — badly mismatched.)

## Answers to the required questions

1. **Extended temporal constants rescue DS?** No — 0.093 |DSI|, 2/8, monotonic but flat in coherence.
2. **Strong transient hidden by averaging?** No — metrics C and D agree with A and are also negative.
3. **Best nonlinear mechanism?** Of the selectable biological families, `N2_COINCIDENCE` produced the largest *magnitude* (|DSI| 0.18) but with inverted sign; none was coherent.
4. **Does plain multiplication suffice?** **No** (signed product-only +0.056, 4/8).
5. **Is antisymmetric/opponent multiplication specifically required?** **Yes**, and on signed drives — the 2×2 shows both ingredients are necessary.
6. **Can divisive/shunting approximate it?** **No** — `N3_SHUNT` gave |DSI| 0.011–0.055, essentially nothing, across all α.
7. **Works across all eight subtypes?** Only `OPPONENT_SIGNED` (8/8). The selected model: 1/8.
8. **Generalises to unseen speeds/widths/contrasts/positions?** Selected model: no. `OPPONENT_SIGNED` is 8/8 coherent across the full held-out mix.
9. **Do geometry and temporal ablations collapse it?** Yes — geometry p = 0.005 (~42 SD), temporal-flat −66 %.
10. **Does interaction ablation collapse it?** Yes — −82 %, mean DSI → +0.001.
11. **ON/OFF specificity restored?** **Yes** — T4→ON, T5→OFF for all eight subtypes.
12. **Does better resolved-weight coverage improve coherence?** No.
13. **Literal multiplication supported?** No. What is supported is a broader
    *correlation-like antisymmetric opponent interaction on sign-preserving inputs*.
14. **Strong enough to justify VP→DN in Experiment 20?** **No.**

## Claim language

A minimal interaction of the **antisymmetric opponent (difference-of-products) class,
computed on sign-preserving arm drives**, is sufficient to read out the MaleCNS T4/T5
geometry under this reconstruction, achieving held-out mean DSI 0.356 with all eight
subtypes matching their anatomically predicted direction. Experiment 19 strengthens the
evidence that an opponent correlation-like computation — absent from the tested
point-neuron and static-nonlinearity models — is required. This is **not** a claim that
biology literally evaluates the textbook Hassenstein-Reichardt equation; indeed the
minimal 1-frame opponent product generalised *better* here than E18's HR formulation,
so the specific delay implementation is not what matters. What matters is the
combination of opponency and preserved synaptic sign.

Equally, Experiment 18's interpretation is refined rather than overturned: E18 concluded
"only an explicit correlator succeeds", and E19 confirms that while showing *why* the
simpler candidates fail — magnitude rectification destroys the sign asymmetry between
T4's inhibitory and T5's excitatory slow arms, which is precisely the information the
computation consumes.

## Limitations

The selectable model family remains small by design, and a genuinely compartmental or
conductance-based dendritic model was not implemented — `N3_SHUNT` is a divisive
approximation, not a membrane simulation. Only ~42 % of T4/T5 synaptic weight is
column-resolvable (Tm3, CT1 unannotated), though the stratification analysis argues this
is not the limiting factor. The front end is deterministic and evaluated on synthetic
bars and gratings. flyvis remains **unresolved / F3**: no pretrained checkpoint path
exists in this environment, so the cross-model comparison (category E) is still open.

## Recommendation

**Do not proceed to VP→DN transfer in Experiment 20.** The mechanism that works is a
control architecture, not a validated biological readout, so propagating it downstream
would inherit an unjustified assumption. The two highest-value next steps are
(a) resolve flyvis — a trained connectome-constrained model would show whether real
optic-lobe networks implement this opponent operation, settling category E; and
(b) implement a genuine two-compartment / conductance dendritic model, which is the one
mechanism class in the original brief that this experiment approximated rather than
implemented.

## Reproduction

```sh
python run_motion_nonlinearity_experiment.py calibrate
python run_motion_nonlinearity_experiment.py heldout
python analyze_motion_nonlinearity_experiment.py
```
