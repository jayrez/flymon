# Experiment 19 — Minimal nonlinearity for T4/T5 motion readout (preregistration)

Frozen before any held-out evaluation. Only Experiment-18 results (already published)
and Experiment-19 **calibration** behaviour may be inspected while freezing this.
**Prohibited throughout:** Pokémon gameplay, closed loop, VP→DN transfer, controller
actions or button mapping, Pokémon RAM / player position / map or room ID / dialogue
or battle state, reward, RL, behaviour cloning, classifier labels, per-neuron fitting,
per-subtype fitting, manual preferred-direction assignment, held-out parameter tuning.
Experiments 1–18 are not modified.

## Question

Experiment 18 showed MaleCNS contains strong subtype-specific T4/T5 spatial geometry
(verified: T4a 147.2°/T4b 326.5° antiparallel 179.3°; T5a 144.9°/T5b 323.1°
antiparallel 178.2°; axes 95.7°/98.4°; ~1,700 cells/subtype; T5 R≈0.88) that its
linear/rectifying model family could not read out (selected M1 τ2/8, held-out mean
|DSI| 0.0446, `met_preregistered_threshold: false`, T4a −0.032 and T5a −0.075
anatomically *inverted*), while an explicit antisymmetric correlator reached 0.3537
with 84.7 % anatomical agreement.

**What minimal biologically plausible temporal or nonlinear mechanism converts that
geometry into coherent held-out direction selectivity — and is an explicit
correlation-like computation actually necessary?**

## Hypotheses

* **H0** E18 failed because its temporal grid stopped at `tau_slow = 8` (the selected edge).
* **H1** Direction selectivity exists transiently and was erased by mean-rate averaging.
* **H2** Geometry + temporal filtering + a static rectifier is insufficient, but a
  simple coincidence/gating nonlinearity suffices.
* **H3** A divisive/shunting interaction suffices.
* **H4** Only an explicit antisymmetric correlation/opponent computation succeeds.
* **H5** None of these reliably succeeds under held-out and generalisation criteria.

## Frozen inputs from Experiment 18 (not recomputed from results)

Arms: T4 fast **Mi1**, slow **Mi9, Mi4**; T5 fast **Tm1, Tm2**, slow **Tm9**.
Coordinates: released `assignedOlHex1/2` only. Weights: actual MaleCNS synaptic
weights. Preferred direction: `centroid(fast) − centroid(slow)` mapped to screen
coordinates **per neuron using that neuron's own eye side** (the right eye is mirrored
into the display). No held-out response may alter anatomy; no manual subtype rotation.

**Measured sign structure** (reported, not assumed): T4's slow arm is inhibitory
(Mi9, Mi4 negative) while T5's slow arm is excitatory (Tm9 positive). Models are
therefore defined on rectified arm **magnitudes**, with the interaction expressed in
the equation rather than hidden in weight sign, so one equation applies to both
pathways.

## Scale analysis (measured, fixed before results)

Median adjacent-column spacing = **3.368 px**. Displacement per frame relative to the
median fast–slow offset (T5 ≈ 0.64 columns): speed 0.25→0.12, 0.5→0.23, 0.75→0.35,
1.0→0.46, 1.5→0.70, 2.0→0.93, 3.0→1.39, 4.0→1.86. A delay-and-compare detector is
expected to peak where `(tau_slow − tau_fast) × speed ≈ offset`, i.e. ~1 frame at
speed 2 and ~9 frames at speed 0.25. Tau is therefore tested jointly with speed, and a
single global tau must trade off across speeds.

## Model families

Common front end for every model (identical geometry, column luminance, polarity,
fast/slow signals, split): per-column contrast signal → cell-class polarity →
cell-class low-pass (`tau_fast` / `tau_slow`) → weighted sum over that neuron's
column-resolved partners. Let `F(t)` and `S(t)` be the fast- and slow-arm drives
(signed for N0/N1, magnitude-based `|w|` for N2–N4), and `Fr = ReLU(F)`, `Sr = ReLU(S)`.
All outputs are rectified (a rate cannot be negative). No per-neuron or per-subtype
parameters anywhere.

| id | rule | hypothesis |
|---|---|---|
| **N0_E18** | `Y = ReLU(F + S)`, τ=(2,8) — exact E18 M1 regression control | — |
| **N1_LEXT** | `Y = ReLU(F + S)` over the extended τ grid | H0 |
| **N2_COINCIDENCE** | `Y = Fr · g(Sr)`, `g = identity` or `g(x)=x/(1+x)` (saturating gate) | H2 |
| **N3_SHUNT** | `Y = Fr / (1 + α·Sr)` | H3 |
| **N4_SUBUNIT** | `Y = ReLU(Fr − β·Sr + γ·Fr·Sr)` | H2/H3 generic |
| **N5_HR_CONTROL** | E18 antisymmetric correlator; **positive control, never selectable** | H4 |
| **P_ONLY** | `Y = ReLU(F(t)·S_delayed(t))` — multiplication without opponency | decomposition |
| **OPPONENT_PRODUCT** | `Y = ReLU(F(t)S(t−Δ) − S(t)F(t−Δ))` | decomposition |

`P_ONLY` vs `OPPONENT_PRODUCT` isolates whether the necessary ingredient is
*multiplication* or *antisymmetric opponency*.

### Parameter grids (small, fixed)

* `N1_LEXT`: `tau_fast ∈ {1,2,4}` × `tau_slow ∈ {3,5,8,12,16,24}`, `tau_slow > tau_fast`
  (the union of E18's grid and its extension, so H0 is tested over 3–24).
* Nonlinear families use four representative pairs `(tau_fast, tau_slow) ∈
  {(1,3), (2,5), (2,8), (4,12)}`.
* `N3`: `α ∈ {0.5, 1, 2}`. `N4`: `β ∈ {0.5, 1}` × `γ ∈ {0.5, 1}`. `N2`: two gates.
* Δ for the decomposition controls = 1 frame.

## Stimuli and split (fresh for Experiment 19, disjoint from E18 held-out)

60 frames, luminance-controlled, constant bar area, generated by the unchanged
`flymon/column_motion.py` parametric generators.

* **Calibration**: speeds **{0.5, 1, 2}** px/frame, width 18, contrast 1.0, canonical
  start, all four directions, ON and OFF; plus gray / frozen / shuffle / reverse controls.
* **Held-out**: speeds **{0.25, 0.75, 1.5, 3, 4}**, widths **{12, 26}**, contrasts
  **{0.25, 0.5, 0.75}**, novel start offsets **{+28, −28}**, novel grating phases
  **{π/4, 3π/4}** with periods {24, 48}; plus the same control set.

No calibration value recurs in the held-out set.

## Temporal metrics (predetermined; no response-dependent window search)

Every model retains `response[neuron, time]`. With `D(t) = R_pref(t) − R_null(t)`:

* **A — whole-sequence mean** (exactly E18's endpoint; **primary**, for comparability).
* **B — `max D(t)` and `min D(t)`** (descriptive only; never a primary endpoint).
* **C — motion-window integral**: mean over frames `[10, 59]`, excluding a fixed
  10-frame startup transient declared here.
* **D — anatomy-local event window**: for each neuron, the bar-crossing frame is
  predicted from the *known stimulus trajectory* and that neuron's fast/slow centroid
  position in screen coordinates, and the response is averaged over a fixed ±6-frame
  window around it. This uses anatomy + rendered stimulus only, never responses. If
  it proves brittle it is reported as such and metric C stands.

Primary is **A**. If A fails while C/D succeed, that is hypothesis **H1** (category B).

## Metrics and statistics

`DSI = (R_pref − R_null) / (R_pref + R_null + 1e−9)`, with the raw `R_pref − R_null`
effect always reported alongside so small responses cannot inflate a ratio.
Statistical unit = **individual eligible neuron** (the front end is deterministic).
Report mean and median DSI, raw effect, responsive fraction, anatomical sign
consistency among responsive neurons, per-subtype values, bootstrap CIs over neurons
(10,000), and Holm correction across the eight subtype tests. Geometry-shuffle uses
**200 permutations** to build an empirical null. Frames are never treated as
independent replicates.

## Selection rule (calibration only)

Prefer the **simplest** family meeting the threshold, in order
`N0 < N1 < N2 < N3 < N4`. `N5_HR_CONTROL`, `P_ONLY` and `OPPONENT_PRODUCT` are controls
and can never be selected as the biological result. Within a family, pick the highest
calibration primary score. If no family meets the threshold, that is reported as such
and the best is labelled explicitly as a best-of-null-set.

## Primary endpoint and thresholds (fixed now)

A model passes only if, on **held-out** data with metric A:

1. mean |DSI| **≥ 0.15** (well above E18's 0.045);
2. responsive-neuron anatomical sign consistency **≥ 0.70** in **≥ 6 of 8** subtypes;
3. **no subtype strongly opposite** (no subtype with DSI ≤ −0.05 against anatomy);
4. generalisation across unseen speed, width, contrast and position;
5. required controls behave causally;
6. credible ON/OFF specificity (T4 favours ON motion, T5 favours OFF) — **required
   secondary**, tested and never forced by subtype-specific coefficients.

**Sign convention.** The anatomy-predicted direction (`fast − slow`) encodes a
coincidence/delay-and-compare convention. A model that is strongly *coherent* but
uniformly **negative** across all eight subtypes is still reading the geometry, under a
null-suppression convention; it is reported prominently as mechanism-informative and
classified **PARTIAL**, not PASS, because criterion 2 is defined against the
anatomy-predicted sign. This is stated before results to prevent a post-hoc flip.

## Required controls and ablations

gray · frozen · temporal shuffle · reverse · **geometry shuffle** (presynaptic columns
permuted within cell type, marginals preserved, 200 permutations) · **temporal-flat**
(`tau_slow = tau_fast`) · **interaction ablation** (mandatory: remove the
multiplicative/divisive/coincidence term, reverting to the simpler parent model).

## Sensitivity analyses (descriptive, not selection)

* **Resolved-weight stratification**: repeat the primary on neurons with resolved
  weight fraction ≥ 0.25, ≥ 0.40, ≥ 0.60. Threshold never chosen by outcome.
* **Eye mirroring**: verify left- and right-eye homologs of a subtype predict mirrored
  screen directions; regression-tested.
* **Tm3 / CT1** coordinates are **not** invented. No inference is attempted in this
  experiment; they remain unresolved and the ~42 % resolved-weight limitation stands.

## flyvis

Status carried forward as **unresolved / F3**. No bounded retry unless the environment
has materially changed; no ensemble training. flyvis is not an Experiment-19 endpoint.

## Verdict categories

A extended temporal regime alone solves it · B transient/time-resolved selectivity
without new nonlinearity · C coincidence/gating sufficient · D divisive/shunting
sufficient · E generic dendritic nonlinear subunit sufficient · F only antisymmetric
correlator / difference-of-products succeeds · G none reliably solves it · H
inconclusive / implementation or metadata limitation.

## Stop condition

Stop after T4/T5. Experiment 20 (VP→DN) is recommended **only** if a frozen, coherent,
generalising mechanism satisfies the full primary endpoint.

---

## AMENDMENT 1 (made after calibration, before any held-out evaluation)

The original preregistration specified `P_ONLY` and `OPPONENT_PRODUCT` as the
mechanism-decomposition controls but did not state which **arm-drive convention** they
use. As implemented they used rectified *magnitude* drives, while `N5_HR_CONTROL`
inherits Experiment-18's *signed* drives. Those two therefore differed in **two** ways
at once (opponency **and** drive convention), which would confound exactly the question
the decomposition exists to answer.

Two additional control families are added — `P_ONLY_SIGNED` and `OPPONENT_SIGNED` —
completing a clean 2 × 2:

| | product only | antisymmetric opponent |
|---|---|---|
| **signed drives** | `P_ONLY_SIGNED` | `OPPONENT_SIGNED` ≈ `N5_HR_CONTROL` |
| **magnitude drives** | `P_ONLY` | `OPPONENT_PRODUCT` |

These are **controls only**, never selectable as the biological result, and they do not
change the selection rule, the thresholds, the primary endpoint, or any calibration
outcome already recorded in `calibration-results.json`. The original text above is
unchanged. No held-out result had been inspected when this amendment was written.
