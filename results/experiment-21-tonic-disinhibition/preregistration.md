# Experiment 21 — Tonic shunting inhibition and release from inhibition in T4 (preregistration)

Written before any Experiment-21 calibration or held-out evaluation. **Prohibited:** gameplay,
closed loop, VP→DN, controller actions, Pokémon RAM / coordinates / map IDs, reward, RL,
behaviour cloning, per-neuron / per-subtype / per-direction / per-stimulus parameters,
held-out tuning, invented Tm3/CT1 geometry. Experiments 1–20 and their artifacts are not
modified; `flymon/conductance_dendrite.py` (E20) is imported unchanged, never edited.

## Base (verified)

`origin/main` = `88522ee` (PR #6 merge). E20 `e60eaf3` and E19 `859e1dc` are in its ancestry.
Branch `worktree-bridge-cse_014eULVFzQa6eL4wV7d1hV9L` was fast-forwarded to `88522ee` (no
rewrite); worktree forbids branch switching. Working tree clean.

Re-read from artifacts: E20 FAIL / E; held-out C0 +0.0067, C1 +0.0057, C2 +0.0246
(T4 mean +0.0127, T5 +0.0364, 5/8 positive); E19 `OPPONENT_SIGNED` +0.3562, 8/8;
E20 arm swap −2.0 %, inhibitory neutralisation −6.4 %, sign destruction −22.3 %, temporal
flat −61.6 %, coupling removal −90.2 %, geometry residual 77.1 %.

## Motivation (verified in code and by measurement)

`conductance_dendrite.py:95` sets presynaptic activation `a = max(s, 0)` on a temporally
mean-centred contrast signal, so inhibitory conductance starts at zero and **cannot be
released**. Measured: at peak, E20's T4a inhibitory conductance reached only 0.087 × g_L
(excitation 0.33 × g_L) — a near-linear regime in which shunting is negligible.

Label-free anatomy (descriptive): along each T4 neuron's fast–slow axis Mi9 lies well to the
slow side (median −0.43 … −0.70 columns from the Mi1 centroid) while Mi4 lies near Mi1
(−0.17 … +0.09), on opposite sides in ~72 % of neurons. Mi9 and Mi4 are therefore kept as
separate synapse classes (their projections are already separate).

## Hypotheses

**H1** — a passive single-compartment conductance model with non-zero tonic inhibitory
conductance, stimulus-driven modulation around that baseline, temporally distinct pathways
and preserved T4 chemistry produces coherent T4 direction selectivity without computing the
E19 opponent expression. **H0** — even with release from tonic inhibition represented,
passive conductance dynamics fail to reproduce it.

## Primary scope: T4

T4 (Mi1 excitatory fast; Mi9 and Mi4 inhibitory slow) is the primary mechanistic endpoint.
T5 (Tm1/Tm2/Tm9, all excitatory; CT1 unplaced) is evaluated **descriptively** with the frozen
T4-derived model and no rescue parameters. T4 verdict, T5 verdict and overall verdict are
reported separately.

## Model

Front end unchanged (E18 geometry, E19 stimuli and split, cell-class polarity and temporal
filters with `tau_fast = 4`, `tau_slow = 12` inherited from E19/E20).

Presynaptic rate for partner `j` of class `c` in column `col(j)`:

```
linear  (D1, D2): r_j(t) = max(0, r0_c + beta * s_c(col_j, t) / sigma)
bounded (D3):     r_j(t) = 2 r0_c * logistic(2 beta s_c / (sigma r0_c))   # inhibitory classes
```

`sigma` = pooled SD of the Mi1/Mi4/Mi9 filtered signals over **calibration stimuli only**,
computed once and frozen. The bounded transfer has baseline `r0_c` at `s = 0`, range
`[0, 2 r0_c]`, and the same small-signal slope as the linear transfer.

Conductance of neuron `i`, class `c`, routed by MaleCNS weight sign (never negative):

```
g_c,i(t) = G * sum_j |w_ij| r_j(t) / M        (w > 0 -> excitatory, w < 0 -> inhibitory)
```

`M` = median over T4 neurons of `sum_j |w_ij|` across Mi1+Mi9+Mi4 (an anatomy-only constant),
so `G` is expressed in units of `g_L`. A class with `r0_c > 0` has tonic conductance; an OFF
cell such as Mi9 falls **below** baseline during ON motion (release), while conductance can
never go below zero.

Membrane (single passive compartment, exact exponential update per frame):

```
C dV/dt = -g_L (V - E_L) - g_E (V - E_E) - sum_c g_c (V - E_I)
```

`C = 1`, `g_L = 1`, `E_L = 0`, `E_E = +1`. V starts at the tonic resting potential `V_rest`;
output = `max(V - V_rest, 0)` (zero on static input by construction).
`R_in,proxy(t) = 1 / (g_L + g_E(t) + g_I(t))`.

## Hierarchy

| id | content |
|---|---|
| **D0** | E20 reproduction, E20 module unchanged (C0 and the E20-selected C2) |
| **D1** | uniform tonic operating point: the same `r0` for Mi1, Mi9, Mi4 |
| **D2** | tonic **inhibition** only: `r0_inh` for Mi9 and Mi4, excitation zero-baseline |
| **D3** | D2 with bounded (logistic) transfer for the inhibitory classes |

No multi-compartment model (E20 showed symmetric compartments do not help); arm swap has no
meaning in a single compartment and is not used as a gate.

## Grid (global, class-shared only)

`G ∈ {1, 4, 16}` · `beta ∈ {0.5, 1, 2}` · `E_I ∈ {0 (pure shunt), −0.2}`.
D1: `r0 ∈ {0, 0.5, 1, 2}` (r0 = 0 is the E20-like zero baseline, recorded but gate-rejected).
D2, D3: `r0_inh ∈ {0.5, 1, 2}`. Total 72 + 54 + 54 = 180 configurations.

## Staged funnel (calibration only)

**Stage 1 — mechanism screen**, T4, calibration ON bars at speed 1.0 in each neuron's
anatomy-predicted preferred direction, frames 10–59:

* **Gate 1 — inhibitory release.** A neuron passes if its Mi9 conductance falls to
  ≤ 0.90 × its baseline for ≥ 3 contiguous frames. Config passes if ≥ 50 % of T4 neurons with
  Mi9 input pass. (Zero-baseline configs fail by construction.)
* **Gate 2 — input-resistance window overlapping excitation.** Excitation window = frames
  where Mi1 conductance above baseline is ≥ 50 % of its peak. Overlap = fraction of window
  frames with `R_in > R_in,rest`. Config passes if median overlap ≥ 0.50 **and** median peak
  `R_in/R_in,rest − 1` ≥ 0.05.
* **Gate 3 — dynamic inhibition is causal.** On all calibration ON bars, T4 mean DSI must be
  > 0 and must fall by ≥ 40 % when every inhibitory conductance is clamped to its baseline.

Rejection reasons are recorded for every configuration.

**Stage 2 — ranking** of gate survivors by calibration T4 mean DSI.
**Stage 3 — freeze.** Simplest family (D1 < D2 < D3) with calibration T4 mean DSI ≥ 0.15 and
4/4 T4 positive; within family, highest T4 mean DSI. Otherwise best survivor, labelled
best-of-null-set. If **no** configuration survives the gates, the best-T4-DSI configuration
is frozen for descriptive held-out only and the mechanistic category is E.
**Stage 4 — held-out once. Stage 5 — ablations and permutations** on the frozen model.

## Primary T4 endpoint (held-out)

All of: mean T4 DSI ≥ **+0.15**; **4/4** T4 subtypes positive; no T4 subtype significantly
inverted (Holm p < 0.05 with negative DSI); all T4 subtypes prefer ON over OFF motion; the
frozen config passed Gates 1–3; geometry shuffle leaves ≤ 50 % of the observed T4 effect.

Secondary: all-eight mean DSI, positive and coherent counts, signResp, T5 ON→OFF preference.

## Ablations (frozen model)

A1 inhibitory baseline → 0 · A2 no release (inhibitory rate floored at `r0`) · A3 inhibition
clamped at baseline · A4 inhibitory reversal → `E_L` (a no-op, reported as such, if the frozen
`E_I` is already 0) · A5 temporal flat · A6 geometry shuffle, 200 permutations, speed-axis
subset · A7 remove Mi1 · A8 remove Mi9 · A9 remove Mi4 · A10 frozen / gray. Also a baseline
sweep over `r0_inh` drawn from the calibration table.

## Statistics

Deterministic front end: statistical unit = individual eligible neuron (project convention);
seed replication is not fabricated. Mean/median DSI, bootstrap CIs over neurons (10,000),
sign test with Holm correction across subtypes, signResp. Geometry null: observed, null mean,
null SD, empirical p, residual fraction.

## Oracle

E19 `OPPONENT_SIGNED` is a benchmark only — mean and subtype DSI, signResp, per-neuron DSI
correlation, temporal cross-correlation of preferred−null traces. Never a training target.

## Anti-cheating

The model never computes `A_fast·B_slow − B_fast·A_slow` or an equivalent, and never imports
the E19 mechanism code. Regression tests enforce this.

## Categories

**A** release from tonic inhibition alone (D1/D2) meets the T4 endpoint · **B** needs the
bounded transfer (D3) · **C** gates pass and held-out T4 DSI ≥ 0.075 but the endpoint is
missed · **D** gates pass but T4 DSI < 0.075 · **E** no configuration passes the mechanistic
gates.

## Progression

VP→DN only if T4 meets the endpoint with causal release, timing and geometry, **and** T5 is
either solved or explicitly isolated with a plan. T4-only success ⇒ recommend a T5-specific
Experiment 22.

---

## AMENDMENT 1 — gate denominators (before any calibration run; no held-out data viewed)

A mechanics smoke test (one configuration, calibration stimuli only, no DSI computed) showed
that a single calibration bar drives only 23–52 % of T4a neurons — the unchanged E19 parametric
bars sweep part of the screen — and that only 57 % of T4a neurons have a column-resolved Mi9
input. The original Gate 1/Gate 2 denominators ("T4 neurons", speed 1.0 only) would therefore
measure stimulus coverage and annotation coverage rather than the mechanism.

Changed, with thresholds unchanged:

* **Gate stimuli**: all calibration ON bars (speeds 0.5, 1, 2) in each neuron's
  anatomy-predicted preferred direction, instead of speed 1.0 alone.
* **Denominator**: (neuron, stimulus) pairs in which the neuron is *driven* — its Mi1
  conductance above baseline peaks at ≥ 10 % of the median positive peak across neurons for
  that stimulus. Gate 1 is further restricted to driven pairs whose neuron has Mi9 input.
* Gate 1 passes if ≥ 50 % of those pairs release (≤ 0.90 × baseline for ≥ 3 contiguous
  frames); Gate 2 if the median overlap ≥ 0.50 and median peak `R_in/R_rest − 1` ≥ 0.05 over
  driven pairs. Gate 3 is unchanged.

The original text above is preserved. The ~43 % of T4 neurons without resolved Mi9 input are a
genuine limitation of the reconstruction and are reported, not hidden.

---

## AMENDMENT 2 — implementation correction to the fallback selection (calibration only; no held-out viewed)

The first calibration run rejected every configuration at the mechanistic gates (18 at Gate 1,
the zero-baseline configurations by construction; 162 at Gate 2). The preregistered fallback
is to freeze "the best-T4-DSI configuration" for descriptive held-out evaluation, but the
runner computed calibration DSI only for gate survivors, so its fallback picked the first grid
entry arbitrarily. The runner now computes calibration T4 DSI for **every** configuration
(descriptively; the gates alone still decide rejection) and applies the fallback as written.
A calibration-only diagnostic (`gate2-diagnostic.json`) is added to explain the Gate-2
outcome. Gates, thresholds, grid and endpoints are unchanged; no held-out data had been run.

---

## AMENDMENT 3 — descriptive family rows (after calibration; before held-out)

Final calibration: 18 configurations failed Gate 1 (zero baseline, by construction) and 162
failed Gate 2; none reached Gate 3. The frozen configuration is therefore the preregistered
fallback (best calibration T4 DSI), evaluated descriptively, and the category is fixed at
**E** by the preregistered rule regardless of its held-out DSI. Because the comparison table
was specified as "best gate survivor per family" and there are none, each family's
calibration-best configuration is evaluated on held-out **descriptively** so that D2 — the
hypothesis itself — is represented. This changes no selection, gate, threshold or endpoint.
No held-out data had been run when this was written.
