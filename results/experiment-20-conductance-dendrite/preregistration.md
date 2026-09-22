# Experiment 20 — Conductance / compartmental dendritic motion readout (preregistration)

Frozen before held-out evaluation. Only Experiment 17–19 artifacts and Experiment-20
**calibration** behaviour may be inspected while freezing this. **Prohibited:** Pokémon
gameplay, closed loop, VP→DN transfer, controller actions, Pokémon RAM / coordinates /
map or room IDs, reward, RL, behaviour cloning, per-neuron or per-subtype fitting,
manual preferred-direction assignment, held-out parameter tuning. Experiments 1–19 are
not modified.

## Base state (verified, not assumed)

`origin/main` = `507ad686` (merge of PR #5). Experiment 19 is on main as `859e1dc`,
whose tree `6b384bcf…` is **byte-identical** to the quoted `f133341`; the SHA differs
only because the commit was rebased onto the PR-#4 merge before review. Working tree
clean. Worktree forbids branch switching, so work continues on
`worktree-bridge-cse_014eULVFzQa6eL4wV7d1hV9L`.

Canonical E19 values re-read from artifacts: verdict FAIL / category F; oracle
`OPPONENT_SIGNED` held-out mean DSI **+0.3562**, **8/8** coherent, mean signResp
**0.8391**, subtype range **+0.3165 … +0.4148**; 2×2 decomposition
signed-opponent +0.3562 [8/8], signed-product +0.0560 [4/8], magnitude-opponent
+0.0359 [4/8], magnitude-product −0.1036 [1/8]; geometry null p = 0.00498.

## Question and hypotheses

**H1** — a minimal sign-preserving conductance/compartmental dendritic model transforms
the E19 pathway drives into coherent direction selectivity approaching the
`OPPONENT_SIGNED` oracle **without explicitly evaluating the opponent formula**.
**H0** — membrane and conductance dynamics alone are insufficient; an explicit or
structurally stronger antisymmetric interaction remains necessary.

## Sign mapping (the crux; fixed now)

Physical conductances are non-negative, so a synapse's sign comes from its reversal
potential. For presynaptic class `t` with MaleCNS weight `w` and polarity-filtered
signal `s_t`: activation `a_t = ReLU(s_t)` (rectified at the **presynaptic cell**, not
on the pooled arm drive); `w > 0` adds `|w|·a_t` to `g_E ≥ 0`; `w < 0` adds `|w|·a_t`
to `g_I ≥ 0`. This is *not* E19's magnitude drive, which pooled `|w|` over all partners
and erased excitatory/inhibitory identity. Verified consequence, with no subtype
coefficients: T4's fast arm is purely excitatory and its slow arm purely inhibitory
(Mi1 vs Mi9/Mi4), while both T5 arms are excitatory (Tm1/Tm2, Tm9).

## Membrane equation

```
C dV/dt = -g_L(V - E_L) - g_E(t)(V - E_E) - g_I(t)(V - E_I) - g_c(V - V_other)
```

Exponential Euler, 8 sub-steps per stimulus frame (exact for piecewise-constant
conductances; coupling semi-implicit). Output = `ReLU(V_out - E_L)`, or
`ReLU(V_out - E_L - threshold)` for C2.

**Compartment assignment is a model hypothesis, not an anatomical fact.** MaleCNS v1.0
contains no subcellular/dendritic compartment annotation for T4/T5, so "fast arm → A,
slow arm → B" is stipulated and is tested by the arm-swap ablation (A6).

## Candidate hierarchy (small and interpretable)

| id | model |
|---|---|
| **C0_SINGLE** | one compartment, signed conductances, no compartmentalisation |
| **C1_TWOCOMP** | fast arm → compartment A, slow arm → compartment B, coupled by `g_c` |
| **C2_TWOCOMP_NL** | C1 plus a thresholded output nonlinearity |

C3 is added only if C0–C2 reveal a specific mechanistic deficiency.

## Parameter grid (global only; no per-subtype or per-direction parameters)

`tau_fast = 4`, `tau_slow = 12` — **fixed to the E19 oracle's values** so the comparison
isolates the readout mechanism rather than re-searching temporal parameters.
`g_syn ∈ {0.5, 1, 2, 4}` · `E_inh ∈ {−0.2, 0.0}` (hyperpolarising vs shunting) ·
`g_couple ∈ {0.1, 0.5, 2.0}` (C1/C2) · `threshold ∈ {0.02, 0.08}` (C2 only).
Fixed: `C_m = 1`, `g_L = 1`, `E_L = 0`, `E_E = +1`, readout = compartment A.
Total 80 configurations (C0 8, C1 24, C2 48).

## Stimuli and split

Reuses Experiment-19's generators and split unchanged, so results are directly
comparable. **Calibration**: speeds {0.5, 1, 2}, width 18, contrast 1.0, four
directions, ON and OFF, plus gray/frozen/shuffle/reverse controls. **Held-out**:
speeds {0.25, 0.75, 1.5, 3, 4}, widths {12, 26}, contrasts {0.25, 0.5, 0.75}, start
offsets {±28}, grating phases {π/4, 3π/4} — disjoint from calibration.

## Metrics

`DSI = (R_pref − R_null)/(R_pref + R_null + 1e−9)` with the raw effect always reported.
Preferred direction is the **anatomy-predicted** direction per neuron (E18 geometry,
per-eye mirroring preserved). Statistical unit = individual eligible neuron. Report
mean/median DSI, per-subtype DSI, coherent-subtype count, responsive fraction,
signResp, bootstrap CIs over neurons (10,000), and Holm correction across the eight
subtypes.

## Model selection (calibration only)

Prefer the **simplest** model meeting the threshold, order C0 < C1 < C2; within a model
take the highest calibration mean DSI. Ties within 0.01 mean DSI are reported as a
sensitivity range rather than cherry-picked. If no model meets the threshold, that is
reported and the best is labelled explicitly as a best-of-null-set.

## Primary success criteria (fixed now)

On **held-out** data: (1) mean DSI **≥ +0.15**; (2) **≥ 7/8** subtypes positive
(8/8 preferred); (3) no systematic T4-vs-T5 sign inversion; (4) ON/OFF specificity
preserved (T4 favours ON motion, T5 favours OFF); (5) one global parameter set across
the entire held-out mixture; (6) no subtype-specific coefficients.

## Required ablations

A1 geometry shuffle (200 permutations, empirical null) · A2 temporal flattening
(`tau_slow = tau_fast`) · A3 coupling removal (`g_c = 0`) · A4 sign destruction (route
all input to `g_E`, the E19 magnitude condition) · A5 inhibitory-reversal neutralisation
(`E_I → E_L`, preserving timing but removing hyperpolarising drive) · A6 arm swap ·
A7 gray/frozen controls · plus trajectory reversal.

## Oracle comparison

Against E19 `OPPONENT_SIGNED`: mean and per-subtype DSI, coherence, signResp, response
amplitude, Pearson and Spearman correlation of per-neuron responses, and **temporal
cross-correlation** of subtype-aggregate preferred/null traces.

## Anti-cheating constraint

The conductance model must never evaluate `A_fast·B_slow − B_fast·A_slow` or any
algebraic equivalent. Direction selectivity must arise from conductance dynamics,
membrane voltage, coupling, timing and reversal potentials. A regression test asserts
the module contains no opponent expression and that the model's output is not a
rescaled copy of the oracle.

## Mechanistic categories

A sign-preserving single-compartment integration suffices · B compartmentalisation
required · C compartmentalisation plus output nonlinearity required · D conductance
dynamics improve selectivity but miss the threshold · E the oracle cannot be reproduced
by this minimal biophysical implementation.

## Stop condition

Stop after T4/T5. VP→DN is recommended only if every primary criterion is met and the
ablations show dependence on geometry, timing and sign-preserving inhibition/excitation.
