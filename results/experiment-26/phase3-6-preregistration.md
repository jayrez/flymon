# Experiment 26 Phases 3–6 — preregistration

Written before any visual-drive, maximum-effect or conditioning measurement. No Pokémon
reward event, gameplay fitness, evolution or E25 code is involved. Later changes are appended
as timestamped amendments.

## Base

* Branch `experiment-26-biological-learning` at `af29393` (PR #10, open); contains `a2decc9`.
* `origin/main` = `b86ddb3`. PR #9 (E25) is open and untouched.
* The audited edge set is reproduced exactly from committed code (`flymon/mb_plasticity.py:
  audited_edge_set`) and committed data (`weights.npz`, `brain.npz`):
  * 418 edges — KCg-d → MBON01 (212, appetitive candidate), KCg-d → MBON11 (206, aversive
    candidate);
  * SHA-256 `65dbc6264c04a2fa…`, identical to `proposed-plastic-edges.json`.

## Common protocol

* **Network:** FlyBrain MaleCNS v1.0 on the GPU, stepped by `PlasticBrain.step`. It
  replicates `FlyBrain.step`, adds vector injection and the plastic-delta current, and is
  bit-identical to `FlyBrain.step` with zero deltas (tested).
* **Seeds:** FlyBrain seeds **2701–2706** (6 seeds, unused before). Every comparison is
  paired by seed.
* **Unit:** a 20 ms brain step.
* **Visual path:** frame → E5 column luminance → frozen E23 T4 model (hash `87829806…`,
  12 frames per decision) → mean T4 response → frozen T4 injection (gain 2.049, cap 0.8,
  16 levels) → 10 brain steps per decision. This is the same loop as E24/E25.
* **Populations:** KCg-d (206), MBON01 (2), MBON11 (2), PAM01 (44), PPL101 (2).

## Precondition A — visual drive of KCg-d (P1)

**Stimuli.** Each trial is 60 decisions (600 steps, 12 s). T4 starts from its adapted state
and FlyBrain from `reset(seed)`.

| stimulus | description |
|---|---|
| `no_visual` | no injection |
| `gray` | uniform display 0.5 |
| `vertical` | static vertical square-wave bars, period 16 px, full contrast |
| `horizontal` | static horizontal bars, period 16 px |
| `pokemon_bedroom` | static bedroom frame |
| `pokemon_house1f` | static RedsHouse1F frame; the state is reached by scripted taps used only to create the stimulus |
| `drifting_vertical` | vertical bars drifting 1 px/frame |
| `pokemon_walk` | recorded sequence of real game frames while walking, replayed |

**Measures (per trial):**

* KCg-d total spikes, rate, fraction of cells active, mean and median per cell;
* L/R split;
* per-cell count vectors;
* also MBON01/11, PAM01 and PPL101 spikes, and the injected T4 cell count.

**P1 gate.** P1 passes if at least one visual condition has, versus **both** `no_visual` and
`gray`:

* a mean paired increase in KCg-d spikes > 0;
* the same sign in ≥ 5/6 seeds;
* Cohen's d_z ≥ 0.8;
* a relative increase ≥ 20 % over `no_visual`.

Trial-to-trial noise is reported as the across-seed SD and CV per condition.

**Visual CS eligibility.** Visual stimuli are used as CS+/CS− only if both of these hold:

* P1 passes;
* two passing conditions have per-cell KCg-d patterns with across-seed mean Pearson
  r < 0.8, and each activates ≥ 10 cells in ≥ 5/6 seeds.

Otherwise conditioning uses **circuit-level CS**: voltage injection into two fixed, disjoint
random subsets of KCg-d (CS+ 40 cells, CS− 40 cells, rng 2601). These results are labelled
*circuit-level, not visual*. With a failed P1, the verdict is at most PARTIAL.

## Phase 3 — reinforcement bridge

* **Request:** `(channel, magnitude)` with channel ∈ {appetitive → PAM01, aversive → PPL101}
  and magnitude ≥ 0. It is applied as voltage `magnitude` added to every cell of the channel's
  DAN population on each step of the stimulation window (10 steps = 200 ms).
* **Amplitude grid:** {0.1, 0.2, 0.3, 0.5, 0.8}.
* **Default amplitude:** the smallest grid value whose mean fraction of DANs spiking at least
  once in the window is ≥ 0.5, measured in stimulation-only trials across the 6 seeds. It is
  chosen once and applied to both channels.
* **Learning gate:** observed simulated DAN spikes, `D_c(t) = max(0, f_c(t) − θ_c)`. θ_c is
  the 99th percentile of per-step spiking fraction during 600 unstimulated baseline steps
  (6 seeds), fixed per channel by this rule.
* **Fast excitatory DAN synapses** are left intact.

## Phase 4 — eligibility

* **Rule:** `e_i(t+1) = min(λ e_i(t) + s_i(t), e_max)` per KCg-d cell. The edge eligibility is
  that of its presynaptic cell. No postsynaptic factor.
* **Constants:** λ = exp(−1/τ); e_max = 5.
* **τ grid:** {5, 25, 100} steps (0.1 / 0.5 / 2 s). One τ for both compartments.

## Phase 5 — plasticity rule

* **Rule:** `Δδ_ij = sign · η · D_c(t) · e_i(t) · w0_ij`, with δ clipped to [−w0, (k−1) w0].
* **Defaults:**
  * sign = −1 (depression);
  * k = 1 (w_eff ∈ [0, w0]);
  * only the edge's own compartment DAN gates it.
* **Potentiation** (sign +1, k = 2) is used only as a diagnostic in the maximum-effect
  bound.
* **η grid (relative):** {0.01, 0.03, 0.1, 0.3, 1.0}.
* **Calibration** (synthetic, appetitive channel, default τ = 25):
  * choose the smallest η whose paired training gives a mean |δ|/w0 on CS+ edges in
    [0.10, 0.80], with no saturation of more than 50 % of CS+ edges and no NaN/Inf;
  * then choose τ from the grid by the same criterion at that η, preferring 25 on ties;
  * freeze both for everything else.

## Phase 6A — maximum-effect bound (gate)

**Setup.** KCg-d is driven either by the best visual condition, if P1 passes, and always by
circuit-level drive: 0.3 V per step into all 206 KCg-d cells for 300 steps.

**Weights compared:**

* baseline;
* all 418 edges at 0 (δ = −w0);
* diagnostic k = 2 (δ = +w0).

**Measures:** MBON01 and MBON11 spike counts, plus the controller-DN spike counts, across the
6 seeds.

**Gate.** The zeroed-edges condition must reduce the target MBON spike count (sum of MBON01
and MBON11) by ≥ 10 % versus baseline, in ≥ 5/6 seeds. Otherwise the result is **NO-GO —
audited plastic edge set is too weak in current FlyBrain normalization**, and no edges are
added.

## Phase 6B — conditioning (each channel separately; appetitive first, then aversive)

Steps are 20 ms each.

* **Pre-test:** CS+ 50 → rest 50 → CS− 50 → rest 50. No DAN, learning off.
* **Training:** 10 × [CS+ 25 steps, DAN window 10 steps coincident with CS+ steps 15–24,
  rest 50, CS− 25, rest 50]. Learning on.
* **Post-test:** identical to the pre-test.
* **Target measure:** target-MBON spike count during CS+ minus during CS−. The learning effect
  is post − pre of that difference. Weight change is measured on CS+ edges versus CS− edges
  (edges whose presynaptic KCg-d cell belongs to each CS set, for circuit-level CS).

**Controls** (same seeds):

| id | control |
|---|---|
| A | paired (as above) |
| B | DAN only (no CS in training) |
| C | CS only (no DAN) |
| D | unpaired (DAN 100 steps after CS+ offset) |
| E | plasticity off (η = 0) |
| F | wrong compartment (appetitive DAN; aversive edges must not change; a cross-compartment diagnostic is also reported) |
| G | shuffled eligibility (KC traces permuted, rng 2602) |

**Timing:** DAN onset at 0 / +5 / +25 / +100 / +250 steps after CS+ offset.

**Dose:** DAN amplitude grid {0.1, 0.2, 0.3, 0.5, 0.8}.

**Reset:**

1. train, then post-test;
2. `brain.reset(seed + 1000)` and clear traces, then post-test again;
3. δ = 0, then post-test.

**Acute vs learned:** MBON spike counts during the DAN window in training, versus the
post-test without DAN.

## Success criteria

| id | criterion |
|---|---|
| P1 | visual-drive gate above |
| P2 | after paired training, every δ ∈ [−w0, 0]; no edge outside the trained compartment changes; frozen-W hash unchanged |
| P3 | paired mean \|δ\|/w0 on CS+ edges ≥ 3 × each of CS-only, DAN-only and unpaired, in ≥ 5/6 seeds per channel |
| P4 | appetitive training changes 0 aversive edges and vice versa (exactly) |
| P5 | after a neural reset, the learned effect keeps the same sign and ≥ 80 % of its magnitude (mean), in ≥ 5/6 seeds |
| P6 | after δ reset, \|mean post − pre\| < 50 % of the learned effect |
| P7 | the learned effect (discrimination change) has the preregistered sign (depression → CS+ target-MBON response decreases relative to CS−) in ≥ 5/6 seeds, with d_z ≥ 0.8, for both channels |

## Verdict

* **GO TO POKÉMON CONDITIONING:** P1–P7 all pass, no numerical instability, and the
  maximum-effect gate passes.
* **PARTIAL:** the mechanism passes synthetically (P2–P7 and the maximum-effect gate), but P1
  fails or the target effect is too small to matter.
* **NO-GO:** the maximum-effect gate fails, or pairing / compartment specificity fails, or
  the effect is only acute.

**No post-hoc circuit changes** within this experiment.
