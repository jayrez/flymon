# Experiment 17 — Optic-Lobe Dynamics Repair (preregistration)

Frozen before held-out evaluation. Calibration seeds **1201–1220** may be inspected
while choosing parameters; held-out seeds **1221–1240** are evaluated exactly once
after the configuration is frozen in `selected-config.json`. Both ranges were
verified unused by Experiments 1–16. This is a mechanistic model-repair experiment:
**no gameplay, no closed loop, no controller change, no DOWN, no reward, no RL, no
behaviour cloning, no game RAM / coordinates / room-id / action labels.**
Experiments 1–16 and their result directories are not modified.

## Question

Experiment 16 found strong retinal motion/change drive but no T4/T5 direction
selectivity (max |DSI| 0.012; primary T5 OFF effect +0.005 spike, p = 0.795).
**Why does retinal motion input fail to become T4/T5 direction selectivity, and can a
biologically constrained dynamics change restore it on held-out seeds?**

## Hypotheses

* **H1 — amplitude.** Retinal drive reaches the right circuit but is too weak /
  poorly scaled. Tested by the Gate-1 gain sweep with FlyBrain dynamics unchanged.
* **H2 — dynamics.** Timing / neuron / synapse dynamics are wrong, so the circuit
  cannot perform delay-and-compare motion detection. Tested by Gate 3.
* **H3 — stimulus.** The Experiment-16 stimuli are themselves inadequate. Tested by
  the Gate-0 flyvis positive control.

## Stimuli

The **Experiment-16 catalogue, unchanged** (`flymon/ethology.py`): ON/OFF moving bars
(left/right/up/down), looming/receding discs, optic-flow gratings (left/right/expand/
contract), static controls, and frozen-first / frozen-mid / shuffle / reverse temporal
controls. Timing unchanged: 10 frames × 20 brain steps = 200 steps, 0 adaptation
steps. Luminance controls unchanged. No easier stimuli are substituted after seeing
failures; any exploratory variant would be reported separately.

## Gate 0 — flyvis functional positive control

Install `flyvis` in an **isolated** environment (never the Flymon venv), feed the same
stimuli as its input conventions permit, and ask: do ON/OFF pathways separate moving
from static input, do T4/T5-like units show direction selectivity, does reversal or
temporal shuffling change the response? Record version/checkpoint/source/device in
`flyvis-benchmark.json`. Neuron-index identity with MaleCNS is **not** required; only
response relationships are compared. Large model weights are never committed. If
installation is impractical, that is documented and Experiment 17 continues with
explicitly reduced confidence.

## Gate 1 — gain / dynamic-range hypothesis (H1)

FlyBrain dynamics unchanged; only the photoreceptor drive is scaled. Preregistered
grid: **0.5×, 1×, 2×, 4×, 8×** on calibration seeds. For each gain record R1–R6 rate,
lamina rate, T4 and T5 rates, per-subtype DSI, max |DSI|, and saturation/silence
diagnostics. If 8× saturates pathologically that is recorded, not replaced.

**A gain does not pass merely because firing increases.** The objective is
direction-selective structure: mean |DSI| over T5 horizontal subtypes, with
preferred-vs-null sign consistency as tie-break, penalising saturation and loss of
dynamic range. Gain-only repair passes only if a calibration-frozen gain yields, on
held-out seeds, a significant preferred-vs-null T5 effect with meaningful DSI,
consistent direction structure and non-pathological firing. If gain raises activity
but DSI stays near zero, we record **gain hypothesis: REJECTED** and proceed to H2.

## Gate 2 — connectivity and temporal audit

Label-free MaleCNS audit of the motion pathway: sign structure of R1–R6 → L1/L2/L3/L5
and the strongest presynaptic cell types onto each T4/T5 subtype, with signed weights.
Then per-step activity traces (not just 200-step totals) for R1–R6, lamina, the
Mi/Tm/CT1 medulla families and T4a–d / T5a–d under preferred and null motion, to test
whether the simulation produces any lead/lag between the excitatory-like and
inhibitory-like arms.

## Gate 3 — small, transparent dynamics grid (H2)

Interventions are opt-in via `flymon/optic_dynamics.py`; the reference FlyBrain path is
untouched and must remain bit-identical. The lamina tonic level is first chosen from
**{0.06, 0.09, 0.12}** on calibration seeds by largest non-pathological downstream
dynamic range. Candidate configurations (6):

| name | change |
|---|---|
| `A_reference` | unchanged LIF |
| `C_lamina_tonic` | maintained depolarisation for L1/L2/L3/L5 |
| `D_tonic_slow_inhibition` | C + τ×3 for Mi9/Mi4/CT1 |
| `E_tonic_delay1` | C + 1-step presynaptic delay on Mi9/Mi4/CT1 |
| `F_tonic_delay2` | C + 2-step presynaptic delay |
| `G_tonic_slow_delay2` | C + τ×3 and 2-step delay |

Gain enters only as the Gate-1-selected scalar. No optimiser, no random search, no
additional parameters.

## Calibration objective (calibration seeds only)

Primary: **mean |DSI| over the T5 horizontal subtypes** (T5a/T5b, both sides).
Tie-break: preferred-vs-null sign consistency. Excluded: configurations that are
pathological — saturated (T4/T5 mean > 100 spikes per 200 steps, i.e. >50 % duty) or
silent (T4/T5 mean < 0.2). Exactly one configuration is then frozen into
`selected-config.json`, together with the per-subtype preferred directions determined
on calibration seeds.

## Primary held-out endpoint

**Held-out (seeds 1221–1240) T5 OFF-motion preferred-vs-null direction selectivity**,
using the same logic as Experiment 16, with preferred directions frozen from
calibration. Reported: mean preferred−null effect, median, DSI, 95 % bootstrap CI,
matched-seed sign-flip permutation p (10 000), and sign consistency. The independent
unit is the CNS seed / complete stimulus trial, never a brain step.

## Secondary held-out endpoints

T4 ON-motion selectivity; DSI for every T4a–d and T5a–d subtype; dynamic-vs-frozen;
ordered-vs-shuffled; forward-vs-reversed; ON/OFF distinction; population firing regime.

## Success criteria (fixed now)

A repair is claimed only if held-out data shows **all** of: (1) reproducible T4/T5
preferred-vs-null separation; (2) DSI clearly larger than Experiment 16's 0.012 and
statistically reliable; (3) correct directional structure across relevant subtypes;
(4) dynamic differs from frozen; (5) shuffling or reversal degrades/alters the
directional signal; (6) non-pathological firing; (7) no held-out retuning.
**An increase in firing rate alone is not a repair.**

## Verdict categories

* **A** stimulus/benchmark failure — positive control does not show expected selectivity.
* **B** gain repair — bounded gain change alone restores held-out selectivity.
* **C** dynamics repair — gain fails but a small biologically constrained dynamics change works.
* **D** partial repair — substantial reproducible improvement, primary criteria not fully met.
* **E** no simple repair — neither gain nor the preregistered dynamics family works.

## Stop rules

If T4/T5 repair fails, the experiment **stops**: no downstream VP/DN failure claim, no
Pokémon control, no DNg13 tuning, no progression test, and no escalation of parameter
complexity inside Experiment 17. An optional small VP check (LPLC1/LPLC2/VS) runs only
if the T4/T5 repair passes strongly. No closed loop regardless of outcome.

## Reproduction

```sh
python run_optic_dynamics_experiment.py audit
python run_optic_dynamics_experiment.py gain
python run_optic_dynamics_experiment.py calibrate
python run_optic_dynamics_experiment.py traces
python run_optic_dynamics_experiment.py heldout
python analyze_optic_dynamics_experiment.py
```
