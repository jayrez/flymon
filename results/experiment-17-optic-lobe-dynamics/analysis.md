# Experiment 17 — Optic-Lobe Dynamics Repair

**Verdict: FAIL. Primary category: E** (no simple repair). The Experiment-16
stimuli are proven adequate, retinal gain is proven irrelevant, and a small
biologically constrained dynamics family **restores signal flow through the optic
lobe without producing any direction selectivity**. The held-out primary endpoint is
null: T5 OFF-motion preferred−null = **−0.0014 spike, DSI −0.00011, p = 0.950**,
bootstrap 95 % CI [−0.041, +0.040], sign consistency 0.50.

The bottleneck is now localised much more precisely than in Experiment 16: it is not
the stimuli, not the input amplitude, and not (only) the missing signal flow — it is
that **FlyBrain's uniform leaky-integrate-and-fire architecture has no mechanism that
can implement a motion correlator**. No gameplay, closed loop, controller change,
DOWN, reward or RL was involved; Experiments 1–16 are unmodified.

## Gate 0 — stimulus adequacy: SUPPORTED

flyvis 1.2.0 installed cleanly in an isolated target directory (torch 2.14.0+cpu,
torchvision 0.29.0+cpu, outside the repo and the Flymon venv, never committed), but
its **pretrained** ensembles are not obtainable here and the package exposes no
download helper; an untrained connectome-initialised network would not show direction
selectivity and so could not serve as a positive control. Dependency work was bounded
rather than escalated (`flyvis-benchmark.json`).

A substitute positive control answers the same question with no dependency: a textbook
**Hassenstein-Reichardt correlator** applied to the *same* Experiment-5 retina samples
of the *same* Experiment-16 stimuli, over 691 adjacent optic-column pairs (825 columns).

| HR stimulus | response |
|---|---:|
| off_left / off_right | **−217.0 / +217.0** |
| on_left / on_right | −217.0 / +217.0 |
| flow_left / flow_right | −933.3 / +932.7 |
| gray | **0.00** |
| frozen | **0.00** |
| shuffle | −51.4 (76 % suppressed) |
| reverse | +217.0 (sign flips) |

Horizontal **|DSI| = 1.000** with a clean sign flip, zero response to gray and frozen
input, 76 % suppression by temporal shuffling, and sign reversal under time reversal.
**The stimuli and the retinal sampling carry complete, cleanly extractable direction
information.** Hypothesis H3 is rejected.

## Gate 1 — gain hypothesis (H1): REJECTED

Calibration-seed sweep with FlyBrain dynamics unchanged:

| gain | R1–R6 | lamina | T4 | T5 | max \|T4 DSI\| | max \|T5 DSI\| | saturated |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5× | 26.4 | 0.94 | 2.30 | 2.22 | 0.0104 | 0.0117 | no |
| 1× | 45.7 | 0.90 | 2.33 | 2.20 | 0.0083 | 0.0043 | no |
| 2× | 77.7 | 0.89 | 2.33 | 2.19 | 0.0055 | 0.0146 | no |
| 4× | 119.9 | 0.88 | 2.36 | 2.19 | 0.0094 | 0.0120 | no |
| 8× | 130.9 | 0.88 | 2.36 | 2.19 | 0.0069 | 0.0101 | no |

This is exactly the diagnostic Experiment 17 was designed to expose: **R1–R6 firing
rises 5× while T4/T5 firing does not move at all and DSI stays at noise with no
trend.** Lamina firing even *decreases* slightly with gain. Amplitude is not the
issue. Selected gain: **1× (reference)**, since no gain improves selectivity.

## Gate 2 — connectivity and temporal audit

**Why the pathway was dead.** Every R1–R6 → lamina connection in MaleCNS is
inhibitory: 9,636 edges, mean weight −0.085, **negative fraction 1.000**
(L1 −388, L2 −279, L3 −152 summed). This is biologically correct — photoreceptors are
histaminergic and hyperpolarise the large monopolar cells — but in a purely spiking
model whose LMCs have no maintained depolarisation, light can only ever subtract
current, so the lamina is pinned at its floor (L1 1.83, L2 0.18 spikes/200 steps
against R1–R6's 49.7) and nothing downstream is modulated.

The motion-detector input structure itself is intact and correctly signed:

| target | excitatory-like | inhibitory-like |
|---|---|---|
| T4a | Mi1 (+471), Tm3 (+204) | Mi9 (−151), CT1 (−93), Mi4 (−90) |
| T5a | Tm2 (+260), Tm9 (+242), Tm1 (+190), Tm4 (+147) | CT1 (−178) |

**Temporal structure: absent.** Per-step traces give a cross-correlation lag between
the excitatory and inhibitory arms of **−1 step in every condition — including the
frozen control**. A lag that is identical for moving and for static input carries no
motion information. Under the reference config the downstream peaks fall at
essentially random steps (Mi1 at step 106, Tm2 at 60, Mi9 at 55), i.e. noise.

## Gate 3 — dynamics candidates

Lamina tonic level chosen on calibration seeds from {0.06, 0.09, 0.12} by largest
non-pathological downstream dynamic range → **0.12**. Six configurations; calibration
objective = mean |DSI| over T5 horizontal subtypes:

| config | objective \|DSI\| | max \|T5 DSI\| | sign | pathological |
|---|---:|---:|---:|---|
| A_reference | 0.0026 | 0.0050 | 0.56 | no |
| C_lamina_tonic | 0.0021 | 0.0042 | 0.60 | no |
| D_tonic_slow_inhibition (τ×3) | 0.0033 | 0.0063 | 0.66 | no |
| **E_tonic_delay1** (selected) | **0.0042** | 0.0060 | 0.67 | no |
| F_tonic_delay2 | 0.0019 | 0.0040 | 0.59 | no |
| G_tonic_slow_delay2 | 0.0029 | 0.0046 | 0.58 | no |

Every candidate sits at noise; `E_tonic_delay1` was selected only as the best of a
uniformly null set, which already predicted the held-out outcome.

## The repair did work — just not for selectivity

The lamina tonic term does exactly what the biology predicts. Held-out firing rates,
reference → repaired:

| population | reference | repaired | change |
|---|---:|---:|---:|
| R1–R6 | 49.7 | 52.1 | — |
| L1 | 1.83 | **23.11** | 12.6× |
| L2 | 0.18 | **7.32** | 41× |
| Tm2 | 0.61 | **6.94** | 11× |
| Mi9 | 1.85 | 7.10 | 3.8× |
| T5a_L | 1.96 | **9.22** | 4.7× |
| LPLC2 | 3.12 | 11.91 | 3.8× |
| VS | 3.76 | 10.30 | 2.7× |

Per-step traces confirm the cascade became stimulus-locked: R1–R6 peak at step 5 →
L1 at 5 → Tm2/Mi9 at 6 → T5a at 7, versus random peaks in the reference. Firing is
non-pathological (T4/T5 mean 5.26, max 10.6; not saturated, not silent).

**And direction selectivity is still exactly zero.**

## Held-out primary and secondary results (seeds 1221–1240)

| measure | reference | repaired (`E_tonic_delay1`) |
|---|---:|---:|
| **primary T5 OFF pref−null** | +0.0016 spike | **−0.0014 spike** |
| primary DSI | +0.00014 | **−0.00011** |
| permutation p | 0.930 | **0.950** |
| sign consistency | — | **0.50** |
| bootstrap 95 % CI | — | [−0.041, +0.040] |
| max \|T5 DSI\| | 0.0124 | **0.0019** |
| max \|T4 DSI\| | 0.0065 | 0.0156 |

Per-subtype T5 DSI under the reference is mirror-antisymmetric noise
(T5a_L +0.0032 / T5a_R −0.0026; T5b_L +0.0053 / T5b_R −0.0054) — left and right
cancel, the signature of chance. Under the repair the values shrink to ±0.002,
because the same noise is now divided by ~5× more spikes. **Experiment 16's
"max |DSI| 0.012" was itself a small-count artefact, and raising the counts makes it
smaller, not larger.**

Dynamic controls (held-out, matched-seed):

| contrast | retina R1–R6 | T5a_L (repaired) |
|---|---:|---:|
| dynamic vs frozen | **+13.2, p = 0.0001, sign 1.00** | +0.052, p = 0.048, sign 0.65 |
| ordered vs shuffled | **−1.08, p = 0.0001, sign 1.00** | +0.015, p = 0.615 |
| forward vs reversed | — | −0.005, p = 0.874 |

The retina is strongly and reproducibly motion- and order-sensitive; T5 is not.
ON/OFF specificity is also absent or inverted: mean T5 OFF-motion vs gray is
**−1.50** (T5 fires *less* to OFF motion than to a blank field), the wrong sign for an
OFF motion detector.

## Required final answers

1. Did flyvis validate the stimuli? **NOT RUN** (pretrained weights unobtainable) —
   the substitute HR control validated them: **YES**, |DSI| = 1.0.
2. Does increasing retinal gain increase T4/T5 firing? **NO** (R1–R6 yes, 5×; T4/T5 flat).
3. Does increasing gain restore direction selectivity? **NO**.
4. Did gain alone solve the Experiment-16 failure? **NO**.
5. Strongest T4/T5 inputs: T4 ← Mi1, Tm3 (excitatory), Mi9, Mi4, CT1 (inhibitory);
   T5 ← Tm2, Tm9, Tm1, Tm4 (excitatory), CT1 (inhibitory).
6. Meaningful temporal lead/lag between those arms? **NO** — lag is −1 step for moving
   *and* frozen stimuli alike.
7. Which dynamics modification restored selectivity? **None.** Lamina tonic drive
   restored signal *flow* (L1 12.6×, T5 4.7×) but not selectivity; τ scaling and
   1–2 step delays on Mi9/Mi4/CT1 added nothing.
8. Does repaired T4/T5 activity generalise to held-out seeds? **NO** (p = 0.950).
9. Does temporal shuffling reduce the repaired signal? **NO** at T5 (p = 0.615);
   **YES** at the retina (p = 0.0001).
10. Does reversing motion reverse the repaired signal? **NO** at T5 (p = 0.874).
11. Is the remaining problem: **C) model architecture.** Gain (A) is excluded, the
    stimulus mapping (D) is excluded, and temporal dynamics (B) were addressed without
    effect.
12. Should Experiment 18 proceed to VP → DN feature transfer? **NO.**

## Why no simple repair can work here

A Hassenstein-Reichardt / Barlow-Levick detector computes a **correlation** — a
multiplicative or strongly nonlinear interaction between a fast input from one column
and a delayed input from its neighbour. FlyBrain integrates a **linear weighted sum**
of presynaptic spikes and applies a threshold. Summing a delayed inhibitory current
with an undelayed excitatory one produces a small *linear* asymmetry, never a
correlator, so the preferred and null directions remain equivalent no matter how the
time constants and delays are set. This is why restoring dynamic range (which visibly
worked) changed firing rates 4–40× and changed DSI not at all. It also explains
Experiments 15 and 16 in one stroke: the optic lobe in this model transmits luminance
change but never computes motion, so nothing motion-specific can reach the
visual-projection or descending neurons.

## Limitations

The repair family was deliberately small (6 transparent configurations) as
preregistered; a larger search was not attempted and is not recommended inside this
experiment. Graded (non-spiking) transfer for photoreceptors and LMCs — biologically
the most faithful option — was listed as a candidate class but not implemented here,
because it changes the transmission model rather than its parameters; it belongs with
the architectural work recommended below. The absence of a *trained* flyvis control
means the cross-model comparison rests on the HR correlator instead, which is
transparent but simpler than a real optic lobe. The lamina tonic term is a minimal
abstraction of a maintained depolarisation, not a measured resting potential.

## Recommendation

Experiment 18 should **not** proceed to VP → DN transfer. The next step is an
**architecturally different optic-lobe model**: graded (non-spiking) photoreceptor and
LMC transfer plus an explicit nonlinear (multiplicative or divisive) dendritic
interaction in T4/T5, or a flyvis-derived optic-lobe front end whose trained T4/T5
units are validated against the HR control before being coupled to MaleCNS. Until the
optic lobe computes motion, downstream biological transfer cannot be tested and
biological closed-loop control is not meaningful.

## Reproduction

```sh
python run_optic_dynamics_experiment.py audit
python run_optic_dynamics_experiment.py gain
python run_positive_control.py
python run_optic_dynamics_experiment.py calibrate
python run_optic_dynamics_experiment.py traces
python run_optic_dynamics_experiment.py heldout
python analyze_optic_dynamics_experiment.py
```
