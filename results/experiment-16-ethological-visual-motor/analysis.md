# Experiment 16 — Ethological Visual-Motor Transfer

**Verdict: FAIL. Primary category: A** (known ON/OFF and direction-selective motion
structure is not reproducibly present at T4/T5). The retina (R1–R6) robustly encodes
motion/change, but the current FlyBrain optic-lobe dynamics do **not** transform it
into direction-selective T4/T5 responses, so the cascade cannot be tested further.
The preregistered Gate A failed on held-out seeds; per the preregistration the VP and
DN feature-transfer stages were **not** run as validated analyses. No gameplay, no
closed loop, no gain tuning, no controller change; Experiments 1–15 unchanged.

This localises the Experiment-15 collapse: it is best read as **wrong / insufficient
optic-lobe dynamics** (the motion computation itself does not occur), not as
biologically appropriate feature compression.

## Design recap

42 deterministic, luminance-controlled stimuli (static, ON/OFF moving bars in four
directions, looming/receding discs, optic-flow gratings, and frozen/shuffle/reverse
temporal controls) were driven through the frozen Experiment-5 biological retina at
gain 1×. Subtype- and side-resolved 200-step spike counts were recorded for R1–R6,
lamina, T4a–d, T5a–d, LPLC1/LPLC2/VS/LC4, and DNp01/03/04/11, DNg13, DNa02, DNg100,
MDN. Preferred directions were fixed on calibration seeds 1101–1120; all statistics
use held-out seeds 1121–1140 with matched-seed sign-flip permutation tests (10 000).

## The retina works; the optic lobe does not compute motion

Held-out mean per-neuron spike counts (200 steps):

| Population | gray | dark | on_left | off_left | loom_dark | flow_expand |
|---|---:|---:|---:|---:|---:|---:|
| **R1–R6** | 23.6 | 12.9 | 30.2 | 67.7 | 55.0 | 68.9 |
| T5a_L | 1.96 | 2.02 | 1.99 | 1.92 | 1.98 | 1.93 |
| DNp01_R | 0.15 | 0.15 | 0.10 | 0.20 | 0.15 | 0.10 |

**R1–R6 motion-vs-frozen (change) responses are large and highly reproducible**
(matched-seed, held-out): flow_expand **+37.7** spikes (p = 0.0001, sign-consistency
1.00), off_left **+13.2**, on_left **+10.6**, off_right **+7.8**, loom_dark **−5.9**.
The retinal encoder clearly injects motion/change information.

**T4/T5 do not turn that into motion structure.** Across all T4/T5 subtypes and sides:
- Direction selectivity: **max |DSI| = 0.012** (≈ 0); **no** subtype shows a
  significant preferred-vs-null direction effect on held-out seeds.
- Motion-vs-frozen change response: statistically detectable but **negligible** —
  max effect 6.3 % of the frozen baseline (e.g. flow_expand|T5b_R −0.135 spike on a
  ~2.1-spike baseline), and it is the same sign for opposing subtypes (a global change
  blip, not motion tuning).

## Primary endpoint (preregistered)

Held-out T5 horizontal OFF-motion direction selectivity (T5a/T5b, both sides;
preferred direction fixed on calibration seeds):
**mean (preferred − null) = 0.0050 spike, permutation p = 0.795, sign-consistency
0.55, bootstrap 95 % CI [−0.030, 0.042]** — indistinguishable from zero. **The
primary endpoint fails.**

## Gate A decision

| Check (held-out) | Result |
|---|---|
| Retina R1–R6 motion-vs-frozen significant | **yes** (large) |
| T4/T5 direction-selective (any subtype, p<0.05, sign≥0.7) | **no** (max |DSI| 0.012) |
| T4/T5 change response meaningful (≥25 % of baseline) | **no** (max 6.3 %) |
| **Gate A pass** | **NO → stop at Gate A, Category A** |

Per the preregistration, the main cascade analysis stops here. VP (LPLC2/VS) and DN
transfer are **not** claimed. The connectome-defined pathway may be intact, but the
functional signal that would travel it (T4/T5 direction selectivity) is not produced
by the current dynamics.

## Exploratory observations (NOT validated; upstream motion detectors are silent)

Because Gate A failed, these are reported only to guide Experiment 17 and are treated
as possible luminance-change confounds, not motion computations:
- **DNp01** (looming/escape): held-out loom_dark vs recede_dark = −0.05 spike
  (recede marginally higher; not a loom preference). The apparent single-seed loom
  elevation seen during calibration did not survive 20 held-out seeds. No looming
  signal is validated.
- **DNg13 / DNa02** horizontal-flow (right−left) asymmetry: negligible.
- **DNg100 / MDN**: no consistent global-motion modulation.
- Dynamic-vs-frozen and ordered-vs-shuffle at DNp01: negligible.

## flyvis functional benchmark

**Not run.** `flyvis` is not installed in the environment, and installing it (Torch
models + pretrained-weight downloads) was deliberately not forced per the experiment
brief. A functional positive control would strengthen the interpretation by confirming
the stimuli *can* evoke T4/T5 direction selectivity in a model built to have it; this
is recommended for Experiment 17. Status recorded in `flyvis-benchmark.json`.

## Natural Pokémon follow-up

**Not run.** It was gated on a validated DN-level synthetic feature, which did not
occur (Gate A failed).

## Required final answers

1. FlyBrain reproduces T4/T5 ON/OFF motion structure? **NO** (T4/T5 flat; change
   response ≤ 6 % of baseline).
2. Direction-selective T4/T5 activity? **NO** (max |DSI| 0.012; none significant).
3. Feature survives into visual-projection neurons? **NO / not tested** (stopped at
   Gate A; there is no direction-selective feature to transfer).
4. Survives into identified descending neurons? **NO** (no validated feature; DN
   contrasts negligible on held-out seeds).
5. Best interpretation of the Experiment-15 collapse: **(a) wrong neural dynamics** —
   the optic-lobe motion computation is not happening. (c) "both" cannot be fully
   excluded, but there is no evidence of biologically appropriate compression because
   the compressed-toward feature (motion selectivity) is itself absent.
6. DNg13 biologically relevant visual-motor signal? **NO** (here; flow asymmetry
   negligible).
7. DNa02 visual-motor signal? **NO** (here).
8. Looming/escape DN cluster responds appropriately? **NO** (DNp01 flat over held-out
   seeds).
9. Does temporal order matter? **PARTIALLY** — strongly at the retina (motion vs
   frozen), not at T4/T5 or beyond.
10. Dynamic vs frozen differ? **PARTIALLY** — retina yes, deep stages no.
11. Next experiment: **repair dynamics / gain (Experiment 17), not closed loop.**
    The optic lobe must first compute motion before any biological visual-motor
    control is meaningful.

## Interpretation

- **What did Experiment 15 fail to preserve?** Five-way Pokémon scene-family identity
  into descending neurons.
- **What does Experiment 16 show the pathway actually preserves?** Robust
  motion/change *drive at the retina* (R1–R6), but **not** the motion *computations*
  (direction/ON-OFF selectivity) that T4/T5 are meant to perform.
- **Best-localised remaining bottleneck:** the optic-lobe dynamics themselves —
  R1–R6/lamina → T4/T5 does not yield direction-selective responses under the current
  FlyBrain leaky-integrate-and-fire parameters and gain. This is upstream of the
  T5 → VP → DN transfer that Experiment 15 examined.
- **Experiment 17 recommendation: (A) dynamics / gain repair** — sweep eye-drive gain
  and the FlyBrain dynamics (and add a flyvis functional positive control) to test
  whether T4/T5 direction selectivity can be recovered, before any biological
  closed-loop control is attempted. Not (B) closed loop; not another identity study.

## Limitations

A generic connectome-constrained leaky-integrate-and-fire model is not guaranteed to
reproduce T4/T5 motion selectivity, which in biology depends on dendritic nonlinearities
and precise temporal filtering not captured here. Gain was frozen at 1×; a different
gain might change T4/T5 excitability (that is exactly the Experiment-17 question).
The synthetic stimuli are not natural fly motion, and the display/eye-pose mapping is
inherited from Experiment 5. Absence of a flyvis positive control means we cannot yet
fully separate "stimuli inadequate" from "dynamics inadequate," though the strong,
structured R1–R6 responses argue the stimuli do drive the eye.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python run_ethological_experiment.py
OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python analyze_ethological_experiment.py
```
