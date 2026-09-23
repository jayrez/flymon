# E26 Phase 6 — Validation

## Precondition A — visual drive (P1): FAIL

Setup: 6 seeds, 600 steps, frozen T4 path.

* **Result:** every condition gives KCg-d **48.57 Hz per cell, 100 % of cells active**. The
  conditions tested were no-visual, gray, vertical / horizontal bars, two static bedroom
  frames, drifting bars and a real walking sequence.
* **Largest difference:** vertical vs no-visual is +15.7 spikes on 120,074 (+0.013 %), in
  1/6 seeds, d_z 0.39.
* **Moving stimuli:** slightly *below* baseline (−15 spikes).
* **Noise:** trial-to-trial CV is 0.08 %; seeds differ by ~100 spikes.
* **Stimulus note:** the stimulus labelled `pokemon_house1f` stayed on map 38 (the scripted
  walk did not reach the stairs). It is a second bedroom frame. This does not change the
  result.

## Diagnostic — saturation of the mushroom body in stock FlyBrain

Source: `diagnose_mb_saturation.py`, no plasticity code.

* **At the ceiling:** all 4,064 KCs, MBON01/11, PAM01, PPL101, APL and DPM spike on
  97.1–97.3 % of steps (48.6 Hz of a 50 Hz maximum). They reach ≥ 90 % active by steps 17–18
  (~0.35 s) after `reset`.
* **Rest of the brain:** 7,596 neurons (4.6 %) are above 90 % duty, mostly central-brain
  intrinsic (5,372) and central-brain sensory (1,748). The brain median is 0.5 Hz, and the
  controller DNs fire at 0.2 Hz.

This is a self-sustaining high-activity attractor of the stock LIF parameters (tonic 0.14,
gain 3.0, no refractoriness) in the recurrently excitatory KC network (KC → KC:
1.15 M synapses).

## Phase 6A — maximum-effect bound: FAIL (gate)

Setup: all 206 KCg-d driven at 0.3 V for 300 steps, 6 seeds.

| weights | MBON01 spikes | MBON11 spikes |
|---|---|---|
| baseline | ≈ 588 | ≈ 588 |
| all 418 edges zeroed | ≈ 586 | ≈ 585 |
| doubled (diagnostic) | ≈ 589 | ≈ 589 |

Zeroing lowers the target MBONs by **−0.34 %**, in the expected direction in 6/6 seeds, but
the gate requires ≤ −10 %. Doubling gives +0.3 %.

The controller DNs vary by noise (16 → 16, 7 → 11, …). The MBONs are at the same ceiling, so
their inputs cannot move them.

## Phase 6B — conditioning: NOT RUN

Per the preregistered workflow, conditioning stops at the maximum-effect gate. It would also
be uninformative: the DAN gate cannot open (Phase 3), and eligibility is saturated
(Phase 4).
