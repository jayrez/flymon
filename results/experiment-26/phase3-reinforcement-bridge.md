# E26 Phase 3 — Reinforcement bridge

**Implementation.** `flymon/mb_plasticity.py` provides the bridge; `run_mb_plasticity.py`
runs the stages.

* **Request:** a request `(channel, magnitude)` becomes a voltage of `magnitude` added to every
  cell of the audited DAN population on each step of a 10-step (200 ms) window:
  * appetitive → PAM01 (44 cells);
  * aversive → PPL101 (2 cells).
* **Learning gate:** the gate is **observed simulated DAN spiking**, not the request:
  `D_c = max(0, f_c − θ_c)`, where f_c is the fraction of the channel's DANs spiking this
  step.
* **Fast synapses:** the DANs' ordinary fast excitatory synapses are untouched.

## Measured (`bridge-results.json`; seeds 2701–2706)

**Baseline** (600 unstimulated steps from reset):

* PAM01 fraction spiking per step, mean 0.972;
* PPL101 fraction spiking per step, mean 0.972;
* θ (p99, preregistered rule) = **1.0** for both channels.

**Stimulation** (all amplitudes 0.1–0.8): 100 % of DANs spike on every step of the window
(PAM01 440 = 44 × 10 spikes; PPL101 20 = 2 × 10). This is no different from baseline.

**The default amplitude by rule** is 0.1, but it is moot: the DANs are already at the 50 Hz
ceiling (one spike per 20 ms step) before any stimulation.

## Consequence

Stimulation cannot raise DAN activity, and the spike-derived gate `D = max(0, f − 1.0)` is
identically 0. With the preregistered spike-derived gate, no reinforcement can reach the
plasticity rule in stock FlyBrain.
