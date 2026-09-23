# E26 Phase 4 — Eligibility traces

* **Rule:** per KCg-d cell i (206 cells; each of the 418 edges uses its presynaptic cell's
  trace),
  `e_i(t+1) = min(λ e_i(t) + s_i(t), e_max)`, with λ = exp(−1/τ), s_i ∈ {0, 1} a spike in a
  20 ms step, and e_max = 5.
* **Normalisation:** none beyond the clip.
* **Postsynaptic factor:** none.
* **τ grid (preregistered):** {5, 25, 100} steps. Not calibrated, because the experiment
  stopped at the maximum-effect gate. The default τ = 25 steps (0.5 s) is recorded.

**Validated in unit tests (`test_mb_plasticity.py`):**

* exact exponential decay;
* clipping;
* activity without DAN changes no weight;
* traces clear on reset;
* shuffled-eligibility control permutes the pattern.

**In FlyBrain:** KCg-d spikes on ~97 % of steps regardless of input, so every trace sits at
its clip (5) within a few steps. Eligibility therefore carries no stimulus information in the
current dynamics.
