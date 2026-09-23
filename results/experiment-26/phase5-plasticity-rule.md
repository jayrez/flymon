# E26 Phase 5 — Three-factor plasticity rule

* **Rule:** `Δδ_ij = sign · η · D_c(t) · e_i(t) · w0_ij`, with δ_ij clipped to
  `[−w0_ij, (k − 1) w0_ij]` and w_eff = w0 + δ.
* **Defaults (preregistered):**
  * sign −1 (depression of co-active KC → MBON synapses, the literature default);
  * k = 1 (w_eff ∈ [0, w0]);
  * η is relative (scaled by w0).
* **Compartment specificity:** PAM01 gates only the 212 KCg-d → MBON01 edges, and PPL101 gates
  only the 206 KCg-d → MBON11 edges. A `cross_compartment` switch exists for the control only.
* **Plastic weights are separate from FlyBrain W.** `PlasticBrain.step` replicates
  `FlyBrain.step` (0.1.0, batch 1) and adds `gain · Σ_e δ_e · spiked(pre_e)` to each MBON at the
  same stage as `W x`. The frozen GPU matrix is only read, and its SHA-256 is unchanged
  (tested).
* **Bit-identity:** with δ = 0 the spikes are bit-identical to FlyBrain's own step (tested on
  the GPU).
* **State:** `PlasticState` (edge hash, δ[418], metadata) with save/load. Load rejects an
  edge-hash mismatch. The schema is in `plastic-edge-state-schema.json`.
* **η grid:** {0.01, 0.03, 0.1, 0.3, 1.0}, preregistered. **Not calibrated** (NO-GO before
  calibration). `plasticity-config.json` records η = 0.

**Validated by unit tests:**

* no DAN → no update;
* no eligibility → no update;
* wrong compartment → no update;
* the cross-compartment diagnostic updates;
* bounds hold under extreme η for both depression and the k = 2 diagnostic;
* determinism;
* finite values;
* a neural reset leaves δ intact, and resetting δ restores baseline spikes exactly (GPU).
