# Experiment 26 Phases 3–6 — Biological learning validation

**Verdict: NO-GO — the audited plastic edge set is too weak in current FlyBrain dynamics.**
The mushroom body sits in a saturated firing state, so no visual, dopaminergic or plastic
signal can move it.

**Phase 7 (Pokémon conditioning) is not justified.**

The plasticity *machinery* works and is fully tested: separate plastic weights, eligibility,
the three-factor rule, compartment gating, bounds and state save/load. Its failure point is
the network's operating regime, not the rule.

Base: `experiment-26-biological-learning` at `af29393` (PR #10), containing `a2decc9`. E25
(PR #9) is untouched. The preregistration (`phase3-6-preregistration.md`) was written before
any measurement.

## Findings

1. **Edge set.** 418 edges (KCg-d → MBON01 212, KCg-d → MBON11 206), SHA-256
   `65dbc6264c04a2fa…`. They are reproduced exactly from committed code (`audited_edge_set`)
   and committed data, matching `proposed-plastic-edges.json`: same edges, same weights, same
   hash.
2. **The mushroom body is saturated in stock FlyBrain.** All 4,064 KCs, MBON01/11, PAM01,
   PPL101, APL and DPM spike on ~97 % of 20 ms steps (48.6 Hz, ceiling 50 Hz). They are
   ≥ 90 % active within ~0.35 s of `reset`, with or without input. 7,596 neurons are in this
   state. The controller DNs sit near 0.2 Hz.
3. **P1 fails.** No visual condition changes KCg-d (largest +0.013 %, 1/6 seeds). Trial noise
   is a CV of 0.08 %.
4. **Reinforcement cannot be signalled by spikes.** DANs are already at the ceiling, so
   stimulation adds nothing, θ = 1.0 and the gate never opens.
5. **Maximum-effect gate fails.** Zeroing all 418 plastic edges changes the target MBONs by
   −0.34 % (6/6 seeds) against a ≥ 10 % requirement. Doubling gives +0.3 %.
6. **Conditioning, controls, timing and dose were not run**, as preregistered for a failed
   maximum-effect gate.

## Plastic circuit and rule

| item | value |
|---|---|
| compartments | appetitive: PAM01 gates KCg-d → MBON01; aversive: PPL101 gates KCg-d → MBON11 |
| bounds | w_eff ∈ [0, w0] (depression-only); k = 2 potentiation only as a diagnostic |
| eligibility | e_i ← min(λ e_i + s_i, 5), τ = 25 steps (0.5 s) default; normalisation: clip only |
| rule | Δδ = sign · η · D_c · e_i · w0, sign −1 |
| η | not calibrated (preregistered grid 0.01–1.0; recorded as 0) |
| gate | observed DAN spiking fraction above the baseline p99 |

## Controls table

| condition | weight change | MBON effect | interpretation |
|---|---|---|---|
| paired | not run | — | stopped at the maximum-effect gate |
| DAN only | not run | — | the rule gives no update without eligibility (unit test) |
| CS only | not run | — | the rule gives no update without DAN (unit test) |
| unpaired | not run | — | — |
| plasticity off | η = 0 → none (unit test) | — | — |
| wrong compartment | none (unit test) | — | compartment gating verified in code |
| shuffled eligibility | pattern permuted (unit test) | — | — |

## Persistence (GPU tests, rule level)

* A neural `reset` leaves δ intact, and the same δ reproduces the same spikes.
* δ = 0 restores baseline spikes exactly.
* Timing specificity and dose response: not run.

## Success criteria

| id | result |
|---|---|
| P1 | **FAIL** |
| P2 | bounds hold in unit tests; not assessable in circuit |
| P3–P7 | not run (NO-GO before conditioning) |
| maximum-effect gate | **FAIL** (−0.34 % vs −10 %) |

## Required answers

1. **Does KCg-d respond to the frozen Pokémon visual pathway?** No. It is saturated at
   48.6 Hz regardless of input.
2. **Which stimuli differentiate KCg-d most strongly?** None meaningfully. Vertical bars
   +0.013 %; drifting bars and walking −0.013 %.
3. **KCg-d trial-to-trial noise?** CV 0.08 % (SD ~100 spikes on 120,074 per 12 s trial).
4. **Can the 418 edges affect MBON01/11?** Only negligibly.
5. **Maximum effect of full depression?** −0.34 % MBON spikes; doubling +0.3 %.
6. **Does PAM01 stimulation produce reliable DAN activity?** It fires on every step, but no
   more than its saturated baseline.
7. **PPL101?** The same.
8. **Does paired CS + DAN change the intended weights?** Not run. In the circuit, the gate
   cannot open (θ = 1.0).
9. **CS only?** Not run. The rule gives no update without DAN (unit test).
10. **DAN only?** Not run. The rule gives no update without eligibility (unit test).
11. **Unpaired?** Not run.
12. **Compartment-specific?** Yes in the rule (unit tests); not tested in circuit.
13. **Does shuffled eligibility weaken the effect?** Not run.
14. **Does the learned response persist after a neural reset?** δ persists and reproduces
    exactly (GPU test). There is no learned response to measure.
15. **Does a learned-state reset restore baseline?** Yes: δ = 0 is bit-identical to baseline.
16. **Is the learned effect distinguishable from acute DAN excitation?** Not assessable: no
    learned effect, and DANs are saturated.
17. **Eligibility window?** Not calibrated. Default τ = 0.5 s; traces saturate in FlyBrain.
18. **Frozen learning rate?** None (η = 0); calibration was not reached.
19. **Numerically stable?** Yes: finite, bounded and deterministic in tests.
20. **Ready for Phase 7?** **No.**

## Phase 7 recommendation and next step

Pokémon conditioning is **not justified**. The blocker is FlyBrain's operating regime: a
saturated, recurrently excitatory mushroom body with no refractoriness. It is neither the
learning rule nor the audited circuit.

A separate, preregistered experiment should first bring the mushroom body into a responsive
regime without touching the audited plastic set. Options, preregistered and evaluated only on
responsiveness:

* FlyBrain's built-in `refractory` parameter;
* an APL-feedback or KC-gain operating point;
* per-population tonic calibration.

Then E26 Phases 3–6 should be re-run unchanged.

## Runtime

| stage | time |
|---|---|
| visual drive | 129 s |
| saturation diagnostic | ~1 min |
| bridge | 28 s |
| maximum effect | 33 s |
| tests | CPU < 1 s; GPU 8 s |

## Files

* `flymon/mb_plasticity.py`, `run_mb_plasticity.py`, `diagnose_mb_saturation.py`,
  `test_mb_plasticity.py`
* `results/experiment-26/`:
  * `phase3-6-preregistration.md`
  * `phase3-reinforcement-bridge.md`, `phase4-eligibility.md`, `phase5-plasticity-rule.md`,
    `phase6-validation.md`
  * `plasticity-config.json`, `plastic-edge-state-schema.json`
  * `visual-drive-results.json`, `bridge-results.json`, `max-effect-results.json`,
    `mb-saturation-diagnostic.json`
  * `conditioning-results.json`, `controls-results.json`, `timing-results.json`,
    `dose-response.json` — each marked NOT RUN, with the reason
* `captures/experiment-26/01–05*.svg`
