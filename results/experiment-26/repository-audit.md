# Experiment 26 — Repository audit (Phase 0)

Audit only; no learning mechanism is implemented. Date 2026-09-23.

## Branches and provenance

| ref | SHA | content |
|---|---|---|
| `origin/main` | `b86ddb3` | PR #8 merge: E1–E24 (E24 `de239d5`) |
| E25 working branch `worktree-bridge-cse_014eULVFzQa6eL4wV7d1hV9L` (= `origin/…`, PR #9 open, **not merged**) | `40d644b` | E25 evolution on top of `b86ddb3` |
| `experiment-25-evolution` (new local alias) | `40d644b` | same commit; a name only, not pushed |
| `experiment-26-biological-learning` (new) | from `b86ddb3` | + `a2decc9` generic infrastructure import + the E26 audits |

**E25 integrity.** E25 is untouched. Its branch, commit, PR #9 and artifacts are unchanged,
and nothing on the E26 branch depends on E25's controller.

E25's local-only checkpoints (gitignored on the E25 branch) remain on disk as untracked files.
They are not part of E26.

## Current framework (on `main`)

| component | location | state |
|---|---|---|
| MaleCNS loader | `flybrain.FlyBrain(data=redfly-benchmark/data)` | 166,700 neurons, 25.58 M signed, input-normalised connections |
| weight storage | FlyBrain `_W` (cupy CSR, rows = post); CPU path uses CSC `indptr/indices/weights` | frozen; sensory rows zeroed when `sensory_input` is set |
| sign rule | `flybrain/build.py` | sign = −1 iff `consensus_nt` matches `gaba\|glutamate\|histamine`, else +1 |
| neuron metadata | `brain.npz` (`ids`, `cell_type`, `side`, `superclass`, `positions`, motor groups) | no transmitter field |
| transmitter annotations | `body-neurotransmitters-male-cns-v1.0.feather` | **was absent locally**; fetched from the official MaleCNS bucket (MD5 matches) into the gitignored raw-data directory (§"Data added") |
| sensory pipeline | `flymon/frozen_t4.py` (E23 T4, frozen, hash `87829806…`), `flymon/retina.py`, `flymon/column_motion.py` | stable |
| stimulation / injection | `FlyBrain.step(inject=[(idx, v)])`; `T4Injection` (gain 2.049, cap 0.8, 16 levels) | frozen |
| action decoder | E14 frozen controller: `flymon/exploration.py`, `flymon/interaction.py`, `flymon/steering.py`, `flymon/locomotion.py` | fixed; no DOWN |
| neural-state reset | `FlyBrain.reset(seed)` | zeroes voltages, spikes and refractory state; reseeds noise; **weights are untouched** |
| save states | `states/bedroom.state` only; `PokemonEmulator.load_state` | no battle / HP / EXP state exists |
| evaluator telemetry | `flymon/gameplay_eval.py` (map, x, y, battle flag, window) | evaluator only |
| episode runner | `run_generation_zero.py` (E24) | reproducible; bit-identical replays |
| existing plasticity | none (searched `flymon/`, `run_*`: no plasticity, STDP, eligibility or dopamine logic) | — |

## E25 components — classification

| component (E25 commit `40d644b`) | class | action for E26 |
|---|---|---|
| `flymon/fast_io.py`: `FastColumnSampler`, `device_pairs`, `VectorInjector` | **GENERIC / REUSABLE** | imported unchanged as clean commit `a2decc9`, with `test_fast_io.py` (sampler equivalence; opt-in GPU bit-identity; both pass) |
| worker-initialisation pattern in `run_evolution.py` (`_worker_init`, spawn pool, lazy brain) | GENERIC pattern | **not imported** (the file is mixed with evolution logic). Re-implement cleanly in E26 if needed |
| `run_evolution.run_task` episode loop (start-wait jitter, per-frame T4 streaming, evaluator after action) | mixed | ideas reusable; code not imported (it calls genomes/policies) |
| `flymon/evolution.py` (`Genome`, `Policy`, `FeatureBuilder`, `next_population`, `rank`, `FitnessTracker`) | **E25-SPECIFIC / DO NOT IMPORT** | excluded |
| `run_evolution.py` stages (screen, main, finalists, held-out), checkpoint lineage | **E25-SPECIFIC / DO NOT IMPORT** | excluded |
| `analyze_evolution.py`, `test_evolution.py` | **E25-SPECIFIC / DO NOT IMPORT** | excluded |
| evolved genomes, finalists, champion, fitness function | **E25-SPECIFIC / DO NOT IMPORT** | excluded |
| E25 throughput data | informational | reused only as reference numbers |

Generic pieces already on `main` (E24) and reusable by E26:

* `frozen_t4.py`
* `gameplay_eval.py`, which also provides the controller-input guard
* `run_generation_zero.py` (episode runner, determinism, replay checks)
* deterministic seed handling

## Data added (not committed)

* **File:** `redfly-benchmark/data/raw/body-neurotransmitters-male-cns-v1.0.feather`
* **Size:** 43,282,834 bytes
* **Source:** `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`
* **Verification:** MD5 `PYQrEv5cSe763lKNfdJKHw==` matches the bucket's `x-goog-hash`.
* **SHA-256:** `95c9289220663abe…`
* **Why it is safe to use:** this is the same table FlyBrain's own `build.py` uses for its sign
  rule. The data directory is outside version control.

## Implications for implementation (later phases)

* **Plastic weights need a separate term.** FlyBrain's `_W` is a single frozen CSR on the GPU.
  Plasticity on 418 edges (see the circuit audit) should be a separate, small plastic-delta
  term applied in a custom step, replicating `FlyBrain.step` the way `VectorInjector` does.
  This avoids mutating the global matrix and makes snapshot/restore trivial.
* **Dopamine is currently a fast excitatory synapse.** FlyBrain simulates it with sign +1 and
  has no neuromodulatory dynamics. DAN stimulation therefore also excites DAN targets
  directly. That direct effect is exactly what control group B (reinforcement, plasticity
  off) must measure.
* **`reset(seed)` does not touch weights.** This makes the required "trained weights +
  reset neural state" control straightforward.
* **Reinforcement events need a new save state.** No existing save state contains HP, EXP or
  battle events. A controlled state must be created, with scripted emulator inputs used only
  for state creation, never as a controller.
