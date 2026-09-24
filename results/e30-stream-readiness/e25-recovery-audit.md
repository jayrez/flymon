# E30 — E25 recovery audit

## Starting point

- **Base:** `origin/main` 3e8cb99, the PR #14 merge. It contains E29 d5389b4, E28 a576fc6, E27 b36e30d and
  E26 336f06b.
- **Historical source:** E25 commit `40d644b2aa5c715091ca35e3b2b7e8360ace9f4a` from PR #9. That PR was never
  merged; its parent is b86ddb3, the PR #8 / E24 merge.
- **Method:** E25 was **not** merged or cherry-picked. Every file E25 changed is classified below and recovered
  deliberately.

| Class | Meaning |
|---|---|
| A | Already present or equivalent on main |
| B | E25-specific and required; ported |
| C | E25-specific but obsolete for E30; ported unmodified for provenance only |
| D | Conflicts with later infrastructure; reconciled by hand |
| E | Historical artifact; kept as a reference, not runtime |

## Every file changed by 40d644b (42 files)

| File | Class | Action |
|---|---|---|
| `flymon/fast_io.py` | **A** | Byte-identical on main. It arrived via `a2decc9` ("infra: import generic gameplay-loop accelerations from experiment 25"). Not touched. |
| `flymon/evolution.py` | **B** | Restored verbatim: `ACTIONS`, T4/DN feature builders, `T4Pooling`, `Genome` (JSON + SHA-256), `Policy`, `FitnessTracker`, `next_population` / `rank`. Imported unchanged by the live runtime. |
| `run_evolution.py` — `run_task`, worker pool | **B** | Restored verbatim. It is the historical reference runner, used for the seed-2601 smoke check. The live loop in `flymon/live_runtime.py` re-implements `run_task` for the T4 architecture and is tested action-for-action against the E25 logs. |
| `run_evolution.py` — training stages (screen / main / finalists / held-out / throughput) | **C** | Kept unmodified inside the same file; not run in E30 (no retraining). |
| `test_evolution.py` | **B** | Restored verbatim; all 23 tests pass on current main. Its sampler and vector-injection tests overlap `test_fast_io.py` (class A). They are kept because they pass unchanged. |
| `analyze_evolution.py` | **E** | Restored as the historical analysis script that produced `champion-evaluation.json`. Not runtime. |
| `.gitignore` | **D** | The E25 section (only final checkpoints committed; smoke checkpoints and videos ignored) was re-applied as a patch on top of main's file, plus a new `/runtime/` rule for live stream state. |
| `PROJECT_STATUS.md` | **D** | Not overwritten. Main's version (E26–E29) is kept, and E29 / E30 are added by hand. The E25 row text is taken from the E25 report. |
| `results/experiment-25-evolution/*` (26 files: report, preregistration, champion weights / evaluation / held-out logs, finalists, held-out raw, generation logs, 4 final checkpoints, elite history, seeds, sensory hash, smoke, throughput) | **E** | Restored exactly as committed, as the historical record: 1.3 MB, final checkpoints only. The champion is extracted from `finalists.json` into `results/e30-stream-readiness/champion-genome.json` (runtime file, hash-verified). Local untracked intermediate checkpoints (`gen-000`…`gen-028`) remain ignored and are not committed. |
| `captures/experiment-25/*.svg` (10 files) | **E** | Restored as the historical figures. |

## E25 runtime dependencies already on main (class A)

Each of these files is byte-identical between 40d644b and 3e8cb99:

- `flymon/frozen_t4.py`: `FrozenT4Readout`, `T4Injection`; sensory SHA-256 87829806e398f66b…
- `flymon/gameplay_eval.py`: `read_telemetry`, `MilestoneTracker`, `LoopDetector`, `ControllerInputGuard`, `MAP_NAMES`
- `flymon/emulator.py`: PyBoy wrapper
- `flymon/column_motion.py` and `flymon/retina.py`
- `run_generation_zero.py`: `STATE`, `sensory_setup`, `brain_setup`, injection constants
- `run_fixed_background_experiment.py`: `contexts`, `permuted_projections`
- `run_population_event_experiment.py` and `run_progression_experiment.py` (only needed by the Gen 0 baseline)
- `states/bedroom.state`: SHA-256 305212ea…
- `results/experiment-24-generation-zero/c0-action-law.json`: matched-random law

`test_fast_io.py` is newer than E25; it was created by a2decc9.

## Conflicts with E26–E29

There are none in code. E26–E29 only added the biological-learning modules (`flymon/mb_plasticity.py`,
`run_mb_*`, `audit_*`, `run_broader_mb_circuit.py`) and their results. No E25 file was modified later, and
`flymon/fast_io.py` is shared and identical. All E26–E29 files, tests, reports and figures are left untouched.

## Champion recovery

The champion was recovered from the committed E25 artifact (preference 1). Its genome SHA-256 is identical in
`finalists.json` (rank 5), `champion-weights.json` and `champion-evaluation.json`. It has 253 parameters: W is
7 × 35, b has 7 entries, plus log T. No weights were created by hand. See `e25-champion-provenance.json`.

## New E30 code

| File | Role |
|---|---|
| `flymon/live_runtime.py` | frozen controller, random baselines, episode loop; controller side accepts framebuffers only |
| `flymon/stream_eval.py` | evaluator / overlay only: fitness, stream milestones, dialogue / stall classification |
| `flymon/stream_state.py` | SQLite stream metadata |
| `flymon/telemetry.py` | schema, hub, GET-only HTTP / SSE / MJPEG server |
| `run_stream.py` | supervisor, watchdog, worker, soak mode |
| `run_stream_readiness.py` | provenance, held-out reproduction, analysis |
| `overlay/` | OBS broadcast overlay and research dashboard |
| `test_live_runtime.py` | E30 tests |
| `analyze_stream_readiness.py` | figures |
