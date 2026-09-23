# Experiment 24 — Generation 0 gameplay benchmark (preregistration / benchmark plan)

Written before any Generation-0 episode. A short smoke test (infrastructure only, not
scored) follows this file. Benchmark settings are frozen at the end of the smoke test.
Any change is appended as a timestamped amendment.

## Base and project decision

Branch `worktree-bridge-cse_014eULVFzQa6eL4wV7d1hV9L`, fast-forwarded (no rewrite) to
canonical `origin/main` `6ba4849`. That is the PR #7 merge containing E21 `74bf6f6`, E22
`09833da` and E23 `e6a88c5`. The working tree was clean.

E23 stays recorded as a preregistered FAIL / category E / NO-GO. The **project-level**
decision is CONDITIONAL GO: gameplay may use the frozen E23 native T4 readout. T5 remains an
unresolved, separate research track.

## Frozen sensory model

`flymon/frozen_t4.py` is a bit-exact streaming form of the E23 native candidate
(`run_spatial_veto_experiment.candidate()`):

| setting | value |
|---|---|
| reference | fixed linear background 0.5; L0 = 0.49693 measured |
| adaptation | adapted start (filters at zero contrast, membrane at tonic rest), identical to E23's 40 L0 frames |
| transfer | G = 16, beta = 2, D1 linear |
| tonic rates | r0_exc = 1; Mi4 phasic (r0 = 0) |
| E_inh | −0.2 |
| filters | tau 4 (fast arm) / 12 (slow arm) |
| constants | sigma = 0.2204, M = 0.1424, chemistry from E21 |

* **Input classes:** everything E23 actually simulated, i.e. all MaleCNS T4 partner classes
  except Mi9. That is Mi1 (83.6 % of |w|), Mi4 (16.0 %), and Tm1 / Tm2 / Tm9 (0.43 %
  combined).
* **Documentation discrepancy:** E23's report calls the candidate "Mi1 + Mi4 only". Its
  code also carried the small Tm inputs. E24 freezes the code as it ran; E23's artifacts are
  not edited.
* **Equivalence:** streaming vs batch E23 simulation differs by at most 3e−16. A regression
  test checks this.
* **Scope:** all 6,845 anatomy-eligible T4a–d neurons.

The configuration is a read-only mapping and the projection arrays are write-protected.
Every run records a SHA-256 of configuration + anatomy.

**Frozen bridge** (`T4Injection`): T4 response → FlyBrain voltage injected into the same
MaleCNS T4 neurons.

* voltage = clip(g · r, 0, 0.8), quantised to 16 levels — the E3/E14 injection convention.
* g = 0.8 / p99 of positive per-neuron per-frame responses to the E22 calibration
  gray-background bars = **2.04903**. It was computed from calibration stimuli only, before
  any gameplay.
* The injection replaces E14's E3 18×20 grid → LPLC2/LC10a injection. The E23 T4 pathway is
  the only visual input.

## Controller (downstream, unchanged)

The Experiment-14 frozen controller is used as is:

* LEFT / RIGHT: E8 DNa02 baseline-relative steering.
* UP: E9 DNg100 locomotion.
* A: E13 P20 RMS-z rising-edge event.
* DOWN, B, START and SELECT are disabled.
* Per-trial no-vision baseline of 10 windows; direction cooldown 1.

These thresholds were selected in E8/E9/E13 on earlier calibration data, **not trained on
gameplay**. E24 adds no optimisation.

**Controller input:** DN / population spike rates only (`ControllerInputGuard`).

## Loop timing

| quantity | value |
|---|---|
| decision | 10 FlyBrain steps (0.2 s simulated) |
| game frames per decision | 12 (8 held + 4 released, 60 fps) — 0.2 s game time |
| T4 update | every game frame (12 per decision) |
| injected value | mean T4 response over the frames since the previous decision, constant over the 10 steps |

No frame skipping and no asynchronous buffering.

## Evaluator (never controller input)

RAM telemetry: map 0xD35E, y 0xD361, x 0xD362, battle 0xD057, window rWY 0xFF4A < 144.
Addresses were verified empirically: spawn is RedsHouse2F (map 38) at (3, 6), and the window
shows on A at the SNES.

Milestones (first decision recorded):

| id | milestone |
|---|---|
| M0 | ≥ 10 non-NONE decisions and ≥ 2 action types |
| M1 | leaves the spawn tile |
| M2 | exits the bedroom |
| M3 | reaches Pallet Town |
| M4 | ≥ 10 unique Pallet Town tiles |
| M5 | another map |
| M6 | text / menu window |
| M7 | Route 1 |
| M8 | battle |

Loop flags:

| flag | rule |
|---|---|
| wall_bump | repeated same-direction press, no tile or map change, no window |
| lr / ud oscillation | ≥ 3 reversals in the last 8 directional decisions, net ≤ 1 tile |
| menu_loop | window shown ≥ 20 consecutive decisions |
| a_spam | ≥ 3 A in 20 decisions |
| inactivity | inside a ≥ 20-decision NONE run |

The loop fraction is the union of these flags.

## Protocol

* **Save state:** `states/bedroom.state`, SHA-256 `305212ea…0dca95`.
* **ROM:** local, not committed; SHA-256 `5ca7ba01…6b7b`. PyBoy 2.7.0.
* **Seeds:** 2401–2410, fixed before any run and verified unused. The seed sets FlyBrain
  noise (`brain.reset(seed)` in E14's baseline).
* **Budget:** 1,500 decisions per episode (E14 convention; 300 s game time).
* **Isolation:** each episode reloads the save state; no reset, rescue or manual input within
  an episode.

| condition | description |
|---|---|
| gen0 | frozen T4 injection live |
| C1 no_visual | T4 computed and logged, injection off; brain and controller live |
| C2 shuffled_geometry | T4 readout built on a column-permuted MaleCNS partner geometry (one fixed permutation per seed, rng 24000000 + seed) |
| C0 random | iid actions from the pooled Generation-0 action law, frozen in `c0-action-law.json` after gen0 and before any C0 episode; numpy rng(seed); same 8 + 4 frame timing; no brain |

**Also run:**

* replay check — the same seed twice, 200 decisions;
* throughput — per-stage timing inside every episode;
* parallelism — 1 / 2 / 4 concurrent instances, 150 decisions each.

## Readiness gate

**READY FOR GENERATIONS** if all of the following hold:

* every episode completes unattended;
* the frozen T4 model is verifiably in the loop (hash recorded, T4 injection count > 0 in
  gen0, and none in C1);
* no controller input beyond neural rates (guard never trips);
* metrics and milestones serialise;
* C0 and C1 complete;
* throughput is measured;
* the replay check reproduces the action sequence exactly or its non-determinism is
  characterised.

**READY WITH CAVEATS** if the infrastructure passes but named issues remain.

**NOT READY** if there are crashes, a bypassed visual model, controller leakage, or
unreliable reset.

Gameplay quality is **not** a criterion.

Comparisons are descriptive. Per-seed paired differences are reported with exact sign-flip
permutation p over 10 seeds for gen0 vs C0 / C1 / C2 on unique tiles, successful moves,
loop fraction and best milestone. Nothing is tuned on them.

### Settings frozen — 2026-09-23T03:46Z (after smoke; no benchmark episode yet)

The smoke test (60 gen0 decisions; 40 decisions per condition) found no infrastructure bug:

* T4 injection reaches ~378 MaleCNS T4 cells per decision in gen0, 0 in C1 and ~742 in C2.
* The guard never tripped.
* 6.0 decisions/s, 1.2× real time.

Benchmark settings are unchanged from above. Gen0, C1 and C2 run as three concurrent
processes, so per-episode stage timings reflect 3-way sharing. The throughput and
parallelism benchmarks are run separately. Gen0 episodes also write local-only video
(not committed).

### Amendment 1 — 2026-09-23T04:37Z (after all benchmark episodes; descriptive metrics only)

Two evaluator metric definitions were wrong. Both are fixed in `gameplay_eval.episode_metrics`
and recomputed offline from the unchanged decision logs (`metrics_version: 2`; the v1 values
are kept in each run file):

* **locomotion_entropy** attributed a tile change to the action of the same decision. A step
  takes 16 frames, but a decision is 12, so most changes land on the following NONE
  decision. This made entropy 0 for every brain-driven run. The move is now attributed to the
  last directional press.
* **revisit_rate** was 1 − unique tiles / decision-level positions, which idle decisions
  swamp. It is now the fraction of successful moves that land on an already-visited tile.

Milestones, loop flags, actions and all controller/sensory behaviour are unaffected (the
milestones were asserted equal). No setting was tuned.

Also noted post hoc: the E14 action set has DOWN disabled, and RedsHouse1F's exit is at
the bottom of the room. M3 and beyond are therefore unreachable by any condition,
including random.
