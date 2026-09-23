# Experiment 24 — Generation 0 gameplay benchmark

**Progression decision: READY WITH CAVEATS.** The closed-loop infrastructure is sound:
* it runs unattended;
* the frozen E23 T4 model is verifiably in the loop;
* the controller receives only neural rates;
* replays are bit-identical within and across processes;
* throughput scales to about 8 instances.

Generation 0 itself shows that **vision currently has almost no influence on behaviour.**
* Gen-0 and no-visual actions are 94 % identical.
* Gen-0 explores fewer tiles than matched-rate random actions.
* The frozen E14 action set cannot leave Red's house, because DOWN is disabled.

These three caveats must shape Experiment 25.

Scientific provenance: gameplay uses the frozen E23 T4 visual readout (E23 `e6a88c5`, via
PR #7 merge `6ba4849`). T5 direction selectivity remains unresolved and is not represented
as a validated motion pathway in this gameplay baseline. E23 stays recorded as a
preregistered FAIL / E / NO-GO; the project-level CONDITIONAL GO is what licenses this
gameplay work.

## 1. Frozen sensory model and pipeline

`flymon/frozen_t4.py` streams the E23 native candidate one game frame at a time. It is
bit-exact against E23's batch simulation (max |Δ| 3e−16 on real stimuli; regression-tested).

| setting | value |
|---|---|
| reference | fixed linear background 0.5; L0 = 0.49693 |
| adaptation | adapted start ≡ 40 L0 frames |
| transfer | G = 16, beta = 2 |
| tonic rates | r0_exc = 1; Mi4 phasic |
| E_inh | −0.2 |
| filters | tau 4 / 12 |
| constants | sigma 0.2204, M 0.1424 |
| population | 6,845 T4a–d neurons |
| configuration hash | `87829806…` |

**Discrepancy found and documented.** MaleCNS T4 projections include small Tm1 / Tm2 /
Tm9 inputs (0.43 % of |w|). E23's code simulated them, while E23's report says "Mi1 + Mi4
only". E24 freezes the code as it actually ran; E23 artifacts are not edited.

**Immutability.** The configuration is a `MappingProxyType`, the projection arrays are
write-protected, and every run stores the SHA-256. The frozen / evolvable boundary is set
out in `frozen-sensory-config.json`.

Pipeline, as actually wired and audited:

```
PyBoy framebuffer, every game frame (60 fps, no skipping, no async buffering)
  -> E5 retina column luminance (linear light, 825 columns)
  -> FrozenT4Readout (12 updates per decision)
  -> mean T4 response over those 12 frames
  -> T4Injection: clip(2.049 r, 0, 0.8) in 16 levels, into the matching MaleCNS T4 cells
  -> FlyBrain LIF MaleCNS v1.0, 10 steps (0.2 s)
  -> frozen E14 controller (E8 DNa02 LEFT/RIGHT, E9 DNg100 UP, E13 P20 A; DOWN/B/START/SELECT off)
  -> button 8 frames held + 4 released (0.2 s game time)
```

* **Rates:** decision rate 5 / s of game time; action rate ≤ 5 / s.
* **Refractory rules:** direction cooldown 1 decision; A refractory 5 decisions.
* **Injection gain:** 0.8 / p99 of positive responses to E22 calibration bars, fixed before
  any gameplay.
* **Replaced input:** this T4 injection takes the place of E14's E3 grid → LPLC2/LC10a
  injection.

## 2. Protocol

* **Episodes:** 10 seeds (2401–2410, fixed in advance and unused before) × 4 conditions ×
  1,500 decisions (300 s game time).
* **Start state:** `states/bedroom.state` (SHA-256 `305212ea…`), spawn RedsHouse2F (3, 6).
* **Emulator:** PyBoy 2.7.0; local ROM (not committed).

| condition | description |
|---|---|
| gen0 | frozen T4 injection live |
| C1 | T4 computed, injection off |
| C2 | per-seed column-shuffled T4 geometry |
| C0 | iid actions from the pooled Gen-0 action law, frozen before C0 |

## 3. Results (means over 10 seeds; min–max)

| metric | Gen 0 | C0 random | C1 no visual | C2 shuffled |
|---|---|---|---|---|
| completed runs | 10/10 | 10/10 | 10/10 | 10/10 |
| non-NONE actions | 119 (20–269) | 116 | 116 | 121 |
| actions / min | 23.8 (4–54) | 23.1 | 23.1 | 24.2 |
| idle fraction | 0.921 | 0.923 | 0.923 | 0.919 |
| successful moves | 37.9 (0–119) | **83.5** | 31.7 | 40.5 |
| unique tiles | 11.5 (1–23) | **22.6** (15–29) | 10.1 | 10.9 |
| unique maps | 1.7 | 1.6 | 1.6 | 1.6 |
| locomotion entropy (bits) | 1.07 | 1.28 | 1.01 | 1.07 |
| revisit rate (moves onto visited tiles) | 0.27 | 0.74 | 0.26 | 0.27 |
| loop fraction (any flag) | 0.645 (0.35–0.98) | 0.704 | 0.646 | 0.653 |
| — inactivity | 0.549 | 0.503 | 0.570 | 0.539 |
| — L/R oscillation | 0.098 | 0.396 | 0.077 | 0.120 |
| — menu loop | 0.043 | 0 | 0.084 | 0.017 |
| — wall bump / A spam | 0 / 0 | 0.001 / 0 | 0 / 0 | 0 / 0 |

Action totals for Gen 0 over 15,000 decisions: NONE 13,809, LEFT 582, RIGHT 449, UP 148,
A 12. DOWN, B, START and SELECT: 0 (disabled).

**Milestone attainment** (fraction of seeds):

| condition | M0 | M1 | M2 | M6 | M3–M5, M7, M8 |
|---|---|---|---|---|---|
| Gen 0 | 1.0 | 0.9 | 0.7 | 0.1 | 0 |
| C0 | 1.0 | 1.0 | 0.6 | 0 | 0 |
| C1 | 1.0 | 0.9 | 0.6 | 0.1 | 0 |
| C2 | 1.0 | 0.9 | 0.6 | 0.1 | 0 |

**Paired comparisons** (exact sign-flip p over 10 seeds; descriptive):

* **Gen 0 vs C0 random** — random explores more:
  * unique tiles −11.1 (0/10 seeds favour Gen 0; p = 0.002);
  * successful moves −45.6 (p = 0.016);
  * loop fraction −0.06 (p = 0.42);
  * best milestone +0.6 (p = 0.50).
* **Gen 0 vs C1 no visual:**
  * unique tiles +1.4 (p = 0.13);
  * successful moves +6.2 (p = 0.06);
  * loop fraction −0.001 (p = 0.97);
  * milestones and maps identical except one seed.
* **Gen 0 vs C2 shuffled:** unique tiles +0.6 (p = 0.75), moves −2.6 (p = 0.25), loop
  −0.009 (p = 0.75).
* **Action identity:** 94.1 % of Gen-0 decisions produce the same action as C1 on the same
  seed (per seed 88–99 %), and the same holds against C2.

**The T4 model is demonstrably in the loop:**
* the Gen-0 runs inject a mean of 274 T4 cells per decision;
* C1 injects 0;
* C2 injects 546, and its hashes differ per seed.

**But the downstream FlyBrain → DN path barely registers the T4 injection within a 10-step
window.** Behaviour is set mainly by the per-seed network noise (brain.reset(seed)).

**Best baseline run (not a champion; nothing tuned on it).** Seed 2404:
* M1 at decision 9, M0 at 74, and M2 (reaches RedsHouse1F) at 910;
* 23 unique tiles, 114 successful moves, loop fraction 0.37;
* 175 LEFT, 55 RIGHT, 15 UP, 1 A.

Like every M2 run, it stops at the 1F stair tiles (6, 1) / (7, 1). The house exit is at the
bottom of the room, and DOWN is not in the frozen action set.

## 4. Dominant failure modes

1. **Idling.** 92 % of decisions are NONE, and 55 % of all decisions sit inside ≥ 20-decision
   idle runs. The E14 DN deadband rarely triggers.
2. **Action-space cap.** Without DOWN, no condition can exit the house, so M3 and beyond are
   unreachable.
3. **Vision-insensitive output.** Actions are 94 % identical with injection off.
4. **L/R dithering** at the top of the bedroom and on the 1F stairs.
5. **Occasional dialogue lock.** Seed 2405: A at the SNES, then the window stays open ~98 %
   of the episode.

## 5. Throughput and parallelism

Single instance, idle machine: **5.9 decisions/s = 70 game fps = 1.17× real time**.

Per decision:

| stage | ms |
|---|---|
| T4 model | 62 (12 frames; ~4 ms/frame column luminance on CPU, ~0.7 ms model) |
| FlyBrain | 100 (28 ms without injection in C1: per-step host→GPU index transfer of ~16 injection groups costs ~70 ms) |
| emulator | 3 |
| controller | 0.3 |
| evaluator | 0.1 |

Memory: 1.3 GB RSS and ~360 MiB GPU per instance.

| instances | per-instance decisions/s | aggregate decisions/s | aggregate × real time | peak GPU mem | GPU util |
|---|---|---|---|---|---|
| 1 | 5.87 | 5.9 | 1.2 | 359 MiB | 5 % |
| 2 | 5.90 | 11.8 | 2.4 | 715 MiB | 18 % |
| 4 | 5.70 | 22.8 | 4.6 | 1.4 GiB | 33 % |
| 8 | 4.52 | 36.2 | 7.2 | 2.9 GiB | 50 % |

**Recommendation.** Run 6–8 concurrent instances on this machine, each near real time.
Before building a population harness, two cheap engineering wins are available:

* cache injection index arrays on the GPU (≈ 3× brain speed-up);
* vectorise column sampling.

Both are engineering changes that do not alter the model. FlyBrain's batch mode (several
flies per sparse multiply) is a further multiplier.

## 6. Reproducibility and anti-cheating

* **Reset.** Each episode reloads `bedroom.state` and verifies the spawn (38, 3, 6). Tests
  show reset restores RAM telemetry exactly, and two emulator instances are isolated.
* **Determinism.** The same seed gives identical actions and positions: two runs in one
  process (200 decisions), and a fresh process against the benchmark log (300 decisions).
  No non-determinism was observed.
* **Recorded per run:** run ID, seed, start time, save-state hash, spawn, sensory hash,
  injection config, controller hash and timing.
* **What the controller can access:** only DN / population spike rates
  (`ControllerInputGuard`, whitelisted keys, never tripped).
* **What the controller cannot access:** RAM, coordinates, map, window state, milestones,
  loop flags, pixels, hashes, scripts.
* **Evaluator RAM reads** happen after the action and are logged only.

## 7. Proposed Experiment 25 fitness (not implemented)

Specified in `fitness-proposal.json`:

```
F = 100 * Σ_k w_k M_k + 2 · N_tiles(capped 60/map) + 10 · N_maps_new + 5 · I_interact(open+close, cap 5)
    − 20 · L_loop(excluding idle) − 10 · I_idle − 0.5 · R_revisit_moves
w(M1..M8) = (0.2, 1, 2, 2, 3, 2, 4, 6)
```

M6 counts only if the window closes again. The terms are designed not to reward:

* raw action counts;
* distance;
* pacing in one room (the tile cap);
* dialogue locks.

Everything comes from evaluator telemetry, never controller input. The frozen T4 model lies
outside the search space.

Two prerequisites from E24:

1. the evolvable action set must include DOWN (and B);
2. the evolvable downstream readout should observe the frozen T4 output directly,
   because the current T4 → FlyBrain → DN route transmits almost nothing.

## 8. Future champion evaluation (not executed)

Specified in `champion-protocol.json`:

* **Training:** cheap fitness on disjoint training seeds.
* **Finalists:** top 8 by mean F over the last 3 generations.
* **Evaluation:** each finalist on ≥ 20 unseen pre-registered seeds, the same save state
  (plus held-out states if available), and the same 1,500-decision budget.
* **Report:** mean, median, 10th percentile and worst case, plus milestone rates, against
  Gen 0 and random on the same seeds.
* **Selection:** maximise the 10th-percentile F, tie-broken by median. Reject brittle
  candidates (worst case below the Gen-0 median, or milestones on fewer than 3 seeds).
* **Vision controls:** re-run the champion with vision off and with shuffled T4 geometry.
* **Livestream:** the brain is the robust champion, with its sensory and controller hashes
  displayed.

## 9. Required answers

1. **Is the frozen E23 T4 model in the closed loop?** Yes. Its hash is recorded, it is
   updated every game frame, and it injects into ~274 MaleCNS T4 cells per decision (0 in
   C1).
2. **Can episodes start and reset reproducibly?** Yes: state reload, verified spawn and
   bit-identical replays.
3. **Does the controller run unattended for the full budget?** Yes. All 40 episodes
   completed 1,500 decisions with no intervention.
4. **How many baseline runs were completed?** 10 Gen-0 runs, plus 30 control runs.
5. **What actions does Generation 0 produce?** 92 % NONE; LEFT 3.9 %, RIGHT 3.0 %, UP
   1.0 %, A 0.08 %.
6. **Does it move meaningfully?** Modestly: 38 successful moves per episode (0–119) and
   11.5 unique tiles.
7. **Does it escape the starting room?** In 7 of 10 runs, reaching RedsHouse1F.
8. **Which milestones are reached?**
   * M0 in 10 runs;
   * M1 in 9;
   * M2 in 7;
   * M6 in 1 (a dialogue that never closes);
   * none beyond M2.
9. **How much exploration occurs?** 11.5 tiles in 1.7 maps on average.
10. **How often does it loop?** 64.5 % of decisions carry a loop flag, mostly idling
    (54.9 %), then L/R dithering (9.8 %).
11. **Does it outperform random actions?** No. Random explores about 2× more tiles
    (p = 0.002). Milestones are similar.
12. **Does removing visual input make behaviour worse?** Barely: 94 % of actions are
    identical, with small, non-significant gains for vision.
13. **Does shuffled geometry make behaviour worse?** No detectable effect.
14. **Are runs reproducible across seeds?** Each seed is exactly reproducible. Across seeds,
    behaviour is dominated by brain noise.
15. **What is the effective simulation speed?** 1.17× real time (5.9 decisions/s) per
    instance.
16. **How many parallel instances are practical?** 6–8 (8 × 4.5 decisions/s = 36 aggregate).
17. **What is the biggest gameplay bottleneck?** The loop does not pass vision through to
    action: T4 injection → DN coupling is negligible. This combines with the DOWN-less
    action set and a 92 % idle deadband. Computationally, the bottleneck is per-step GPU
    injection transfers.
18. **What fitness should Experiment 25 use?** The §7 formula, after enabling DOWN and giving
    the evolvable readout access to T4.
19. **How should future champions be evaluated?** By the §8 robust multi-seed protocol.
20. **Is the system ready for generation experiments?** **READY WITH CAVEATS.**

## Runtime

* Setup, smoke and preregistration: ~10 min.
* Gen-0, C1 and C2 batch: 3 concurrent processes, ~45 min wall.
  * Gen-0 / C2: ~250 s per episode.
  * C1: ~143 s per episode.
* C0 random: 40 s.
* Replay checks: ~3 min.
* Parallelism benchmark: ~4 min.
* Analysis: < 1 min.

## Files

Results (this directory):

* `preregistration.md` — frozen settings plus Amendment 1
* `run-metadata.json`, `frozen-sensory-config.json`, `run-manifest.json`
* `runs/<condition>/seed-*.json` — per-run metrics, metadata and performance
* `episodes/<condition>/seed-*.jsonl.gz` — replayable decision logs, 2 MB total
* `c0-action-law.json`
* `aggregate-metrics.json` — aggregates, milestones, controls, in-loop verification, best
  run, parallel scaling
* `readiness.json`, `replay-check.json`
* `parallel/`
* `fitness-proposal.json`, `champion-protocol.json`

Figures: `captures/experiment-24/01–09*.svg`. Videos and live-status JSON are local-only
(gitignored).

Code:

* `flymon/frozen_t4.py`
* `flymon/gameplay_eval.py`
* `run_generation_zero.py`
* `analyze_generation_zero.py`
* `test_generation_zero.py`
