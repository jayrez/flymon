# Experiment 25 — First evolutionary Pokémon controller (preregistration)

Written before any evolution run. A smoke run follows; it is infrastructure only, uses a tiny
population and budget, and is not used for any decision. Settings are frozen after the
smoke run. Every later change is appended as a timestamped amendment.

## Base

Branch `worktree-bridge-cse_014eULVFzQa6eL4wV7d1hV9L`, fast-forwarded (no rewrite) to
`origin/main` `b86ddb3`, the PR #8 merge containing E24 `de239d5`, E23 `e6a88c5`,
E22 `09833da` and E21 `74bf6f6`. The working tree was clean.

## Frozen sensory model (not evolvable)

This is exactly the E24 gameplay sensory configuration:

* framebuffer → E5 retina column luminance → the fixed-reference, pre-adapted E23 native T4
  model (`flymon/frozen_t4.py`);
* frozen projection weights and temporal constants;
* the frozen T4 → FlyBrain injection (gain 2.04903, cap 0.8, 16 levels).

The configuration + anatomy SHA-256 is `87829806e398f66b…` (`sensory-hash.json`), identical to
E24. Genomes contain no sensory parameter. Tests enforce immutability.

**Engineering accelerations** (`flymon/fast_io.py`) change no semantics:

| change | numerical check | speed-up |
|---|---|---|
| column sampling via sRGB LUT + sparse sampling matrix | max \|Δ luminance\| 3e−8; max \|Δ T4\| 2e−8 | 4.05 → 0.51 ms/frame |
| injection indices moved to GPU once per decision | spikes bit-identical | 11.8 → 6.8 ms/step |
| `VectorInjector`: all injection groups in one indexed add (replicates `FlyBrain.step` op by op) | spikes bit-identical over 200 steps, 2.85 M spikes | 6.7 → 2.2 ms/step |

## Controllers

**Architectures.** Same action space, genome form and evolutionary machinery for all three.

| id | features | total features |
|---|---|---|
| **dn** (A) | E14 DN rates DNa02 L/R, DNg100 L/R, MDN L/R, DNp01 L/R, z-scored against the episode's own no-vision baseline (SD floor 1 Hz), + P20 RMS-z; + one-decision deltas (18) | 25 |
| **t4** (B) | frozen-T4 pooled activity: 4 subtypes × 2 eyes, and a 2 × 3 grid of RF-centre screen regions (u halves × v thirds), scaled by 1 / 0.016453 (E24 calibration median); + one-decision deltas (28) | 35 |
| **t4dn** (C) | B + A | 53 |

* All architectures also receive a one-hot of the controller's own previous action (7).
* Features are clipped to ±10.
* The T4 → FlyBrain injection stays active wherever FlyBrain runs (dn, t4dn, Gen0), because it
  is the current downstream path.

**Decoder.** Linear softmax, `p = softmax((W x + b) / T)`, sampled with the per-episode rng.

* This is the "C1: linear + short history" level: history enters through the deltas and the
  previous action.
* Genome = W (7 × F), b (7), log T (clipped to [−3, 1.6]).
* Nothing else is evolvable: no timing, no sensory values, no game-state logic.

**Actions.** NONE, UP, DOWN, LEFT, RIGHT, A, B. START and SELECT are disabled.

**Timing** (E24, unchanged): each decision is 10 FlyBrain steps (0.2 s) plus 12 game frames
(8 held + 4 released). There is no refractory period beyond the softmax policy.

**Controller input** is restricted to the features above; `ControllerInputGuard` checks DN
rate keys. RAM, coordinates, map, dialogue, battle, milestones and fitness never reach the
controller.

## Episodes and seeds

* Start state: `bedroom.state`.
* Seed-derived start wait of 0–59 idle frames (`rng([seed, 2500])`). All frames are seen by
  the frozen T4 model.
* The seed also sets FlyBrain noise (`brain.reset(seed)` in E14's baseline windows) and the
  policy rng (`rng([seed, 2501])`).
* **Training seeds:** 2501–2508 (new). Generation g evaluates **every** candidate, elites
  included, on the pair `SEED_PAIRS[g mod 4]`. The rotation is deterministic and balanced, and
  all candidates in a generation share the same seeds.
* **Held-out seeds:** 2601–2620 (new; never used in training or selection).
* **Budget:** 500 decisions in training (100 s of game time); 1,500 decisions on held-out
  (the E24 budget).

## Fitness (evaluator only; `FitnessTracker`)

```
F = Σ milestone bonus  + 5 · Σ_maps min(tiles_in_map, 40) + 50 · (maps − 1)
    + 100 · min(completed window cycles, 3)
    − 300 · loop_fraction(non-idle) − 300 · idle_fraction − 200 · min(dialogue locks, 3) − 100 · revisit_rate
```

**Milestone bonuses** (E24 definitions):

| milestone | bonus |
|---|---|
| M1 leaves spawn tile | 100 |
| M2 leaves bedroom | 1,000 |
| M3 reaches Pallet Town | 3,000 |
| M4 ≥ 10 Pallet tiles | 1,000 |
| M5 another map | 3,000 |
| M7 Route 1 | 5,000 |
| M8 battle | 5,000 |

**Interaction.** A completed window cycle means the window opens and then closes within ≤ 30
decisions. A dialogue lock is a window open for 30 consecutive decisions. M6 (window shown)
earns nothing by itself.

**Dominance.** Leaving the house (M3, 3,000) beats any in-house exploration + interaction
(maximum 750). Leaving the bedroom (M2, 1,000) beats any bedroom-only score (maximum 500).

**Anti-gaming:**

* no reward for raw actions, distance or A presses;
* tiles are capped per map and counted as positions, not decisions;
* repeated transitions between the same two rooms add nothing (unique maps only);
* an open dialogue earns nothing and a lock is penalised.

Candidate fitness is the mean over the candidate's two seeds.

## Evolution

Mutation + elitism, no crossover:

* elites: top ⌈12.5 %⌉, kept and re-evaluated each generation;
* offspring: Gaussian mutation of the elites (σ_W 0.1, σ_b 0.2, σ_logT 0.1);
* immigrants: 10 % fresh random genomes;
* initial genomes: W ~ N(0, 0.1), b ~ N(0, 0.1), T = 1 (near-uniform random policy);
* ranking: deterministic (fitness, then genome hash).

**Stage 1 — smoke:** 6 genomes, 2 generations, 60 decisions, all architectures.
Infrastructure only.

**Stage 2a — architecture screen (training seeds only):**

* each architecture: population 24, 8 generations, independent rng (25000 + index);
* architecture score = mean over the last 3 generations of the top-quartile mean fitness;
* the highest score wins.

**Stage 2b — main run.** The winning screen lineage continues:

* population 32 (its elites plus mutants and immigrants);
* 30 more generations;
* rng 25100.

**Early stop:** after main generation 15, stop if the top-quartile score has not exceeded
its best for 12 consecutive generations, or on an infrastructure error.

**Checkpoints.** Every generation saves:

* the population, fitness and per-seed metrics;
* elites and the next population;
* the rng state, sensory hash, architecture and seeds.

Runs are resumable.

**Finalists.** Genomes evaluated in ≥ 2 of the last 3 main generations (≥ 4 episodes),
ranked by mean training fitness over those generations; top 8, filled from the rest if
needed. Finalists are frozen.

## Held-out evaluation (once, after finalists are frozen)

* 8 finalists × 20 held-out seeds × 1,500 decisions × {V0 normal, V1 no vision (T4
  features zero, T4 injection off), V2 shuffled T4 geometry (per-seed column permutation,
  rng 25000000 + seed; affects features and injection)}.
* **Baselines on the same seeds:**
  * B0 — Gen0 (the frozen E14 controller, E24 path, no DOWN);
  * B1a — E24 matched-rate random;
  * B1b — uniform random over the 7 actions;
  * the best final-generation genome of each non-winning screen architecture.
* **If t4dn wins:** champion with DN features zeroed (T4 part only) and with T4 features
  zeroed (DN part only; injection still on).

**Champion** among finalists (V0): highest 10th-percentile fitness, then median, then mean.

**Rejection** — a finalist is not a livestream candidate if any of these holds:

* worst case < Gen0 median;
* M3-or-later on < 3 seeds;
* no-vision median within 10 % of normal;
* shuffled median within 10 % of normal;
* median non-idle loop fraction > 0.5;
* dialogue lock on > 20 % of seeds.

## Success criteria (held-out, champion, V0)

| id | criterion |
|---|---|
| S1 | median F > Gen0 median and > both random medians, each with exact paired sign-flip p < 0.05 over 20 seeds |
| S2 | M3 or later on ≥ 25 % of seeds |
| S3 | DOWN ≥ 1 % of decisions |
| S4 | median F under V1 ≥ 30 % below V0 **and** under V2 ≥ 15 % below V0 |

## Readiness

| outcome | condition |
|---|---|
| **READY FOR CHAMPION STAGING** | S1–S4 all met, and the champion passes every rejection check |
| **EVOLUTION WORKS, NOT STREAM-READY** | training improves (median of the last 3 main generations > median of screen generation 0 of the same architecture) and the champion beats Gen0 on held-out (S1 vs Gen0), but the S2/S4/robustness checks are incomplete |
| **NO USEFUL EVOLUTION** | otherwise |

Vision-dependence is claimed only if S4 holds.

### Settings frozen — 2026-09-23T05:12Z (after smoke + throughput; before the screen)

Smoke run (6 genomes × 2 generations × 60 decisions, all architectures) found one
infrastructure bug: a missing directory for generation summaries. It is fixed; the smoke
outputs live in `smoke/` and are not used for any decision.

Throughput (`throughput.json`, 8 workers):

* t4: 320 decisions/s (near-linear);
* t4dn: 67 decisions/s (GPU-bound, ~96 % utilisation);
* peak GPU memory 2.9 GB.

All settings above are unchanged.

### Amendment 1 — 2026-09-23T08:52Z (after the held-out run; nothing re-run, no decision changed)

Observations that affect interpretation only:

* **Screen margin.** The screen chose t4 by a margin smaller than generation-to-generation
  noise (scores: t4 1601, dn 1586, t4dn 1265). The preregistered rule was applied as written.
* **Finalist #8** had only 2 training evaluations. It filled the list per the rule, because
  only 7 genomes had ≥ 4.
* **"Dialogue lock" measure.** Any text window open ≥ 30 consecutive decisions counts as a
  lock. On the 5 champion seeds that reach Oak's Lab, the long windows are Professor Oak's
  scripted story dialogue after the player walks north from Pallet Town. Those windows are
  progression, not a stuck menu. The rejection criterion is still applied as registered.
  E26 should separate scripted-story dialogue from stuck menus.
* **FlyBrain contribution.** Because t4 won, the planned T4 + DN feature ablation of the
  champion was not applicable. FlyBrain's contribution is instead assessed descriptively from
  the held-out scores of the best screen dn and t4dn genomes, which are different decoders.
