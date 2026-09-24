# E30 — Reproduction, vision-ablation and dialogue-classification preregistration

This document was frozen before the held-out reproduction run; its hash is in
`reproduction-preregistration-freeze.json`.

**Disclosed smoke check.** One earlier probe ran the unmodified historical `run_evolution.run_task` on seed
2601. All 1,500 decisions and the fitness (1488.517) were identical to the E25 log. It does not replace
the run below.

## Runtime under test

The run below uses the new runtime, not the historical runner: `flymon/live_runtime.py` (`Episode`,
`FrozenController`, `RandomController`) with the evaluator `flymon/stream_eval.py`.

- **Champion:** `results/e30-stream-readiness/champion-genome.json`, SHA-256 9fd9795d01af5113…, recovered from
  E25 commit 40d644b.
- **Seeds:** held-out 2601–2620 (the original E25 held-out seeds).
- **Budget:** 1,500 decisions.
- **Timing:** E24/E25, 8 frames held + 4 frames released.

Conditions:

| Condition | Description |
|---|---|
| champion \| normal | intact vision |
| champion \| none | no vision |
| champion \| shuffled | T4 columns shuffled with seed 25,000,000 + seed, as in E25 |
| random_matched | E24 C0 action law |
| random_uniform | uniform over the 7 actions |

## S1 — Historical reproduction

The primary expectation is **exact** equivalence. The T4 runtime is CPU-only, and E25 reported exact
replays. The pass rule is:

- the champion's median fitness is within ±10 % of the historical 5,428.46;
- house exits are common (rate ≥ 0.40);
- Oak's Lab is reached sometimes (0 < rate < 1);
- Route 1 is not reliably reached (rate < 0.5).

The following are reported alongside the pass rule: per-seed identity against `heldout-raw.json.gz` for
every condition, and per-decision identity against `champion-heldout-logs.json.gz`. The champion is not
tuned.

## S2 — Vision dependence

Both no-vision and shuffled-T4 must reduce the champion's median held-out fitness by ≥ 50 %.

## Statistics

- Paired by seed.
- Exact two-sided sign-flip test on per-seed fitness differences (2^20 permutations).
- Reported: median difference, mean difference, dz and the number of seeds favouring the champion.
- No seeds are removed.
- The comparisons against matched-rate random and against uniform random are both reported.

## Dialogue / menu classification (evaluator only; `flymon/stream_eval.py`)

**Long window event:** at least 30 consecutive decisions with the window layer shown (rWY < 144). Each
event is classified in this order:

1. **Stall**, if either:
   - the window-region image is unchanged for ≥ 30 consecutive decisions within the event; or
   - the event starts at a (map, x, y) where ≥ 2 earlier long events started in the same episode
     (repeated re-triggering).
2. **Scripted:** the Oak intercept at the Pallet Town north exit (map 0, y ≤ 1), or the first long event in
   Oak's Lab (the arrival story).
3. **Navigable:** everything else.

**Overworld stall:** at least 150 consecutive decisions with no window and no (map, x, y) change.

**Genuine-stall rate:** stall-event decisions plus overworld-stall decisions, divided by all decisions.
Seeds with any genuine stall are also reported.

This classification may read RAM and the framebuffer. It is never an input to any controller.

## Stream milestones (evaluator only)

| Milestone | Condition |
|---|---|
| bedroom_exited | map ≠ 38 |
| house_exited | map ∉ {38, 37} |
| pallet_explored | ≥ 10 Pallet Town tiles |
| oak_lab | map 40 |
| route1 | map 12 |
| battle | wIsInBattle ≠ 0 |

E25 M1–M8 are reported unchanged.
