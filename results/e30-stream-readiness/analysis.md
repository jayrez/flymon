# E30 — E25 live-controller recovery and stream readiness

**Verdict: STREAM READY WITH CAVEATS.** All five stream-readiness gates pass. The infrastructure is ready for
an unattended research stream. The gameplay is honest but weak: it never reaches Route 1 or a battle, and it is
not conventionally better than uniform-random button presses on aggregate fitness.

## 0. Repository

| Item | Value |
|---|---|
| E29 status | merged before E30 began: PR #14 at `origin/main` 3e8cb99, containing d5389b4 |
| E30 base | 3e8cb99a1c49de190dd07db2bd2ff246be7ff946 |
| Branch | `e30-e25-live-controller-recovery` |
| E25 source | commit 40d644b (PR #9, never merged), recovered file by file; see `e25-recovery-audit.md` |

## 1. Recovery

What was already on main (class A):

- `flymon/fast_io.py`, byte-identical;
- every E25 runtime dependency: `frozen_t4`, `gameplay_eval`, `emulator`, `column_motion`, `retina`,
  `run_generation_zero`, `run_fixed_background_experiment`, and the start state.

What was ported (class B), all verbatim:

- `flymon/evolution.py`;
- `run_evolution.py` (its training stages are class C);
- `test_evolution.py`, whose 23 tests pass on current main.

Historical artifacts (class E), restored unchanged: the E25 results and figures.

Reconciled by hand (class D): `.gitignore` and `PROJECT_STATUS.md`.

**Champion.** Taken from the committed artifact (`finalists.json` rank 5, cross-checked with `champion-weights.json`):

- genome SHA-256 **9fd9795d01af51133ff26b349d186417e5611b452e722dafb4a8187fc1ad31df**;
- 253 parameters: W 7 × 35, b 7, log T; T = 2.56;
- actions: NONE, UP, DOWN, LEFT, RIGHT, A, B;
- 35 inputs: T4a–d × L/R, a 2 × 3 region grid, their deltas, and the previous action.

## 2. Reproduction (S1)

The reproduction ran on the E25 held-out seeds 2601–2620, 1,500 decisions each, using the new runtime
(`flymon/live_runtime.py`). Tolerances were preregistered.

**The result is exact.** In all five conditions, every seed has identical fitness and identical E25 milestones
(100/100 episodes). Every champion per-decision log (20 × 1,500 decisions) is identical to the E25 log.

| Metric | Historical E25 | Recovered (E30) |
|---|---|---|
| Median fitness | 5,428.46 | 5,428.46 |
| Leaves bedroom | 100 % | 100 % |
| Leaves house (M3) | 65 % | 65 % |
| Oak's Lab | 25 % | 25 % |
| Route 1 | 0 % | 0 % |
| Battle | 0 % | 0 % |
| Unique tiles (mean) | 82.4 | 82.4 |

## 3. Vision dependence and baselines (S2)

Paired by seed, exact sign-flip test.

| Condition | Median | Change vs intact | Seeds favouring champion | p | House exit | Oak's Lab |
|---|---|---|---|---|---|---|
| intact champion | 5,428 | — | — | — | 0.65 | 0.25 |
| no vision | 1,327 | −75.6 % | 16/20 | 0.0045 | 0.15 | 0.10 |
| shuffled T4 geometry | 1,243 | −77.1 % | 20/20 | 2e−6 | 0.00 | 0.00 |
| matched-rate random | 905 | −83.3 % | 20/20 | 2e−6 | 0.00 | 0.00 |
| uniform random | 1,384 | −74.5 % | 13/20 | **0.060** | 0.45 | 0.05 |

**Does the controller's useful behaviour depend on the frozen visual pathway? Yes.** Removing vision or
scrambling the T4 geometry destroys house exits and Oak's Lab visits.

The aggregate-fitness comparison against uniform random is still not conventionally significant (p = 0.060).
This is unchanged from E25 and is reported, not hidden.

## 4. Dialogue and stalls

The evaluator-only classifier was preregistered. It labels each long window event (≥ 30 decisions) as a
*stall*, *scripted* or *navigable*:

- **Stall:** the window image is frozen for ≥ 30 decisions, or the event is re-triggered at the same spot for a
  third time or more.
- **Scripted:** Oak's intercept, or the arrival speech in Oak's Lab.
- **Navigable:** everything else.

An overworld stall is ≥ 150 decisions with no movement.

| Condition | Long events | Scripted | Navigable | Stall | Seeds with genuine stall | Stall decisions |
|---|---|---|---|---|---|---|
| champion | 30 | 11 (37 %) | 15 (50 %) | 4 (13 %; all repeat loops) | 2/20 | 2.7 % |
| no vision | 37 | 4 | 31 | 2 | 1/20 | — |
| shuffled | 27 | 0 | 26 | 1 | 2/20 | — |
| uniform random | 59 | 2 | 46 | 11 | 5/20 | — |

E25's "text-window lock on 40 % of seeds" was mostly legitimate: Oak's scripted story (37 %) and advancing
menus (50 %). Genuine stalls are rare. They are repeated starter / "don't go away" prompts in Oak's Lab, where
the champion keeps declining with B.

Because of this, no controller-improvement experiment was run in E30. The preregistered E30 improvement
experiment was optional, and the diagnosis does not show a dialogue-robustness problem big enough to justify
one. The real limit is progress past Oak's Lab.

## 5. Controller input boundary

- The controller side (`FrozenController`, `RandomController`) accepts only 144 × 160 RGBA framebuffers
  (`observe_frames`) and a no-argument `decide()`. It holds no emulator handle.
- RAM is read only after the action, by the evaluator.

The tests prove the following:

- controller sources never reference RAM, the evaluator, milestones or the watchdog;
- inputs that are not frames are rejected;
- two episodes with identical frames but *different* RAM streams produce identical features and actions;
- the worker receives only `(episode_id, seed)`;
- the genome hash is re-checked after every decision (the controller is frozen).

## 6. Runtime architecture

| Component | Role |
|---|---|
| Simulation worker | PyBoy, frozen T4 pathway, frozen controller and evaluator. It sends telemetry outwards and receives only episode assignments and an abort flag. |
| Supervisor | watchdog, SQLite persistence (`runtime/stream/stream-state.sqlite`) and the telemetry server |
| Telemetry server | GET-only HTTP: `/events` (SSE), `/state`, `/schema`, `/provenance`, `/frame.jpg`, `/stream.mjpg`, `/overlay` (broadcast, transparent, OBS browser source), `/dashboard` (research). POST, PUT and DELETE return 405. |

- **Episode policy** (`runtime-config.json`): 3,000 decisions (10 minutes at real time), 60 fps pacing, seeds
  40001 and up (persisted), and every episode starts from the approved `bedroom.state` (SHA-256 305212ea…).
- **Recovery:** no mid-episode resume.
- **Hashes:** the ROM SHA-256 is recorded but the ROM is not committed.

## 7. Watchdog

The watchdog may:

- restart the worker on process exit or on a heartbeat older than 60 s;
- end an episode early and rotate the seed, on a frozen frame for 600 decisions, an identical
  (map, x, y, window) state for 900 decisions, or a wall-clock time above 900 s.

It never presses buttons, advances text, moves the player or edits RAM (source test). Every intervention is
logged, marked on the episode and pushed to the overlay.

## 8. Soak test (S3)

Preregistered in `runtime-config.json`: 6 hours at real time, with injected faults at 1 h (SIGKILL, a simulated
crash) and at 3 h (SIGSTOP, a simulated hang).

| Metric | Value |
|---|---|
| Uptime | **6.00 h**, unattended |
| Episodes | 37 started, **33 completed** (5, 5, 6, 5, 6, 6 per hour); mean 600 s |
| Crashes | 1 (the injected kill), recovered in 0.5 s |
| Hangs | 1 (the injected stop), detected after 60 s and restarted |
| Watchdog interventions | 3: crash restart, hang restart, and one stuck-state abort (episode 23, which had reached Oak's Lab, at decision 2,101) |
| Unplanned crashes or hangs | 0 |
| Longest uninterrupted run | 10,740 s (2.98 h, between the injected faults) |
| Decisions per second | median 5.00 (min 4.99) |
| Real-time ratio | 1.00 |
| Worker CPU | median 15 % |
| Worker RSS | 413.7 → 414.1 MB (+0.4 MB) |
| Supervisor RSS | ≤ 41 MB |
| GPU | the runtime is CPU-only. The 35 % / 775 MB peak came from the E26–E30 GPU test suite run concurrently during the soak, not from the stream. |

Progress in the 33 completed 10-minute episodes (furthest milestone reached):

| Milestone | Share of episodes |
|---|---|
| left bedroom | 100 % |
| left house | 70 % |
| explored Pallet Town | 55 % |
| Oak's Lab | 48 % |
| Route 1 | 0 % |
| battle | 0 % |

The median fitness was 5,871. Oak's Lab was first reached in episode 1.

## 9. Observability (S4) and provenance (S5)

**S4.** `observability-check.json` records a check against the live soak stream:

- every endpoint answered;
- the decision messages validate against `telemetry-schema.json`;
- all required fields were present (action, recent actions, seed, episode, runtime, decisions, decisions/s,
  visual-pathway status, T4 summary, controller features and output probabilities, current / episode / all-time
  milestone, unique tiles, watchdog);
- POST was rejected.

The overlay JavaScript was executed with a live message and a stubbed DOM, and it fills every panel.

**Not verified:** the rendered page has not been viewed in a real browser or in OBS. This must be done before
public launch.

**S5.** Provenance appears in three places: `scientific-provenance.json`, the dashboard's provenance panel, and
the broadcast footer and README architecture section, which state explicitly that RAM never enters the
controller. The whole-brain FlyBrain simulation is disclosed as **not** being in the live loop.

## 10. Gates

| Gate | Result |
|---|---|
| S1 Historical reproduction | **PASS**: exact (100/100 episodes, all decision logs) |
| S2 Vision dependence | **PASS**: −75.6 % no vision, −77.1 % shuffled |
| S3 Unattended robustness | **PASS**: 6.00 h, both injected faults auto-recovered, 0 unplanned failures |
| S4 Stream observability | **PASS**: all endpoints and fields; read-only |
| S5 Honest provenance | **PASS** |

**STREAM READY WITH CAVEATS.** The caveats:

- weak gameplay: no Route 1 or battle, and loops at the Oak's Lab starter prompts;
- p = 0.06 vs uniform random on aggregate fitness;
- the overlay has not been viewed in OBS.

## 11. Scientific framing

**The live system is:** a frozen, connectome-derived fly T4 motion-vision model that turns Pokémon frames into
neural activity, plus a 253-parameter linear readout. The readout was evolved outside the fly model (E25) and
maps that activity to buttons.

**It is not:** within-lifetime learning, dopamine-driven learning, a whole-brain simulation in the control loop,
or a system that learned Pokémon. Nothing is trained while it plays.

## 12. Required answers

1. **Was E29 merged before E30 began?** Yes (PR #14, 3e8cb99).
2. **Base commit:** 3e8cb99a1c49de190dd07db2bd2ff246be7ff946.
3. **E25 functionality already present on main:** `flymon/fast_io.py` (identical) and all E25 runtime
   dependencies (`frozen_t4`, `gameplay_eval`, `emulator`, `column_motion`, `retina`, `run_generation_zero`,
   E22 contexts, start state).
4. **Components recovered:** `flymon/evolution.py`, `run_evolution.py`, `test_evolution.py`,
   `analyze_evolution.py`, and the E25 results and figures (historical).
5. **Was the historical champion recovered exactly?** Yes, from the committed artifact, with the hash verified.
6. **Parameter hash:** 9fd9795d01af51133ff26b349d186417e5611b452e722dafb4a8187fc1ad31df (253 parameters).
7. **Does it reproduce E25 performance?** Yes, exactly: median 5,428.46, identical per seed and per decision.
8. **Does no-vision still degrade it?** Yes: −75.6 % (p = 0.0045).
9. **Does shuffled T4 geometry still degrade it?** Yes: −77.1 % (20/20 seeds).
10. **Versus matched-rate random:** it wins on 20/20 seeds (p = 2e−6); median 5,428 vs 905.
11. **Versus uniform random:** it wins on 13/20 seeds, p = 0.060 (not conventionally significant); median 5,428
    vs 1,384.
12. **Leaves the bedroom:** 100 % (held-out and soak).
13. **Leaves the house:** 65 % held-out (1,500 decisions); 70 % in the soak (3,000 decisions).
14. **Reaches Oak's Lab:** 25 % held-out; 48 % in the soak.
15. **Reaches Route 1:** no.
16. **Enters a battle:** no.
17. **Share of long dialogue events that are scripted:** 37 % scripted and 50 % navigable; only 13 % are
    genuine stalls.
18. **Genuine stall rate:** 2/20 seeds, 2.7 % of decisions (all repeat loops in Oak's Lab).
19. **Can it run unattended?** Yes: 6.00 h with no human action.
20. **Longest uninterrupted runtime:** 2.98 h, bounded by the injected faults. The supervisor itself ran 6.00 h.
21. **Crash count:** 1 (injected); 0 unplanned.
22. **Watchdog interventions:** 3 (2 for the injected faults, 1 stuck-state episode abort).
23. **Decisions per second:** 5.00 at real time. Unpaced, a single process runs the champion at about 48 decisions/s (1,500 decisions in 31.5 s).
24. **Real-time ratio:** 1.00.
25. **Is controller RAM isolation proven by tests?** Yes (`test_live_runtime.InputBoundary`: 4 tests,
    including the different-RAM / identical-actions test).
26. **Does the overlay work independently of the controller?** Yes. It is GET-only with one-way data flow, POST
    returns 405, and the worker never reads telemetry.
27. **Is provenance displayed accurately?** Yes, in JSON, the dashboard panel, the broadcast footer and the
    README. Whole-brain FlyBrain is disclosed as not in the loop.
28. **Status:** STREAM READY WITH CAVEATS.
29. **Biggest blocker before public launch:** gameplay quality. The agent reliably stalls in Oak's Lab (starter
    prompt declined with B, "don't go away" loops) and never reaches Route 1. On the technical side, the overlay
    still needs a real OBS / browser visual check.
30. **Next task:** E31 stream hardening. Verify the overlay in OBS, run a 24 h soak, package the launch. Then a
    preregistered E32 progress experiment: evolve beyond Oak's Lab, for example a short action-history /
    temporal readout, without RAM inputs, measured against this champion on held-out seeds.
