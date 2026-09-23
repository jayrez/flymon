# Experiment 25 — First evolutionary Pokémon controller

**Outcomes:**
* **Evolution:** WORKS.
* **Vision dependence:** YES for the champion under the preregistered S4 test, but not uniform
  across finalists.
* **Livestream readiness:** EVOLUTION WORKS, NOT STREAM-READY.

A 253-parameter linear decoder reading only pooled frozen-T4 activity was evolved for 38
generations (8 screen + 30 main). On 20 unseen seeds it:

* leaves Red's house on 65 % of seeds and reaches Oak's Lab on 25 %;
* crushes Gen 0 and the E24 matched-rate random baseline (20/20 seeds each, p = 2e−6);
* loses 76–77 % of its median fitness when vision is removed or the T4 geometry is shuffled.

It does **not** significantly beat uniform random over the full 7-button action space
(13/20 seeds, p = 0.06). It is rejected as a livestream champion because text windows stay open
≥ 30 decisions on 40 % of seeds; 5 of those 8 seeds are the ones that reach Oak's Lab, where the
long window is Oak's scripted story dialogue.

Base `b86ddb3` (PR #8). Frozen sensory hash `87829806e398f66b…` is identical to E24 and was
checked in every generation.

## 1. Setup

* **Frozen sensory model:** E23 native T4 via `flymon/frozen_t4.py`. Never modified; no
  genome contains a sensory parameter; tests enforce this.
* **Engineering accelerations** (no semantic change, tested):
  * LUT + sparse column sampling: 8× faster, |Δ| ≤ 3e−8;
  * vectorised FlyBrain injection: bit-identical spikes, 3× faster per step.
* **Throughput** (8 workers): t4 320 decisions/s; t4dn 67 decisions/s (GPU-bound).
* **Actions:** NONE, UP, DOWN, LEFT, RIGHT, A, B (START/SELECT disabled); E24 timing
  (8 + 4 frames per decision).
* **Genome:** linear softmax decoder W (7 × F), b (7), temperature.
* **Features** (all include the previous action):
  * dn: 9 DN / population features + deltas;
  * t4: 14 pooled T4 features (subtype × eye, 2 × 3 screen regions) + deltas;
  * t4dn: both.
* **Fitness:** progression-dominant (M2 1,000, M3 3,000, M4 1,000, M5 3,000, M7/M8 5,000),
  plus capped tile exploration, completed interactions, and penalties for loops, idling,
  locks and revisits. Evaluator RAM only.
* **Training:** seeds 2501–2508, rotating pairs, 500 decisions per episode.
* **Held-out:** 20 unseen seeds (2601–2620), 1,500 decisions per episode.

## 2. Architecture screen (training seeds only; population 24, 8 generations)

| architecture | score (top-quartile mean, last 3 generations) | best fitness | median last generation | best M3 rate in a generation |
|---|---|---|---|---|
| DN only | 1,586 | 3,455 | 873 | 0.12 |
| **T4 only** | **1,601** | 5,021 | 835 | 0.08 |
| T4 + DN | 1,265 | 3,057 | 1,191 | 0.02 |

T4-only won by the preregistered rule, with a margin within noise. T4 + DN was the weakest
screen lineage. DN-only was nearly tied with T4-only; on held-out, its best screen genome
scored a median of 1,450, against 1,326 for the best T4 + DN genome.

## 3. Main evolution (T4 only, population 32, 30 generations, no early stop)

The top-quartile training score rose from about 1,300 at the start of the main run to
2,000–3,400 in its second half; main generation 19 reached 3,400. The median training fitness
rose from 633 (screen generation 0) to 1,214 (mean of the last 3 generations). Leaving the
house (M3) went from 0–8 % of episodes to 5–16 %. Training at 500 decisions never reached
Route 1 or a battle.

**First occurrences** (lineage generation index; screen generations 0–7, then main from 8):

| event | generation |
|---|---|
| first DOWN use | 0 |
| first bedroom exit | 0 |
| first house exit | 0 (random initial genomes already leave occasionally once DOWN exists) |
| first outdoor navigation (M4) | 0 |
| first completed interaction | 0 |
| first consistent bedroom exit (≥ 50 % of episodes) | 3 |
| first other map (M5) | 3 |
| Route 1 / battle | never |

## 4. Finalists (held-out, normal vision; 20 seeds × 1,500 decisions)

| # | genome | mean | median | p10 | worst | M3 rate | M5 rate | no-vision median | shuffled median | rejections |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 17ddb220 | 4,300 | 5,088 | 372 | 222 | 0.60 | 0.20 | 1,295 | 1,365 | lock 30 % |
| 2 | f91da80e | 3,942 | 3,102 | 1,458 | 1,337 | 0.50 | 0.15 | 1,211 | 1,420 | lock 25 % |
| 3 | c43ec633 | 3,993 | 3,072 | 1,418 | 1,221 | 0.50 | 0.20 | 1,284 | 1,282 | lock 35 % |
| 4 | 22cbbc50 | 3,207 | 2,999 | 955 | 868 | 0.50 | 0.00 | 1,194 | 1,327 | lock 65 % |
| **5** | **9fd9795d** | **4,753** | **5,428** | **1,485** | **1,327** | **0.65** | **0.25** | **1,327** | **1,243** | **lock 40 %** |
| 6 | 5863ee08 | 2,948 | 1,551 | 1,295 | 364 | 0.40 | 0.00 | 1,445 | 1,106 | no-vision within 10 % |
| 7 | 4897636a | 3,246 | 1,543 | 362 | 357 | 0.45 | 0.05 | **4,414** | 1,241 | — |
| 8 | d0c2f05e | 3,232 | 1,516 | 392 | 370 | 0.40 | 0.10 | **5,451** | 1,343 | lock 25 % |

**Champion: #5 `9fd9795d01af`.** It has the highest 10th percentile (then median).

* Temperature 2.56.
* Action use: UP 22 %, B 22 %, LEFT 16 %, RIGHT 15 %, DOWN 11 %, NONE 9 %, A 5 %.
* Held-out reach: Pallet Town on 13/20 seeds, Oak's Lab on 5/20. Every replay reproduces its
  held-out fitness exactly.

## 5. Baseline comparison (held-out medians)

| controller | median | p10 | unique tiles | maps | M3 rate | loop (median) | idle |
|---|---|---|---|---|---|---|---|
| Gen 0 (E14 controller) | −23 | −756 | 11.8 | 1.2 | 0 | 0.3 | 0.40 |
| random, E24 matched rate | 905 | −170 | 21.9 | 1.7 | 0 | 0.4 | 0.50 |
| random, uniform over 7 buttons | 1,384 | 730 | 78.5 | 2.4 | 0.45 | 0.7 | 0.00 |
| best screen DN-only | 1,450 | 851 | 88.7 | 2.8 | 0.45 | 0.5 | 0.00 |
| best screen T4 + DN | 1,326 | 1,229 | 36.1 | 2.0 | 0 | 0.4 | 0.00 |
| **evolved champion (T4)** | **5,428** | **1,485** | 82.4 | 2.9 | **0.65** | **0.3** | 0.00 |

Paired over 20 seeds (exact sign-flip test):

| comparison | seeds favouring champion | mean difference | p |
|---|---|---|---|
| champion vs Gen 0 | 20/20 | +4,741 | 2e−6 |
| champion vs matched random | 20/20 | +4,171 | 2e−6 |
| champion vs uniform random | 13/20 | +1,930 | **0.060** |

## 6. Vision causality (champion, held-out)

| condition | median | p10 | M3 rate | M5 rate | drop vs normal | paired |
|---|---|---|---|---|---|---|
| normal T4 | 5,428 | 1,485 | 0.65 | 0.25 | — | — |
| no vision (T4 features zero) | 1,327 | 907 | 0.15 | 0.10 | **−76 %** | 16/20 seeds, p = 0.004 |
| shuffled T4 geometry | 1,243 | 910 | **0.00** | 0.00 | **−77 %** | 20/20 seeds, p = 2e−6 |

S4 is met. With its visual input removed or scrambled, the champion falls to random-like
bedroom and ground-floor wandering; it never leaves the house with shuffled geometry.

Vision dependence is **not** universal across the evolved population:

* finalists #1–#5 lose 58–76 % under no-vision;
* #6 is unaffected;
* #7 and #8 *improve* without vision (their no-vision behaviour stumbles into better
  outcomes).

The shuffled-geometry drop holds for all finalists (11–77 %).

## 7. FlyBrain contribution

T4-only won the screen, so the champion has no DN inputs. Two observations bear on
FlyBrain's contribution:

* **Training screen:** T4 + DN (1,265) was below both T4-only (1,601) and DN-only (1,586).
* **Held-out best screen genomes:**
  * DN-only median 1,450; not different from uniform random (paired, 11/20 seeds, p = 0.38);
  * T4 + DN median 1,326; significantly *worse* than uniform random (mean −1,483, p = 0.015).

At this budget, FlyBrain DN features did not add useful information beyond direct T4
features. The screen margins are small, so this is weak evidence, not a proof of no value.

## 8. Champion weights (descriptive only; causality comes from §6)

* **Weight magnitude by group:** T4 0.41, previous action 0.53.
* **Weight magnitude by T4 subtype:** T4a 0.55, T4c 0.48, T4d 0.45, T4b 0.33.

Strongest weights per action:

| action | strongest weights |
|---|---|
| DOWN | + T4a (left eye), + previous DOWN / RIGHT, + right-middle region activity |
| LEFT | + left-middle region activity, − T4c (left eye), − change in right-eye T4b |
| RIGHT | + T4a (right eye), + change in right-top region, + previous UP / DOWN |
| UP | + change in left-eye T4d, − change in right-eye T4a, − right-eye T4d |
| B | + previous NONE, + bottom-region activity (left and right), − T4a (left eye) |
| A | suppressed after A; − T4a (right eye), − top-left region activity |

A plausible reading, not proven: the bottom-region activity that drives B is where the text box
sits; and screen-half activity plus lateral T4 subtypes bias turning. The ablations, not the
weights, establish that the policy uses vision.

## 9. Required answers

1. **Screen winner:** T4 only, by a near-tie with DN only.
2. **Did DN-only evolve useful behaviour?** Partly. It leaves the house on held-out
   (M3 45 %, M5 25 %), with median 1,450, but does not beat uniform random (p = 0.38).
3. **Did T4-only evolve useful behaviour?** Yes: the champion reaches median 5,428.
4. **Did T4 + DN outperform both?** No. It was the lowest in the screen, and on held-out it was
   significantly worse than uniform random (p = 0.015).
5. **How many generations were run?** 38 (8 screen + 30 main), plus 8 each for the losing
   architectures.
6. **Population size:** 24 in the screen, 32 in the main run.
7. **Did fitness improve?** Yes. Top-quartile score ~1,300 → 2,000–3,400; median 633 → 1,214;
   M3 rate up to 16 % in training.
8. **Which milestones appeared first?** DOWN, bedroom exit, house exit, outdoor navigation and
   interaction at generation 0 (the new action space alone enables them); a consistent bedroom
   exit and another map at generation 3; Route 1 and battle never.
9. **Is DOWN used meaningfully?** Yes: 11 % of the champion's actions, and required for every
   house exit.
10. **Did any candidate leave Red's house?** Yes: the champion on 13/20 held-out seeds, and
    all finalists on 8–13 seeds.
11. **Route 1 or later?** No. Oak's Lab (story event) on 5/20 seeds; no Route 1, no battle.
12. **Beats Gen 0?** Yes, 20/20 seeds, p = 2e−6.
13. **Beats random?** Beats matched-rate random (p = 2e−6). Beats uniform random on median
    5,428 vs 1,384, but not significantly (13/20, p = 0.06).
14. **Does disabling vision reduce performance?** Yes: −76 % (p = 0.004).
15. **Does shuffled geometry reduce performance?** Yes: −77 % (p = 2e−6), and no house exits.
16. **Did FlyBrain/DN add value beyond T4?** Not detectably.
17. **Finalist robustness:** p10 of 360–1,485, with worst cases above the Gen-0 median for all
    8, but high variance. The champion's p10 of 1,485 is only about a bedroom exit.
18. **Remaining failure modes:**
    * long text-window runs (40 % of seeds, partly Oak's story dialogue);
    * bimodal outcomes (house exit or not);
    * a 30 % non-idle loop fraction (dithering);
    * no Route 1;
    * the 500-decision training budget is shorter than evaluation;
    * vision dependence varies across the population.
19. **Livestream candidate?** Not yet: EVOLUTION WORKS, NOT STREAM-READY.
20. **E26:** see §10.

## 10. Experiment 26 recommendation

One concrete next step: **continue the T4-only lineage with a 1,500-decision training budget
and a revised, preregistered interaction term.**

* The interaction term should reward advancing a text window (A or B within N decisions of
  the window appearing) and should separate scripted-story windows from stuck-menu windows.
* Add **no-vision and shuffled-geometry episodes to training fitness** as penalties, so
  selection explicitly favours vision-dependent genomes.
* Then re-run this held-out protocol on fresh seeds.

## 11. Runtime and throughput

| stage | wall time |
|---|---|
| architecture screen | 1 h 45 min (t4 10 min, dn 47 min, t4dn 48 min) |
| main training (30 generations × ~96 s) | 48 min |
| held-out (580 episodes × 1,500 decisions) | 47 min |
| champion logs | 2 min |
| throughput benchmark | ~5 min |
| analysis | < 1 min |

**Episode counts:**

* screen: 1,152;
* main: 1,920;
* held-out: 580;
* champion logs: 20;
* total: **3,672 episodes (~2.8 M decisions)**.

**Parallel efficiency:**

| architecture | workers | decisions/s | efficiency |
|---|---|---|---|
| t4 | 1 | 45 | — |
| t4 | 8 | 320 | 89 % |
| t4dn | 1 | 20 | — |
| t4dn | 8 | 67 | 42 %, GPU-bound, peak GPU memory 2.9 GB |

## Files

Results (this directory):

* `preregistration.md` (plus settings freeze and Amendment 1)
* `sensory-hash.json`, `seeds.json`, `throughput.json`
* `screen-*-generations.jsonl`, `main-generations.jsonl`, `architecture-screen.json`
* `checkpoint-manifest.json`, `elite-history.json.gz`
* `checkpoints/<phase>/` — final checkpoint of each phase committed; the rest local
* `finalists.json`, `heldout-raw.json.gz`, `champion-evaluation.json`, `champion-weights.json`
* `champion-heldout-logs.json.gz`
* `smoke/`

Figures: `captures/experiment-25/01–10*.svg`.

Code:

* `flymon/fast_io.py`
* `flymon/evolution.py`
* `run_evolution.py`
* `analyze_evolution.py`
* `test_evolution.py`
