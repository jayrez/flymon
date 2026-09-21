# Experiment 14 — preregistered autonomous progression attempt

Written before any Experiment 14 neural trial. Experiments 1–13 remain unchanged.

## Question and fixed controller

Can the fully frozen `LEFT` / `RIGHT` / `UP` / `A` MaleCNS controller produce sustained autonomous game-state progression from `states/bedroom.state` without reward, scripts, RAM, semantic detectors, random rescue actions, or policy training?

The controller is reused without retuning:

- `LEFT` / `RIGHT`: Experiment 8 DNa02 baseline-relative steering; multiplier 2.0, pooled SD floor 1.62992073549913 Hz, anatomical L evidence maps to LEFT.
- `UP`: Experiment 9 DNg100 baseline-relative locomotion; multiplier 2.0, SD floor 0.5 Hz.
- `DOWN`: disabled.
- `A`: Experiment 13 frozen E4/E6 P20 RMS-z norm, one-decision rising edge, threshold 1.5, per-cell SD floor 1.989834412880423 Hz, hysteresis 0.5, refractory 5 decisions.
- Direction cooldown: 1 decision. Arbitration remains `A` event first, followed by Experiment 10 normalized directional competition.
- Baseline: 10 frozen no-vision windows per trial. Neural cadence: 10 steps / 0.2 simulated seconds per decision. Button hold/release: 8 / 4 PyBoy frames.

P20 membership and order come unchanged from `results/experiment-13-population-event/population.json`. No parameter may be changed after observing Experiment 14 behavior. P20 RMS is interpreted only as aggregate visual drive, never as a semantic interaction signal.

## Seeds, conditions, and horizons

Matched CNS seeds are 921–930. Repository search found no earlier use of these values as CNS seeds. The same ten seeds run all four conditions for 1,500 decisions:

1. `live_full`: current framebuffer; LEFT/RIGHT/UP/A enabled.
2. `live_no_a`: current framebuffer; the same P20 decoder is BCI-ablated; LEFT/RIGHT/UP remain enabled.
3. `frozen_full`: initial bedroom framebuffer is always encoded; LEFT/RIGHT/UP/A remain enabled while actions still advance PyBoy.
4. `no_vision_full`: no visual injection; the full neural model and controller remain active.

One observational `live_full` trial uses CNS seed 950 for 5,000 decisions. Seed 950 previously appeared only as an offline RNG value or capture wait value, not a CNS seed. Trial order is seed-major and condition order is the four-item list above. Every trial reloads the same bedroom savestate once and then receives no reset, manual input, or rescue action.

## Frame timing and local media

Metric decision indices are zero based. At decision `t`, the **input framebuffer** is the exact image used for neural sensory input: the current game frame for live conditions, the frozen initial frame for `frozen_full`, and null for `no_vision_full`. After decoding and applying 8 hold plus 4 release frames, the **resulting framebuffer** is hashed, analyzed, written as the single video frame, and optionally checkpointed.

Videos stream RGB24 160×144 frames directly from memory to FFmpeg stdin at 5 fps. FFmpeg nearest-neighbor scales to 320×288 and writes H.264/libx264, preset `veryfast`, CRF 32, yuv420p, no audio, and faststart. No image sequence or raw-video file is created. FFprobe validation, file size, SHA-256, codec, dimensions, frame rate, duration, and frame count enter the manifest.

Checkpoint boundary 0 is the initial rendered bedroom frame before any decision. A post-action framebuffer is checkpointed whenever `decision + 1` is divisible by 250, and at the final boundary if it is not already divisible by 250. Thus 1,500-decision trials have boundaries 0, 250, …, 1,500; the long run has 0, 250, …, 5,000. Checkpoints contain the unmodified Game Boy framebuffer.

Decision metrics stream as gzip JSON Lines. They are never accumulated as a full trial in the runner. The schema is frozen in `metrics-schema.json`. Each successful standard log must contain exactly 1,500 contiguous rows and the long log exactly 5,000. Allowed actions are LEFT, RIGHT, UP, A, and null/NONE; DOWN is forbidden, and A is forbidden in `live_no_a`.

Videos and checkpoint collections are local-only via narrow `.gitignore` entries. Compressed metric logs will be committed if their total size is below 40 MB; otherwise they remain local with hashes and sizes in the manifest.

## Visual trajectory metrics

Definitions reuse Experiment 10 where applicable:

- Exact state is SHA-256 of the resulting RGBA framebuffer.
- Coarse state is SHA-256 of the Experiment 3 18×20 grayscale grid quantized to 16 levels.
- A coarse transition occurs when consecutive resulting coarse hashes differ.
- Frame distance is mean absolute RGB difference divided by 255.
- A repeated-state run is a consecutive run of one coarse hash. Stuck fraction marks every decision belonging to a run of length at least 10, matching Experiment 10. Longest repeated run is also reported.
- Action entropy is base-2 Shannon entropy over LEFT, RIGHT, UP, A, and NONE.

Novelty is analysis-only. The reference set is every coarse state observed in the first 50 decision results of that trial. A later decision is novel if its state is absent from that reference set. A candidate sustained window begins at decision `t >= 50` when at least 5 of decisions `t..t+9` are novel. Overlapping candidate windows are merged into one sustained novel-state episode. Its start is the first novel decision in the merged interval. Time to first novel state and first sustained episode are reported. None of these values enters the controller.

## A-event consequence analysis

For every actual A decision, record offline frame change at `t`, `t+1`, and maximum change in `t+2..t+5`; whether any resulting coarse transition occurs in `t..t+5`; and whether a sustained novel episode begins in `t+1..t+10`.

Each A event is deterministically paired, without replacement where possible, to the nearest NONE/directional decision in the same trial and same 100-decision block that is at least six decisions from every A event. Ties choose the lower decision index. If no candidate exists in that block, use the nearest qualifying decision in the trial. This comparison is observational; decisions are not treated as independent experimental replicates.

## Trial statistics and interpretation

One seed/trial is the independent unit. Exact paired, two-sided sign-permutation tests enumerate all 2^10 sign assignments for `live_full` against `live_no_a`, `frozen_full`, and `no_vision_full`. Tests are reported for unique coarse states, coarse transitions, sustained novel episodes, stuck fraction, maximum frame distance, and action entropy.

The preregistered primary A endpoint is sustained novel-state episodes for `live_full` versus `live_no_a`. A has a **YES** trajectory effect only if this comparison has p<0.05; **PARTIALLY** if it does not but at least one listed secondary A comparison has p<0.05; otherwise **NO**. Direction of every effect is reported, including harmful effects.

PASS requires either a human-verified M2–M4 semantic milestone, or category B/C visual/interaction evidence together with at least one p<0.05 preregistered primary matched comparison. WEAK applies to category B/C changes without that primary statistical support, or sparse M1 evidence. FAIL applies to category A: no verified semantic progression, no sustained progression-like visual result, and no meaningful live/controller effect. Semantic categories M1–M4 require direct post-hoc human review; hashes alone cannot establish them.

## Review candidates and leakage boundary

At most 12 review candidates are selected from `live_full` matched trials and the long run: largest sustained frame changes, starts of sustained novel episodes, largest changes shortly after A, and maximum distance points. Candidates reference MP4 timestamps and existing checkpoint boundaries; no extra screenshots are written. If no actual visual review occurs, the milestone is `UNVERIFIED`.

Controller input is restricted to MaleCNS motor-DN rates plus frozen baselines and controller temporal state. RAM, coordinates, map/room state, class labels, OCR, detectors, hashes, pixel distances, novelty, stuck metrics, annotations, rewards, and trial outcomes are recording or analysis values only and cannot influence actions.
