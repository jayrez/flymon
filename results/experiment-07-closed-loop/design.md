# Experiment 7 — fixed motor-DN BCI protocol

This protocol was written before Experiment 7 neural calibration or closed-loop
outcomes were inspected. Experiments 1–6 and their interpretations remain fixed.

## Question and boundaries

Test whether the unchanged Experiment 3 18×20 LC10a/LPLC2 projection can drive
identified MaleCNS descending-neuron activity that a fixed engineered BCI turns
into reproducible directional Game Boy inputs. The BCI is not a natural Pokémon
motor circuit. It has no learned decoder, reward, plasticity, action labels,
screen class, direct pixel features, game RAM, route, or gameplay objective.

## Verified candidate populations

Local `brain.npz` metadata, rather than hardcoded assumptions, must resolve and
record FlyWire ID, brain index, side, type, superclass, and population count for
DNa02, DNg100, MDN, and DNp01. Empty or non-descending populations abort.

## Calibration

Seeds 301–320. Conditions: left-half dark stimulus, right-half dark stimulus,
uniform gray, no injection, canonical bedroom, Experiment 2 menu frame, and
Experiment 2 dialogue frame. Each condition uses 20 decisions of 10 neural steps
(200 ms per decision, 4 s total). Static images use zero temporal change after
the first presentation. Record per-cell and grouped spike rates.

Controller thresholds are frozen from calibration only: each tonic population
threshold is the no-vision pooled 95th percentile plus 0.5 Hz; the steering
deadband is the no-vision 95th percentile of absolute DNa02 L−R decision-window
rate difference plus 0.5 Hz. Threshold floors are 1 Hz. The DNa02 anatomical side
whose left-minus-right stimulus contrast is larger becomes screen-LEFT evidence;
ties retain L. These rules do not inspect gameplay success. If calibration gives
complete silence or saturation, closed-loop trials still run and the limitation
is reported.

The controller uses a 150 ms exponential smoother, one-decision minimum action
hold, one-decision cooldown, and deterministic largest-excess competition with
tie priority LEFT, RIGHT, UP, DOWN. DNp01 is diagnostic only because escape has
no faithful directional mapping in the allowed control space.

## Coupling cadence and trials

One decision captures the current rendered framebuffer, computes the Experiment
3 grid and change from the prior decision, injects the resulting fixed projection
for 10 MaleCNS steps, and aggregates candidate DN spikes. One selected direction
is held for 8 PyBoy frames and released for 4 frames. Thus neural simulated time
is 200 ms/decision; the nominal Game Boy segment is 12 frames (about 200 ms at
59.7 fps); effective cadence is about 5 decisions/s. Each trial has 40 decisions.

Primary/control seeds are 321–340, distinct from calibration and Experiments 1–6.
Every trial reloads `states/bedroom.state`, renders one frame, resets MaleCNS,
resets encoder history, and resets the controller. Conditions are live current
vision, no vision, frozen initial framebuffer, deterministic spatial shuffle
(one fixed pixel permutation per seed), and controller-null. Null runs compute
the candidate action but apply none. No Pokémon RAM is read or recorded.

The left/right causal assay uses calibration seeds 301–320 and the frozen BCI on
left- and right-biased synthetic patterns for 40 decisions. Interventions on live
vision set DNa02, DNg100, or MDN controller input rates to zero without changing
MaleCNS. Ten seeds 321–330 are used for each intervention to limit GPU cost; all
20 seeds are used for primary controls. Same-seed live repeats use 321–325.

## Records and metrics

Each decision records source and post-action framebuffer SHA-256, grid/encoder
hash and statistics, LC10a/LPLC2 spike counts, candidate DN counts and rates,
smoothed state, selected and applied action, and hold/release frames. Stored data
must reconstruct frame → encoder → DN → action → next frame.

Report five-way action distributions including no-action, actions/trial, entropy,
same-seed agreement, total variation and Jensen–Shannon distances, exact
seed-stratified permutation tests (10,000 random swaps), live/frozen framebuffer
divergence, future DN-rate divergence, and future action disagreement. The causal
assay reports LEFT/RIGHT ratios and a seed-level permutation test. Interventions
must alter their corresponding action channels in the predicted direction.

PASS requires a stable directional stream, significant visual dependence,
predictable left/right manipulation, and measurable later live-versus-frozen
sensory/neural/action divergence. WEAK means actions exist but one or more causal
links are weak or inconsistent. FAIL means silence/saturation/random output or
live vision is essentially indistinguishable from no vision. Leaving the bedroom
is not a metric.

## Prior-art constraint

Upstream `fly.ai` identifies DNa02 steering, DNg100 forward walking, MDN backward
walking, and DNp01 escape groups and uses fixed spike windows, margins, voting,
and cooldowns. DoomFly's baseline smooths verified identified readouts and maps
them through declared fixed joystick gains while excluding pixels, reward, and
game state from its controller. This experiment adopts the transparent fixed-BCI
principle and does not copy DoomFly's solver or gain values.
