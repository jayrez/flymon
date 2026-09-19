# Experiment 11 - preregistered neural interface expansion

Written before Experiment 11 calibration or held-out outcomes. Experiments 1-10 remain unchanged.

## Architecture and fixed cadence

The experiment extends `flymon.motor.resolve_motor_populations`, the frozen-baseline abstractions used by Experiments 8-9, and Experiment 10's `FrozenExplorationController`. The Experiment 3 18x20 LC10a/LPLC2 projection, 10 neural steps (200 ms) per decision, 10 no-injection baseline windows, 8 PyBoy hold frames, 4 release frames, and one-decision directional cooldown are unchanged. Baselines are frozen before active stimulation.

No controller receives pixels, encoder statistics, labels, RAM, coordinates, novelty, stuck state, reward, routes, or desired actions.

## DOWN shortlist and rationale

Calibration seeds 701-720 (selection 701-710, internal validation 711-720); held-out seeds 721-740. These blocks are absent from Experiments 1-10.

- MDN: established backward-walking reference; summed bilateral population.
- DNp01: giant-fiber escape output, included as a retreat/escape comparator; summed bilateral population.
- DNa05 and DNa07: literature-annotated slow-locomotion DNs; summed bilateral population.
- DNb02 and DNd02: literature-annotated slow-locomotion DNs; summed population.

Exact MaleCNS indices, IDs, sides, and counts are resolved from local metadata and written before simulation. The fixed calibration panel is no vision, uniform, looming, receding, vertical motion, horizontal motion, left-biased, right-biased, canonical bedroom, and deterministically shuffled bedroom. Each has 20 windows.

Candidate thresholds are 1.0, 1.5, and 2.0 times max(per-trial baseline SD, pooled selection-seed within-baseline residual SD floor, 0.5 Hz). For each candidate and multiplier, objective is visual threshold-crossing fraction minus no-vision crossing, constrained to no-vision crossing <=10%; ties favor larger multipliers. Population selection maximizes that frozen objective; ties use shortlist order. Held-out tests compare live bedroom with no vision using paired seed sign permutations for continuous delta and action-rate differences. PASS requires positive neural and action effects with two-sided p<0.05 plus successful BCI-side ablation. MDN is always reported as reference.

## A-event shortlist and rationale

Calibration seeds 741-760 (selection 741-750, internal validation 751-760); held-out seeds 761-780.

- DNp01: giant-fiber phasic escape command with looming input.
- DNp09: looming-associated freezing/stopping output.
- DNp10 and DNp07: rapid landing/leg-extension descending pathways.
- DNp02: posterior descending population included as a nearby escape-pathway comparator.
- MDN: tonic locomotor comparator.

Calibration uses no vision, uniform, abrupt luminance onset, abrupt luminance offset, looming onset, large contrast alternation, left onset, and right onset, each for 20 windows. Candidate threshold selection uses the same multipliers/floor and <=10% no-vision event constraint. Objective is controlled-event rate minus no-vision event rate. The event decoder requires a below-to-above threshold crossing and has a five-decision refractory. A is held 8 frames and released 4. Candidate and threshold freeze before held-out seeds. PASS requires a positive live-vs-no-vision held-out event effect with two-sided paired p<0.05, sparse A output <=10% decisions, successful refractory checks, and ablation eliminating A.

## Natural controls and integration

Both held-out arms run 60 decisions from `states/bedroom.state`: live, no vision, frozen initial frame, and deterministic spatial shuffle. Trial/seed is the independent unit. Controls use matched seeds.

Only independently PASS channels enter integration. Integrated seeds 781-800 run 500 live decisions from the bedroom. Existing DNa02 LEFT/RIGHT and DNg100 UP remain frozen. A has priority when its rising event fires; otherwise all enabled directions compete by normalized threshold excess, retaining Experiment 10 tie order. No channel is enabled merely because it helps gameplay. Visual transitions are post-hoc annotations only.
