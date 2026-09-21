# Experiment 14 analysis

This report uses seed/trial as the inferential unit. Semantic milestones require direct human review and are never inferred from hashes or distances.

```text
Experiment 14 verdict:
PASS
Primary category:
D
Base commit:
9027e4727e3f17b9a6f840a0c285d517d0cf3740
Question:
Can the frozen LEFT/RIGHT/UP/A MaleCNS controller autonomously progress Pokemon from bedroom.state over long horizons?

CONTROLLER
LEFT: frozen Experiment 8 DNa02 baseline-relative steering
RIGHT: frozen Experiment 8 DNa02 baseline-relative steering
UP: frozen Experiment 9 DNg100 baseline-relative locomotion
DOWN: disabled
A: frozen Experiment 13 P20 population RMS-z rising edge
NONE: existing deadband / cooldown
Were any controller parameters changed?
NO

PRIMARY MATCHED RUNS
Seeds:
921-930
Horizon:
1500 decisions
Conditions:
LIVE_FULL
LIVE_NO_A
FROZEN_FULL
NO_VISION_FULL

LIVE_FULL
LEFT %: 19.3333
RIGHT %: 7.8867
UP %: 1.0267
A %: 1.4733
NONE %: 70.2800
Unique coarse states: 60.8000
Sustained novel-state episodes: 1.4000
Stuck fraction: 0.200067
Action entropy: 1.089278
Max frame distance: 0.541530
Coarse transitions: 795.3000

LIVE_NO_A
LEFT %: 19.8200
RIGHT %: 8.1733
UP %: 1.0867
A %: 0.0000
NONE %: 70.9200
Unique coarse states: 69.1000
Sustained novel-state episodes: 1.2000
Stuck fraction: 0.248200
Action entropy: 1.026127
Max frame distance: 0.576171
Coarse transitions: 791.9000

FROZEN_FULL
LEFT %: 18.1000
RIGHT %: 4.8867
UP %: 0.9467
A %: 3.2467
NONE %: 72.8200
Unique coarse states: 68.5000
Sustained novel-state episodes: 2.1000
Stuck fraction: 0.301333
Action entropy: 1.101221
Max frame distance: 0.582768
Coarse transitions: 630.7000

NO_VISION_FULL
LEFT %: 9.0600
RIGHT %: 2.7600
UP %: 0.6800
A %: 0.0533
NONE %: 87.4467
Unique coarse states: 47.9000
Sustained novel-state episodes: 1.8000
Stuck fraction: 0.458467
Action entropy: 0.626191
Max frame distance: 0.534484
Coarse transitions: 383.7000

MATCHED STATISTICS
LIVE_FULL vs LIVE_NO_A:
unique-state effect: -8.300000
p: 0.492188
novel-episode effect: 0.200000
p: 1.000000
stuck effect: -0.048133
p: 0.433594
frame-distance effect: -0.034641
p: 0.250000
transition-count effect: 3.400000
p: 0.949219
action-entropy effect: 0.063151
p: 0.001953

LIVE_FULL vs FROZEN_FULL:
unique-state effect: -7.700000
p: 0.402344
novel-episode effect: -0.700000
p: 0.250000
stuck effect: -0.101267
p: 0.074219
frame-distance effect: -0.041238
p: 0.046875
transition-count effect: 164.600000
p: 0.054688
action-entropy effect: -0.011943
p: 0.832031

LIVE_FULL vs NO_VISION_FULL:
unique-state effect: 12.900000
p: 0.230469
novel-episode effect: -0.400000
p: 0.625000
stuck effect: -0.258400
p: 0.001953
frame-distance effect: 0.007046
p: 0.312500
transition-count effect: 411.600000
p: 0.001953
action-entropy effect: 0.463087
p: 0.001953

EFFECT OF A
Did A significantly change autonomous trajectories?
NO
Preregistered broad A-effect classification:
PARTIALLY (triggered only by action entropy; no visual trajectory endpoint was significant)
Direction of effect:
unique_coarse_states -8.3 (p=0.492188); coarse_transitions +3.4 (p=0.949219); sustained_novel_state_episodes +0.2 (p=1); stuck_fraction -0.0481333 (p=0.433594); maximum_frame_distance -0.0346412 (p=0.25); action_entropy +0.0631514 (p=0.00195312)
A events:
221 matched LIVE_FULL; 124 long run; 345 reviewed total
A-associated visual changes:
{"coarse_transition_within_t_to_t_plus_5": 0.9246376811594202, "frame_change_t": 0.025688506002368647, "frame_change_t_plus_1": 0.01141444269487511, "max_frame_change_t_plus_2_to_5": 0.032697171417420325, "sustained_novel_episode_within_t_plus_1_to_10": 0.008695652173913044}
Matched non-A comparison:
means {"coarse_transition_within_t_to_t_plus_5": 0.8782608695652174, "frame_change_t": 0.01394242317498658, "frame_change_t_plus_1": 0.011888477925389157, "max_frame_change_t_plus_2_to_5": 0.021260794082125604, "sustained_novel_episode_within_t_plus_1_to_10": 0.02318840579710145}; A-minus-matched {"coarse_transition_within_t_to_t_plus_5": 0.046376811594202816, "frame_change_t": 0.011746082827382067, "frame_change_t_plus_1": -0.0004740352305140482, "max_frame_change_t_plus_2_to_5": 0.01143637733529472, "sustained_novel_episode_within_t_plus_1_to_10": -0.014492753623188406}

LONG RUN
Condition:
LIVE_FULL
Seed:
950
Horizon:
5000
LEFT %: 19.9200
RIGHT %: 9.4600
UP %: 0.9600
A %: 2.4800
NONE %: 67.1800
Unique coarse states: 151
Sustained novel-state episodes: 1
Longest stuck interval: 34
Maximum distance from initial: 0.648524
Coarse transitions: 2815

SEMANTIC REVIEW
Highest verified milestone:
M3
Verified observations:
["Human post-hoc MP4 review: live seed 921 autonomously traversed from the initial upstairs bedroom to the downstairs living room and back repeatedly around decisions 285-345.", "Human post-hoc checkpoint review: independent live seeds 923 and 925 also reached the downstairs living room after starting in the bedroom; seed 921 scheduled checkpoints independently show the same room change.", "Human post-hoc checkpoint review: observational long-run seed 950 reached and moved within the downstairs living room.", "Human post-hoc MP4 review: the neural A emitted at live seed 922 decision 0 opened the visible 'Flymon is Playing the SNES!' interaction; later neural A pulses advanced and closed the text.", "Live seed 928 produced repeated visible SNES interaction text states, consistent with recorded neural A pulses.", "A balanced audit of every scheduled checkpoint in all matched conditions found downstairs-room views in LIVE_NO_A seeds 923, 925, 927, and 930; FROZEN_FULL seeds 923 and 925; and NO_VISION_FULL seeds 923 and 925.", "The same balanced control audit found PC/menu or SNES interaction states in FROZEN_FULL and an SNES text state in NO_VISION_FULL.", "No exterior location, overworld traversal, or task-like M4 progression was verified.", "Matched controls also changed rooms and interacted; semantic review therefore does not show that live feedback or A uniquely caused progression."]
Candidate events needing manual review:
12
Was the initial bedroom visibly left?
YES
Did A cause a verified game interaction?
YES

ARTIFACTS
MP4 files:
41
Total MP4 size:
9.15 MiB
Checkpoint PNGs:
301
Checkpoint cadence:
every 250 decisions
Total checkpoint size:
0.55 MiB
Per-decision metrics:
65000
Expected metric records:
65000
Metrics integrity:
PASS
Compressed metrics size:
3.63 MiB
Accidentally persisted raw framebuffer files:
0
Videos verified:
PASS
Manifest:
results/experiment-14-progression/trial-manifest.json

Leakage checks:
RAM: PASS
player coordinates: PASS
map/room state into controller: PASS
screen classifier into controller: PASS
dialogue detector into controller: PASS
frame hash into controller: PASS
visual novelty into controller: PASS
stuck recovery: PASS
scripted route: PASS
scripted A: PASS
random actions: PASS
trained policy: PASS
reward: PASS

Primary result:
D
Did Flymon achieve autonomous Pokemon progression?
YES
Current scientifically defensible controls:
LEFT
RIGHT
UP
A
Primary limitation:
DOWN remains unavailable; without it the controller can become geometrically trapped. P20 A is aggregate visual drive rather than a semantic interact-now signal. Room changes and interactions also occurred in matched controls, so Experiment 14 does not show that live vision or A uniquely caused the verified semantic progression.
```

## Interpretation

Flymon produced verified unassisted room-to-room movement in multiple LIVE_FULL trials and in the long run, so this is limited autonomous Pokemon progression (category D). The same checkpoint audit found room changes in LIVE_NO_A, FROZEN_FULL, and NO_VISION_FULL controls. No favorable LIVE_FULL-vs-FROZEN_FULL visual trajectory endpoint reached p < 0.05, so the stronger category E claim is not supported. LIVE_FULL did reduce stuck fraction and increase coarse transitions relative to NO_VISION_FULL, showing that visual drive changed long-horizon behavior, but this does not isolate dynamic feedback from the presence of vision.

Adding A did not significantly change any visual trajectory endpoint. The preregistered broad rule says PARTIALLY solely because action entropy changed; that change is expected when an extra action category is enabled and is not evidence of a richer trajectory. Human review nevertheless verified that particular neural A pulses opened and advanced the bedroom SNES interaction. This is an observed causal button consequence, not evidence that P20 represents semantic interaction intent.

The sustained-novel-state count is coarse: overlapping 10-decision windows are merged, so one long episode can cover many state changes. It is kept as preregistered and is not interpreted as a count of semantic milestones.
