# Experiment 10 analysis

```text
Experiment 10 verdict:
PASS

Primary controller:
LEFT: frozen Experiment 8 DNa02
RIGHT: frozen Experiment 8 DNa02
UP: frozen Experiment 9 DNg100
DOWN: disabled
NONE: deadband/cooldown

Primary seeds: 601-620
Trial horizon: 300
Long-run seed: 650
Long-run horizon: 1000

LIVE - 3 DIRECTION

LEFT %: 21.200
RIGHT %: 9.833
UP %: 1.283
NONE %: 67.683

unique exact frames: 22.80
unique coarse visual states: 22.65
unique encoder states: 34.05
mean final frame distance: 0.436629
mean maximum frame distance: 0.442709
mean stuck fraction: 0.152333
action entropy: 0.947130

FROZEN VISION

unique coarse visual states: 30.05
mean final frame distance: 0.450039
mean stuck fraction: 0.332000
action entropy: 0.844586

NO VISION

unique coarse visual states: 22.90
mean final frame distance: 0.387192
mean stuck fraction: 0.548667
action entropy: 0.550866

LIVE vs FROZEN
visual-state diversity effect: -7.4000
p-value: 0.0084
frame-distance effect: -0.020035
p-value: 0.3774
stuck-fraction effect: -0.179667
p-value: 0.0047

LIVE vs NO VISION
visual-state diversity effect: -0.2500
p-value: 0.9418
frame-distance effect: 0.044392
p-value: 0.1385
stuck-fraction effect: -0.396333
p-value: 0.0001

Observed scene transitions:
count: 5
seeds: [601, 617, 618]
decision indices: [286, 118, 234, 233, 247]

LONG RUN
decisions: 1000
actions: 250
unique coarse states: 32
scene transitions: 1
longest stuck interval: 96
most common action: NONE

SECONDARY FOUR-DIRECTION ARM

LEFT %: 26.283
RIGHT %: 4.967
UP %: 0.883
DOWN %: 2.800
NONE %: 65.067

unique coarse states: 19.00
scene transitions: 0.00
stuck fraction: 0.120000

3-dir vs 4-dir exploration difference: -3.6500 mean coarse states

Leakage / hidden-policy checks:
RAM: PASS
player coordinates: PASS
screen-class input: PASS
stuck recovery: PASS
random action injection: PASS
scripted route: PASS
trained policy: PASS
reward: PASS

Primary result:
C

Did Flymon produce sustained, non-scripted, visually contingent autonomous Pokemon exploration using only pixels -> MaleCNS -> identified descending-neuron BCI control?
PARTIALLY
```

Scene transitions are visual-only annotations. MDN DOWN remains unvalidated in the secondary arm.

## Behavioral interpretation

The frozen three-direction controller produced sustained movement: 32.317% of decisions emitted an action, dominated by LEFT (21.200%), followed by RIGHT (9.833%) and sparse UP (1.283%). No primary decision emitted DOWN.

Live feedback changed repetition more clearly than it changed visual-state count. Mean stuck fraction fell from 0.3320 under frozen vision and 0.5487 under no vision to 0.1523 live. These paired effects were significant (p=0.0047 and p=0.0001). Live coarse-state diversity was lower than frozen by 7.4 states (p=0.0084) and indistinguishable from no vision (effect -0.25, p=0.9418). The defensible conclusion is trajectory modulation and reduced repetition, not richer state coverage.

Five preregistered visual-only large-transition events occurred in live trials at seed 601 decision 286, seed 617 decision 118, and seed 618 decisions 233, 234, and 247. These are coarse-frame events and are not semantic claims that Red left the bedroom.

The 1,000-decision showcase emitted 250 actions, visited 32 coarse states, recorded one large visual transition, and had a longest identical-state interval of 96 decisions. The secondary MDN-enabled controller visited fewer coarse states on average (19.00 versus 22.65) despite adding DOWN. This does not validate MDN as a natural visual channel.

## Performance

Mean primary decision timing was 0.656 ms visual encoding, 67.381 ms MaleCNS simulation and aggregation, 0.026 ms motor decoding, 1.252 ms PyBoy update, and 73.586 ms total. Primary trials simulated 3,600 neural seconds in 1,325.3 wall seconds, averaging 15.89 decisions/s.

## Scope

Experiment 10 passes because live visual feedback significantly changed trial-level repetition while the fixed controller sustained autonomous actions without hidden policies. Category C is appropriate because behavior remained repetitive and visual-state coverage did not improve relative to controls. No navigation success, room identity, coordinates, or game state was measured.
