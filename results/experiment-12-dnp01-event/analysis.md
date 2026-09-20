# Experiment 12 analysis

```text
Experiment 12 verdict:
WEAK

Base commit:
2462e286de465c2fa5db4b061b5fbf07585ef26f

Question:
Can DNp01's significant continuous visual modulation be converted into a reliable discrete interaction event?

Calibration seeds: 801-820
Held-out seeds: 821-840
Integrated seeds: not run; held-out A validation did not pass

DNp01 cells:
10001 / R; 10010 / L

EXPERIMENT 11 REFERENCE

Decoder:
2.0 SD positive rising edge

Live A rate:
1.4167%

No-vision A rate:
1.9167%

Effect:
-0.500 percentage points

p-value:
0.148438

EXPERIMENT 12 CALIBRATION

Decoder families tested:
rising_edge, level, recovery, derivative, cumulative

Selected decoder: derivative
Aggregation: summed bilateral DNp01
Window: 3 decisions
Threshold: 1.500 z
Hysteresis: 0.5 z
Refractory: 5 decisions

Selection objective:
0.013583

HELD-OUT DNp01 SIGNAL

Live mean delta: 0.141667 Hz
No-vision mean delta: 0.004167 Hz
Effect: 0.137500 Hz
p-value: 0.002014160

Does continuous modulation replicate?
YES

HELD-OUT EVENTS

LIVE:
event rate: 1.3333%

NO VISION:
event rate: 1.2500%

FROZEN:
event rate: 1.4167%

SHUFFLED:
event rate: 1.0833%

Live-vs-no-vision event effect: 0.0833 percentage points
p-value: 1.000000

Live-vs-frozen effect: -0.0833 percentage points
p-value: 1.000000

Live-vs-shuffled effect: 0.2500 percentage points
p-value: 0.453125

Events per trial: 0.8000
A decision percentage: 1.3333%
Median inter-event interval: 24.0
Mean inter-event interval: 25.0
Minimum inter-event interval: 8
Longest event burst: 1
Trials with >=1 A: 65.00%
Trials with zero A: 35.00%
Refractory suppressions: 0

Ablation:
result: A events = 0

Temporal-shuffle diagnostic:
result: live-vs-no event effect 1.4167 percentage points, p=0.005859

Event-triggered visual-change diagnostic:
result: event-aligned {'-2': 0.00010036892361111111, '-1': 0.00011393229166666667, '0': 0.0002875434027777778, '1': 0.23226851851851857, '2': 0.0007870370370370373}; matched non-events {'-2': 0.0, '-1': 0.0, '0': 0.0, '1': 0.0, '2': 0.0}

A CHANNEL VERDICT:
WEAK

INTEGRATED CONTROLLER

Enabled controls:
LEFT: validated, integration not run
RIGHT: validated, integration not run
UP: validated, integration not run
DOWN: disabled
A: disabled
NONE: available

Seeds: none
Horizon: not run

LEFT %: N/A
RIGHT %: N/A
UP %: N/A
A %: N/A
NONE %: N/A

A events: 0 integrated events
Coarse visual transitions: N/A
Transitions shortly after A: N/A
Notable human-reviewed interactions: none; integration was not run

Leakage checks:
RAM: PASS
player coordinates: PASS
screen classification into controller: PASS
dialogue detector: PASS
pixel statistics into A decoder: PASS
scripted A timing: PASS
random actions: PASS
trained policy: PASS
reward: PASS

Primary result:
B

Was Experiment 11's A failure primarily a decoder mismatch?
NO

Is A now a scientifically defensible Flymon control?
NO

Usable controls:
LEFT, RIGHT, UP
```

Temporal shuffling increased rather than abolished event separation, so the original temporal ordering is not a reliable event code. The large frame change at t+1 follows the A press and is therefore an action consequence, not controller input. DNp01 continuous visual modulation replicated, but the frozen derivative decoder did not produce held-out event-rate separation. The result supports category B: the neural modulation remains insufficient for a reliable discrete Game Boy event under the transparent decoder family tested.
