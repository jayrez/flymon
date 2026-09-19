# Experiment 11 analysis

```text
Experiment 11 verdict:
WEAK

Base commit:
88c54d7c14655c68f786942ffbad1fc9a4d050e9

Architecture reused:
flymon.motor.run_neural_window/OnlineE3Encoder; flymon.steering and flymon.locomotion baselines; flymon.exploration.FrozenExplorationController; PokemonEmulator

Architecture added/refactored:
resolve_motor_populations; FrozenPopulationBaseline; BaselinePopulationDecoder; BaselineEventController; FrozenInterfaceController

Validation/tests:
15 deterministic tests PASS before simulation

DOWN

Candidates:
MDN, DNp01, DNa05, DNa07, DNb02, DNd02
Selected candidate: DNp01
Cell count: 2
Calibration seeds: 701-720
Held-out seeds: 721-740

Baseline: frozen 10 no-vision windows per trial
Threshold: 1.0 SD, floor 1.1625 Hz

Live neural effect: -0.079167 Hz
No-vision neural effect: -0.108333 Hz
Effect size: 0.029167 Hz
p-value: 0.203125

Live DOWN rate: 0.012500
No-vision DOWN rate: 0.008333
Frozen DOWN rate: 0.011667
Shuffled DOWN rate: 0.008333

MDN reference result: neural effect 0.012500 Hz (p=0.885223); action effect 0.000833 (p=1.000000)

Ablation result: DOWN actions = 0

DOWN verdict:
FAIL

A INTERACTION

Candidates:
DNp01, DNp09, DNp10, DNp07, DNp02, MDN
Selected candidate: DNp01
Cell count: 2
Calibration seeds: 741-760
Held-out seeds: 761-780

Baseline: frozen 10 no-vision windows per trial
Threshold: 2.0 SD, floor 1.1626 Hz
Refractory: 5 decisions, rising edge required
A hold/release timing: 8 / 4 PyBoy frames

Live A rate: 0.014167
No-vision A rate: 0.019167
Frozen A rate: 0.013333
Shuffled A rate: 0.010833

Effect size: -0.005000
p-value: 0.148438

Median inter-A interval: 14.0
A decision percentage: 1.4167%
Refractory suppressions: 0

Ablation result: A events = 0

A verdict:
WEAK

INTEGRATED CONTROLLER

Enabled:
LEFT: yes (Experiment 8 DNa02)
RIGHT: yes (Experiment 8 DNa02)
UP: yes (Experiment 9 DNg100)
DOWN: no
A: no
NONE: yes

Seeds: 781-800
Horizon: 500

LEFT %: 21.380
RIGHT %: 7.530
UP %: 1.330
DOWN %: 0.000
A %: 0.000
NONE %: 69.760

Coarse visual transitions: 14
Interaction-like transitions: 0
Notable human-reviewed events: none claimed

Leakage checks:
RAM: PASS
player coordinates: PASS
map/room state: PASS
screen class into controller: PASS
dialogue detector: PASS
stuck recovery: PASS
random action injection: PASS
scripted route: PASS
trained policy: PASS
reward: PASS

Primary result:
A

Usable Game Boy controls after Experiment 11:
LEFT, RIGHT, UP

Is Flymon ready for the first autonomous Pokemon progression experiment?
NO
```

DNp01 showed held-out continuous visual modulation for the event question, but thresholded A events did not separate live vision from no vision. Neither candidate channel was enabled.
