# Experiment 13 analysis

```text
Experiment 13 verdict:
PASS

Base commit:
4699f66201fadcf337a409f253a9846a228e7975

Question:
Can a frozen low-dimensional population-level DN signal support a reliable visually driven discrete A interaction channel?

Calibration seeds: 861-880
Held-out seeds: 881-900
Integrated seeds: 901-920

FROZEN DN POPULATION

Source experiment: Experiment 4 informative_dn_top20, frozen and reused by Experiment 6
Population size: 20
Cell types: DNa16, DNae001, DNae002, DNb01, DNbe006, DNg111, DNg13, DNg40, DNge103, DNp01, DNp04, DNp13, DNp54, DNp71, DNpe020, DNpe056
Population IDs: 521262, 11706, 11222, 10654, 519771, 531898, 10361, 512006, 10759, 11002, 10223, 11505, 11563, 13615, 10010, 10388, 512071, 12885, 10151, 10543
Selection rule: exact persisted E4/E6 P20 order; no Experiment 13 reranking

SIGNALS TESTED
mean, norm, change, window_change (3 and 5 decisions)

Selected signal: population RMS z norm
Selected event decoder: rising edge
Window: 1 decision
Threshold: 1.5 RMS-z units
Hysteresis: 0.5
Refractory: 5 decisions
Calibration objective: 0.064432489

HELD-OUT POPULATION SIGNAL

LIVE: 1.125880901
NO VISION: 0.899616036
FROZEN: 1.151979616
SHUFFLED: 1.159292896
Live-vs-no-vision effect: 0.226264866
p-value: 0.000001907
Live-vs-frozen effect: -0.026098714
p-value: 0.000915527
Live-vs-shuffled effect: -0.033411994
p-value: 0.001554489

HELD-OUT A EVENTS

LIVE: 1.8333%
NO VISION: 0.0833%
FROZEN: 2.1667%
SHUFFLED: 3.4167%
Live-vs-no-vision effect: 1.7500 percentage points
p-value: 0.000122070
Live-vs-frozen effect: -0.3333 percentage points
p-value: 0.359375000
Live-vs-shuffled effect: -1.5833 percentage points
p-value: 0.000396729

A events/trial: 1.1000
A decision percentage: 1.8333%
Median inter-event interval: 24.0 decisions
Minimum inter-event interval: 6 decisions
Longest event run: 1
Refractory suppressions: 1
Trials with >=1 A: 75.00%
Trials with zero A: 25.00%

DNp01 Experiment 12 reference on same seeds:
event effect: 0.2500 percentage points
p-value: 0.711975098

Population ablation:
result: A events = 0

Cell permutation:
result: live-vs-no effect 2.5000 percentage points, p=0.000061035; separation was preserved

Temporal shuffle:
result: live-vs-no effect 2.5833 percentage points, p=0.000061035; separation increased

Subpopulation stability:
result: ranks 1-10 effect 4.9167 points (p=0.000001907); ranks 11-20 effect 2.7500 points (p=0.000007629)

Does temporal structure contribute?
NO

Does distributed cell identity contribute?
NO

A CHANNEL VERDICT:
PASS

INTEGRATED CONTROLLER

Run?
YES

Enabled:
LEFT: DNa02 baseline-relative steering
RIGHT: DNa02 baseline-relative steering
UP: DNg100 baseline-relative locomotion
DOWN: disabled
A: P20 RMS-norm rising event
NONE: deadband

Seeds: 901-920
Horizon: 500

LEFT %: 30.0000%
RIGHT %: 3.5400%
UP %: 0.8100%
A %: 3.8600%
NONE %: 61.7900%

A events: 386
Coarse visual transitions: 6704
Transitions shortly after A: 162 events within 1 decision; 364 events within 2-5 decisions
Notable human-reviewed events: none claimed; no semantic interaction inference was used

Leakage checks:
RAM: PASS
player coordinates: PASS
map state: PASS
screen classifier into controller: PASS
dialogue detector: PASS
pixel statistics into A decoder: PASS
scripted A: PASS
random actions: PASS
trained policy: PASS
reward: PASS

Primary result:
D

Did population-level decoding succeed where DNp01 failed?
YES

Is A now a scientifically defensible Flymon control?
YES

Usable controls:
LEFT, RIGHT, UP, A

Is Flymon ready for a separate autonomous progression experiment?
YES
```

## Interpretation

The preregistered held-out gate passed. The frozen historical P20 RMS signal was higher under live bedroom vision than under no visual injection, and its sparse rising-edge output separated live from no vision. The same-seed Experiment 12 DNp01 decoder did not separate those conditions, and removing DNp01 from P20 retained a significant effect. Both historical-rank halves also retained the effect.

The evidence supports a fixed, visually driven population A pulse, with important limits. Frozen and shuffled vision produced event rates equal to or above live vision. Temporal shuffling and cell-identity permutation preserved or increased separation. The selected one-window RMS magnitude therefore reports aggregate visual drive rather than a temporally ordered, identity-specific interaction code. Secondary natural contexts were heterogeneous: intro produced events in 80% of trials, title in 10%, and dialogue/menu in 0%. No semantic interaction understanding or successful game interaction is claimed.

The integrated run used event-first arbitration without changing DNa02 or DNg100 thresholds. Visual-transition counts are observational only and were never controller inputs. A transition after A is not treated as proof of an interaction because ordinary movement and animation also create many visual changes.
