# Experiment 9 analysis

```text
Experiment 9 verdict:
PASS

Calibration seeds:
501-520
Primary test seeds:
521-540

DNg100 population:
cells: 2
baseline: 0.0500 Hz
threshold: 2.0 SD, floor 0.5000 Hz
natural vision effect: 0.0542 Hz
p-value: 0.0013

MDN population:
cells: 4
baseline: 1.0000 Hz
threshold: 2.0 SD, floor 2.4660 Hz
natural vision effect: 0.0792 Hz
p-value: 0.1320

LOCOMOTION-ONLY LIVE

UP %: 1.917
DOWN %: 4.167
NO ACTION %: 93.917
actions/trial: 3.650

NO VISION

UP %: 0.833
DOWN %: 3.917
NO ACTION %: 95.250

FROZEN

UP %: 1.667
DOWN %: 3.833
NO ACTION %: 94.500

SHUFFLED

UP %: 1.667
DOWN %: 2.833
NO ACTION %: 95.500

DNg100 live-vs-no-vision:
effect: 0.0542 Hz
p-value: 0.0013

MDN live-vs-no-vision:
effect: 0.0792 Hz
p-value: 0.1320

DNg100 ablation:
result: UP 2.667% -> 0.000%

MDN ablation:
result: DOWN 6.667% -> 0.000%

Experiment 7 DNg100 action rate: 1.750%
Experiment 9 DNg100 action rate: 1.917%

Experiment 7 MDN action rate: 2.125%
Experiment 9 MDN action rate: 4.167%

Exploratory four-direction controller:
LEFT %: 15.000
RIGHT %: 4.000
UP %: 0.667
DOWN %: 5.000
NONE %: 75.333

Primary result:
C

Leakage checks:
RAM: PASS
framebuffer directly into controller: PASS
class labels: PASS
player position: PASS
scripted route: PASS
trained policy: PASS
reward: PASS

Is Flymon ready for unrestricted four-direction free exploration in Experiment 10?
PARTIALLY
```

## Interpretation

One locomotion channel generalized to natural Pokemon vision.

The integrated run is exploratory and did not determine the verdict. No navigation outcome was scored.

## Interpretation details

Controlled-stimulus validation did not pass for either population. DNg100's visual-minus-no-vision effect was +0.0556 Hz (p=0.15625); MDN's was -0.0222 Hz (p=0.79297). Both selectors chose 2.0 SD.

In the independent natural test, DNg100 showed a small but consistent +0.0542 Hz live-minus-no-vision effect (p=0.0013). Its crossing rate rose from 0.833% to 1.917%, and the matched action distribution changed by TV=0.01333 (p=0.0330). MDN increased by +0.0792 Hz, but inconsistently across seeds (p=0.1320); it does not qualify as a visually useful DOWN channel.

Live versus frozen behavior was not significant (action TV=0.00583, p=0.2589), so the DNg100 result reflects the presence of bedroom visual drive more than changing closed-loop feedback. The title animation also failed to improve locomotion. BCI ablations eliminated their declared channels exactly.

The exploratory controller emitted every direction, but DOWN lacks validated natural visual dependence. Experiment 9 is therefore a minimal PASS through the DNg100 UP channel. Unrestricted four-direction exploration is not yet fully supported; any next integration should label DOWN as exploratory and tonic/noise dominated unless a separately justified channel replaces it.
