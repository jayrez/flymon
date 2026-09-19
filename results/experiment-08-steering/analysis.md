# Experiment 8 analysis

```text
Experiment 8 verdict:
PASS

Calibration seeds:
401-420
Primary test seeds:
421-440

Baseline method:
10 no-vision windows (2.0 simulated s), per-trial frozen DNa02 L/R means and L-R variance
Threshold method:
inner selection among 1.0/1.5/2.0 x max(per-trial baseline SD, calibration pooled SD floor)
Selected threshold:
2.0 SD; pooled floor 1.6299 Hz

CONTROLLED LATERAL ASSAY

Left-biased stimulus:
mean steering signal: 3.0333 Hz
LEFT/RIGHT ratio: 27.1333

Right-biased stimulus:
mean steering signal: -1.6583 Hz
LEFT/RIGHT ratio: 0.1986

lateral assay p-value: 0.001953

NATURAL BEDROOM - LIVE

LEFT %: 32.167
RIGHT %: 2.667
NO ACTION %: 65.167
mean steering signal: 2.6083 Hz
steering SD: 3.2662 Hz
threshold-crossing rate: 0.5558

NO VISION

LEFT %: 9.500
RIGHT %: 1.917
NO ACTION %: 88.583
mean steering signal: 0.3875 Hz

FROZEN FRAME
action distribution: {"LEFT": 0.21083333333333334, "NONE": 0.7466666666666667, "RIGHT": 0.0425}
mean steering signal: 1.1083 Hz

SHUFFLED VISION
action distribution: {"LEFT": 0.18666666666666668, "NONE": 0.745, "RIGHT": 0.06833333333333333}
mean steering signal: 0.7667 Hz

MIRRORED VISION
action distribution: {"LEFT": 0.14, "NONE": 0.7608333333333334, "RIGHT": 0.09916666666666667}
mean steering signal: 0.3667 Hz

Live vs no-vision:
effect size: action TV 0.234167; steering difference 2.2208 Hz
p-value: actions 0.0002; steering 0.0001

Live vs frozen:
effect size: action TV 0.110833; steering difference 1.5000 Hz
p-value: actions 0.0001; steering 0.0006

Normal vs mirrored:
effect size: action TV 0.181667; steering difference 2.2417 Hz
p-value: actions 0.0002; steering 0.0010

Experiment 7 action rate: 0.041250
Experiment 8 action rate: 0.348333

Experiment 7 live-vs-no-vision effect: TV 0.017500
Experiment 8 live-vs-no-vision effect: TV 0.234167

Dynamic Pokemon condition:
condition: title animation
action rate: 0.135000
vision effect: action TV 0.011667, p=0.4472; steering p=0.7549

Primary result:
E

Controller leakage checks:
pixels directly into controller: PASS
RAM: PASS
class labels: PASS
trained policy: PASS
scripted route: PASS

Did baseline-relative DNa02 decoding convert the controlled visual steering effect into significant natural Pokemon visual-motor behavior?
YES
```

## Interpretation

Natural vision robustly modulated steering and materially changed feedback.

This is an engineered BCI and does not establish natural Pokemon motor semantics or semantic understanding.

## Calibration constraint and validation

No candidate satisfied the preregistered neutral/no-vision action ceiling of 10%. The fallback rule therefore selected 2.0 SD, the quietest candidate. Its selection neutral/no-vision action fraction was 16.33%, and its held-out validation fraction was 20.67%. This tonic false-action rate is a material limitation, although matched natural-vision effects were much larger.

Held-out calibration retained the directional response: left-biased input produced +3.0333 Hz mean steering and a 27.13 LEFT/RIGHT ratio; right-biased input produced -1.6583 Hz and a 0.199 ratio. The exact ten-seed paired sign test gave p=0.001953.

## Natural vision and interventions

Live bedroom vision changed both thresholded actions and the continuous signal relative to no vision. Horizontal mirroring did not produce a full sign reversal of the population mean, but it substantially reduced the live positive bias (2.6083 to 0.3667 Hz), increased RIGHT relative to LEFT, and differed significantly in actions and continuous steering. The encoder's diagnostic left-right activity correlated 0.510 with DNa02 steering; this diagnostic never entered the controller.

Swapping only the DNa02 controller mapping changed the matched ten-seed action distribution from LEFT/RIGHT/NONE = 0.3767/0.0300/0.5933 to 0.1250/0.1617/0.7133 (TV 0.2517). This establishes BCI mapping causality separately from visual neural causality.

Live versus frozen feedback improved substantially over Experiment 7: action TV rose from 0.0050 to 0.1108, 46.10% of matched post-action frame hashes diverged, mean absolute steering divergence was 2.4191 Hz, and actions disagreed at 35.08% of later decisions.

## Dynamic-condition and scope limits

The title animation did not show significant visual dependence (action TV 0.0117, p=0.4472; continuous p=0.7549). Greater raw screen motion therefore did not guarantee useful lateral DNa02 drive. The positive primary result is specific to the canonical bedroom frames, the declared Experiment 3 projection, this baseline procedure, and these seeds.

The bedroom results identify Experiment 7's absolute decoding method as the main tested limitation in that environment. They do not show that every natural Pokémon sequence supplies directional drive, and they do not establish navigation, semantic understanding, or a natural motor circuit.
