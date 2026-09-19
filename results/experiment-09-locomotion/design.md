# Experiment 9 — preregistered baseline-relative locomotion

Written before Experiment 9 neural outcomes. Experiments 1–8 remain unchanged.

## Populations and controller

Local MaleCNS metadata must resolve DNg100, MDN, and diagnostic DNp01 before use.
DNg100 and MDN activity is the sum of spikes across each population divided by
the 200 ms window; summation treats the identified population as one command
channel and preserves the extra MDN cell count rather than silently averaging it
away. Per-neuron rates are also retained. The primary controller receives only
these two summed rates and a frozen baseline. It emits UP, DOWN, or NONE. If both
cross, the larger baseline-normalized z score wins; exact ties favor UP. One
decision cooldown follows an action.

Each trial begins with ten no-injection 10-step windows: 2.0 simulated seconds.
The population means, sample variances, and per-neuron means are frozen. Active
decisions use 10 neural steps, 8 PyBoy hold frames, and 4 release frames.

## Calibration and threshold selection

Seeds 501–520. Each declared condition has 20 active windows after its baseline:
no vision; uniform; left-biased; right-biased; vertical motion (a horizontal dark
bar oscillating vertically); horizontal motion (a vertical bar oscillating
horizontally); centered looming square; centered receding square; canonical
bedroom frame; and the fixed menu frame. All synthetic sequences are defined in
source before outcomes.

Seeds 501–510 select independently among [1.0, 1.5, 2.0] SD for each population;
511–520 validate. Threshold = multiplier × max(per-trial baseline SD, pooled
selection-set within-baseline residual SD floor), with a 0.5 Hz minimum floor.
For each population, maximize mean crossing fraction across all nine declared
visual conditions minus no-vision crossing, subject to no-vision crossing ≤10%;
ties choose the larger multiplier. If none qualify, choose the lowest no-vision
crossing, then greatest objective, then larger multiplier. No condition is chosen
post hoc as preferred. Threshold files are frozen before seeds 521–540.

Validation reports every condition. A population is calibration useful only when
the preregistered aggregate visual-versus-no-vision paired seed mean delta is
positive with a two-sided paired sign-permutation p<0.05.

## Primary natural trials

Seeds 521–540, 60 decisions. Bedroom controls: live, no vision, frozen initial
frame, and deterministic fixed spatial shuffle per seed. A separately booted title
animation supplies the natural dynamic live/no-vision comparison. DNg100 and MDN
BCI-input ablations use seeds 521–530 under live bedroom vision. They do not alter
MaleCNS. Primary matched comparisons are live/no, live/frozen, and live/shuffled.
Actions use total variation with 9,999 seed-stratified label swaps. Each continuous
population delta uses the absolute paired difference of seed means with 9,999
paired sign flips. Tests were fixed before primary results.

## Exploratory integration

Only after primary records exist, seeds 521–525 run the frozen Experiment 8 DNa02
baseline-relative channel together with the frozen Experiment 9 channels. Signals
compete as normalized threshold excesses; the largest wins with fixed tie order
LEFT, RIGHT, UP, DOWN. Exactly one action or NONE is emitted. This run cannot alter
the Experiment 9 verdict and has no navigation score.

PASS requires at least one population with significant positive natural visual
delta, significant action-distribution dependence, and successful corresponding
ablation. Both populations meeting these rules is a stronger PASS. Controlled-only
effects are WEAK. Neither population differing from no vision is FAIL.
