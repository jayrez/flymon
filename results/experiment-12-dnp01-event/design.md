# Experiment 12 - preregistered DNp01 interaction decoder

Written after the historical Experiment 11 diagnostic and before acquiring Experiment 12 seeds. Experiments 1-11 remain unchanged.

## Diagnostic basis

Experiment 11 live DNp01 was less suppressed than no vision (+0.091667 Hz, p=0.011475), although both condition means were below the frozen baseline. Positive excursions were rare (2.42% live, 2.00% no vision); live showed greater persistence (lag-1 autocorrelation 0.326 versus -0.033) and longer positive runs (maximum 3 versus 1). Three-window means retained the level shift. Derivative activity was sparse without a dominant transient signature. The preregistered family therefore tests level, recovery, derivative, and cumulative interpretations alongside the unchanged Experiment 11 rising-edge reference.

## Frozen neural population and aggregation

DNp01 is the only primary event population. Its two metadata-resolved cells are recorded independently as spike counts, rates, and baseline-relative deltas. Sum, mean, max, and right-minus-left are saved as diagnostics. The primary decoder aggregation is frozen as the summed bilateral rate because the summed signal was the independently held-out Experiment 11 result. No aggregation is selected on held-out data.

## Seeds and cadence

Calibration seeds: 801-820. Held-out seeds: 821-840. Conditional integrated seeds: 841-860. These ranges were checked against all earlier experimental CNS seed declarations and are unused. Each trial uses ten no-injection baseline windows, ten neural steps (200 ms) per decision, 8 PyBoy hold frames, and 4 release frames. Calibration has 60 natural windows and 20 controlled windows per condition. Held-out trials have 60 decisions. Integration, only after PASS, has 500 decisions.

## Five decoder families and fixed grids

All strategies use summed DNp01 and a five-decision refractory. The Experiment 11 reference rearms on any below-threshold sample exactly as before. The other families use 0.5 z hysteresis and rising threshold crossings to avoid sustained-signal spam. Cumulative history clears after each emitted event.

1. rising_edge reference: window 1; threshold z in [1.0, 1.5, 2.0].
2. level: window in [1, 3, 5]; threshold z in [-0.5, 0.0, 0.5].
3. recovery: window in [3, 5]; threshold z in [0.5, 1.0, 1.5]; evidence is current window mean minus the lowest preceding z in retained history.
4. derivative: window in [1, 3]; threshold z in [0.5, 1.0, 1.5]; evidence is recent-window mean minus preceding-window mean.
5. cumulative: window in [3, 5]; threshold z in [0.0, 0.5, 1.0]; evidence is window z sum divided by sqrt(window).

The scale is max(per-trial summed baseline SD, the pooled within-baseline calibration residual SD floor, 0.5 Hz). The baseline and scale freeze before each active trial. Reset clears all history, arming, and refractory state.

## Calibration stimuli and selection

Controlled conditions are no vision, uniform, luminance onset, luminance offset, looming onset, left onset, and right onset. Natural conditions are live bedroom, frozen bedroom, and deterministically shuffled bedroom. Existing captured menu, dialogue, title, and intro frames are secondary diagnostics and do not enter selection.

Every grid point is replayed over calibration neural records. For each seed, compute bedroom-live event rate, no-vision event rate, controlled-onset event rate, refractory suppressions, and maximum consecutive event burst. The fixed objective is:

live rate - no-vision rate + 0.25 * (controlled rate - no-vision rate) - 2 * max(0, no-vision rate - 0.05) - max(0, live rate - 0.10) - 0.01 * suppression fraction

Eligible configurations require no-vision rate <=5%, live rate <=10%, and maximum event burst <=1. Select maximum objective; ties prefer fewer total parameters, family order listed above, smaller window, then higher threshold. Pokemon progression and screen outcomes never enter selection.

## Held-out validation and statistics

The selected configuration freezes before seeds 821-840. Each seed runs live, no vision, frozen, and deterministic shuffled bedroom conditions. Primary neural and event comparisons are live minus no vision. Secondary comparisons are live minus frozen and live minus shuffled. Each statistic is the paired seed mean difference with an exact two-sided sign permutation test over 20 seeds.

PASS requires replicated positive continuous DNp01 modulation at p<0.05, positive live/no event-rate separation at p<0.05, live A <=10%, maximum event burst 1, and BCI ablation producing zero A. WEAK requires replicated neural modulation with marginal or inconsistent events. Otherwise FAIL.

Temporal shuffle replays each held-out trial after a deterministic within-trial permutation of DNp01 windows. Cell-drop replays zero each cell separately. These are diagnostics only. Visual transition magnitudes at t-2 through t+2 are post-hoc analysis and never enter the controller.

## Conditional integration

Only PASS enables A in FrozenInterfaceController. LEFT/RIGHT remain frozen Experiment 8 DNa02, UP remains frozen Experiment 9 DNg100, and DOWN stays disabled. A has event-first arbitration. Failed or WEAK A produces an empty, disabled integrated result rather than an autonomous run.
