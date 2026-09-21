# Experiment 13 — preregistered frozen population event decoding

Written before generating any Experiment 13 neural data. Experiments 1–12 remain unchanged.

## Frozen population and provenance

The repository does not persist an exact historical P50 or P100 identity list. It does persist one fixed population: the 20 entries in Experiment 4 `informative_dn_top20`, transferred unchanged as Experiment 6 `frozen_experiment4_ids` and scoring 87.50% on unseen instances and CNS seeds. Experiment 13 therefore uses this exact ordered **E4/E6 P20**. It is not reranked. `population.json` records historical rank, Experiment 4 DN slot, verified MaleCNS brain index/body ID, type, side, selection frequency, and source for every cell. One member is DNp01-L (rank 15); held-out DNp01 removal is a required diagnostic. No complete historical per-cell sign/effect vector was persisted, so historical-sign projection is omitted.

## Seeds, cadence, and baseline

Calibration CNS seeds: **861–880**. Held-out CNS seeds: **881–900**. Conditional integrated seeds: **901–920**. Repository search found no prior use of these blocks as CNS seeds; 910 and 920 appeared only as offline permutation RNG seeds in an Experiment 9 analyzer. Each trial resets MaleCNS, runs 10 no-injection baseline windows, and freezes its per-cell baseline. Each window is 10 neural steps / 200 ms. A uses 8 PyBoy hold frames and 4 release frames. Natural held-out trials have 60 decisions; conditional integration has 500.

For cell i, z_i(t) = (rate_i(t) - frozen_mean_i) / max(frozen_SD_i, floor). The common floor is max(0.5 Hz, pooled within-trial baseline residual SD) computed only from calibration baselines.

## Calibration conditions

Controlled: no vision, uniform, abrupt luminance onset, abrupt luminance offset, loom onset, left-biased onset, right-biased onset. Natural primary: live bedroom, no-vision bedroom, frozen bedroom, deterministic spatially shuffled bedroom. Existing Experiment 3 title, intro, New Game menu, and Oak dialogue frame sequences are secondary diagnostics and never enter decoder selection.

## Transparent population signals

Four preregistered signal families:

1. `mean`: mean z across P20.
2. `norm`: RMS z across P20.
3. `change`: RMS cellwise z(t)-z(t-1).
4. `window_change`: RMS difference between adjacent 3- or 5-decision population mean vectors.

No PCA, classifier, fitted weight vector, supervised direction, or Experiment 13 label-derived cell selection is used.

## Decoder grid

The fixed base candidates are:

- mean + level, one-decision evidence
- mean + cumulative evidence, 3 decisions
- mean + cumulative evidence, 5 decisions
- norm + rising edge
- change + rising edge
- window_change + rising edge, 3 decisions
- window_change + rising edge, 5 decisions

Each uses threshold {1.0, 1.5, 2.0, 2.5}, 0.5 signal-unit hysteresis where applicable, and a five-decision refractory. Total grid size: **28**, with no continuous optimization.

## Frozen calibration objective

For each configuration compute trial-level live/no-vision event effects, controlled-onset gain, fraction of controlled onset conditions with an event, fraction of seeds with positive live/no effect, seed-effect SD, no-vision rate, live tonic rate, refractory suppression fraction, and longest event run.

Objective:

`mean(live-no) + 0.25*mean(controlled-no) + 0.01*controlled_consistency + 0.01*positive_seed_fraction - 0.25*SD(live-no) - 2*max(0,no_rate-0.05) - max(0,live_rate-0.10) - 0.01*suppression_fraction`

Eligible configurations require no-vision rate <=5%, live rate <=10%, and maximum consecutive event run <=1. Maximize objective; ties prefer the listed candidate order, then the higher threshold. Pokémon outcomes, frame changes, and secondary natural contexts never enter selection.

## Held-out inference and controls

The frozen configuration runs on seeds 881–900 under live, no vision, frozen, and shuffled bedroom conditions. The independent unit is one seed/trial. Continuous decoder evidence and A-event rates use exact paired two-sided sign-permutation tests for live minus each control.

Offline controls reset the decoder for each trial:

- full population ablation
- independently permuted cell identity in each decision window
- temporal window-order shuffle
- historical ranks 1–10 and 11–20
- leave-one-multi-cell-type-out, including explicit removal of rank-15 DNp01
- frozen Experiment 12 DNp01 derivative decoder on the same neural trajectories

The primary population is never retuned from these diagnostics.

## PASS and integration rule

PASS requires all of: positive live/no population-evidence modulation at p<0.05; positive live/no A-event separation at p<0.05; live A <=10%; no-vision A <=5%; maximum event run <=1; ablation A=0; and the frozen decoder after DNp01 removal retains positive significant live/no event separation. WEAK requires significant population modulation with marginal, inconsistent, or environment-specific discrete output. Otherwise FAIL.

Only PASS enables A in the existing `FrozenInterfaceController` with frozen DNa02 LEFT/RIGHT and DNg100 UP. DOWN remains disabled. A keeps event-first arbitration. WEAK/FAIL writes an explicitly disabled integrated artifact and runs no progression or autonomous integration.
