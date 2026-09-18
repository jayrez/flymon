# Experiment 4 — temporal DN readout versus informative DN subsets

**Verdict: FAIL for the temporal-readout hypothesis. Primary finding: C — screen identity survives in a small DN subpopulation and is masked by population-wide noise.** The prespecified all-DN nested bin readout changed five-screen accuracy from **40.0%** aggregate to **41.7%**. In contrast, training-selected DNs using **aggregate counts** reached **98.3%**, versus **90.0%** for nested temporal bins with selected DNs. Chance is **20%**.

## Experiment 3 classifier verification and seed isolation

Experiment 3 used leave-one-**seed**-out Euclidean nearest centroid: each fold trained on all five conditions from 11 seeds and tested all five conditions of the twelfth. No seed was shared between training and test. We independently reproduced the original **40.0% (24/60)** and its exact confusion matrix from the saved Experiment 3 DN vectors. The Experiment 4 replay also reproduced that aggregate vector and classifier exactly. The original 40% was **not seed leakage**. The four-screen aggregate result excluding Title was **45.8%** versus 25% chance.

## Controlled replay and DN recording

Experiment 3 saved only 200-step per-DN totals, so we replayed its exact saved 460-element injection vectors. The five visual conditions, five static controls, no-vision baseline, canonical bedroom sequence, encoder, 460 projection cells, 1,314 DN cells, 12 seeds (`101..112`), reset procedure and 200 simulation steps were unchanged. No ROM or Pokémon RAM was accessed during Experiment 4. [DN bins](dn-bins-5.npz) contains compressed **11 conditions × 12 seeds × 40 five-step bins × 1,314 DNs**. The 10-, 20-, 50- and 200-step representations are exact sums, with no raw spike dumps.

Every trial's complete per-DN aggregate was required to match Experiment 3. Intermittent CUDA replay divergence required **5 extra attempts in 1 of 132 trials**; the fixed eight-attempt limit and attempts are recorded in [capture metrics](capture-metrics.json). In repeated checks, matching aggregate runs had the same entire five-step tensor hash. This conditional replay is a limitation, and no sensory or readout parameter was changed to make a run match.

## Raw DN separability: aggregate versus time bins

A representation is the chronological raw firing count for each DN in each bin, flattened for Euclidean nearest centroid; no test-derived normalization is used. Within is the average of each screen's median **different-seed** distance. Between is the average of ten screens' median cross-seed distance. Same-seed/different-screen distances hold the stochastic seed constant and are listed separately. Cosines compare condition mean vectors. Absolute Euclidean scales differ across dimensions, so ratios are compared within each row.

| Readout | Dimensions | Within | Between | Between/within | Mean cosine | Same-seed different-screen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggregate | 1,314 × 1 | 177.0 | 169.3 | 0.957 | 0.99843 | 63.5 |
| 50-step bins | 1,314 × 4 | 111.9 | 109.5 | 0.979 | 0.99733 | 57.7 |
| 20-step bins | 1,314 × 10 | 95.6 | 94.4 | 0.988 | 0.99489 | 61.9 |
| 10-step bins | 1,314 × 20 | 96.0 | 95.6 | 0.996 | 0.98783 | 72.6 |
| 5-step bins | 1,314 × 40 | 109.0 | 108.7 | 0.998 | 0.96628 | 90.9 |

None of the all-DN binned representations raises between-screen distance above different-seed within-screen distance. Per-pair Euclidean and cosine values are in [classification.json](classification.json).

## Grouped-by-seed classification

Every outer fold holds out **all conditions of one seed**. Aggregate and each fixed bin size use raw-count Euclidean nearest centroid. The prespecified nested temporal method selects a bin size using grouped inner validation on the 11 training seeds; only then is the outer seed predicted. Static classifiers use static training examples and the bin chosen on dynamic training data. Five-screen chance is 20%; four-screen chance excluding Title is 25%. Balanced accuracy equals ordinary accuracy because each class has 12 trials.

| All-DN readout | Dynamic five-screen | Static five-screen | Dynamic four-screen |
| --- | ---: | ---: | ---: |
| Aggregate | 40.0% | 41.7% | 45.8% |
| 50-step bins | 41.7% | 41.7% | 50.0% |
| 20-step bins | 40.0% | 41.7% | 47.9% |
| 10-step bins | 45.0% | 41.7% | 54.2% |
| 5-step bins | 33.3% | 38.3% | 43.8% |
| **Nested bin choice** | **41.7%** | **41.7%** | **50.0%** |
| Exploratory selected 20-step window | 53.3% | 53.3% | 54.2% |

Nested bin choice selected **{10: 9, 20: 1, 50: 2}** across 12 outer folds (modal **10 steps**). Its **41.7%** versus **40.0%** all-DN difference is only **+1.7 percentage points**. A fixed 10-step result is descriptive, since that fixed bin's test accuracy was inspected among candidates.

### All-DN aggregate confusion matrix

| Actual \ predicted | intro | title | menu | dialogue | bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| intro | 3 | 0 | 0 | 0 | 9 |
| title | 1 | 2 | 1 | 5 | 3 |
| menu | 0 | 0 | 3 | 7 | 2 |
| dialogue | 0 | 0 | 3 | 7 | 2 |
| bedroom | 3 | 0 | 0 | 0 | 9 |

### Prespecified nested temporal-bin confusion matrix

| Actual \ predicted | intro | title | menu | dialogue | bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| intro | 5 | 1 | 0 | 0 | 6 |
| title | 1 | 3 | 2 | 5 | 1 |
| menu | 0 | 3 | 3 | 4 | 2 |
| dialogue | 0 | 1 | 3 | 7 | 1 |
| bedroom | 2 | 1 | 0 | 2 | 7 |

## Time localization

The single-window scan was exploratory. Each row uses only one 20-step window and the same grouped classifier. A follow-up selector chose among all ten windows using **inner training seeds only**; the outer test seed never chose a window.

| Steps | Five-screen accuracy | Between/within |
| --- | ---: | ---: |
| 0–20 | 53.3% | 0.998 |
| 20–40 | 35.0% | 0.987 |
| 40–60 | 28.3% | 0.986 |
| 60–80 | 31.7% | 0.984 |
| 80–100 | 28.3% | 0.987 |
| 100–120 | 26.7% | 0.979 |
| 120–140 | 38.3% | 0.978 |
| 140–160 | 35.0% | 0.984 |
| 160–180 | 38.3% | 0.985 |
| 180–200 | 30.0% | 0.984 |

The inner selector chose **steps 0–20 in all 12 outer folds** (`[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`), scoring **53.3%**. The identical first frame in the static control also scored **53.3%**, so this is an **initial static visual response**, not evidence that changing game frames helps. Its +13.3-point gain over aggregate has seed-group bootstrap 95% interval **-3.3 to +30.0 points** and paired sign-flip p **0.211**. The advantage over aggregate is uncertain.

## DN population-size analysis: the main finding

Within **every training fold**, DNs were ranked by an ANOVA-like score: between-condition variation of training class means divided by within-condition residual variation, summed across bins. Top 500/250/100/50 were reselected independently for each held-out seed. Test labels and test spikes never entered the ranking. This is a screen-identity diagnostic, not an action decoder.

| Readout | All 1,314 | Top 500 | Top 250 | Top 100 | Top 50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aggregate | 40.0% | 81.7% | 90.0% | 98.3% | 98.3% |
| 50-step bins | 41.7% | 75.0% | 73.3% | 95.0% | 98.3% |
| 20-step bins | 40.0% | 61.7% | 70.0% | 83.3% | 91.7% |
| 10-step bins | 45.0% | 66.7% | 66.7% | 71.7% | 91.7% |
| 5-step bins | 33.3% | 43.3% | 40.0% | 53.3% | 56.7% |

Aggregate **population size** was also chosen by inner grouped validation for each outer fold. It chose **{50: 1, 100: 11}**, modal **100 DNs**, and achieved **98.3% (59/60)**. The same selected sizes with static training reached **100.0%**; four-screen aggregate reached **97.9%**. A separate nested bin-plus-population procedure chose modal **50 DNs** and reached **90.0%** dynamic versus **70.0%** static. Thus **time binning is unnecessary for the strongest grouped result**.

### Aggregate selected-DN confusion matrix

| Actual \ predicted | intro | title | menu | dialogue | bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| intro | 12 | 0 | 0 | 0 | 0 |
| title | 0 | 12 | 0 | 0 | 0 |
| menu | 0 | 0 | 11 | 1 | 0 |
| dialogue | 0 | 0 | 0 | 12 | 0 |
| bedroom | 0 | 0 | 0 | 0 | 12 |

### Temporal-bin selected-DN confusion matrix

| Actual \ predicted | intro | title | menu | dialogue | bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| intro | 11 | 1 | 0 | 0 | 0 |
| title | 0 | 10 | 0 | 1 | 1 |
| menu | 0 | 1 | 11 | 0 | 0 |
| dialogue | 0 | 1 | 1 | 10 | 0 |
| bedroom | 0 | 0 | 0 | 0 | 12 |

Temporal selected-DN accuracy minus aggregate selected-DN accuracy is **-8.3 points**, with seed-group bootstrap 95% interval **-21.7 to +1.7** and paired sign-flip p **0.375**. This does not support a temporal advantage. The median number of aggregate training-fold DNs with F>1 was **359**; that threshold is descriptive, not a validated cutoff.

| FlyWire ID | DN type | Side | Training folds in top 50 |
| ---: | --- | :---: | ---: |
| 521262 | DNae002 | R | 12/12 |
| 11706 | DNg111 | R | 12/12 |
| 11222 | DNg111 | L | 12/12 |
| 10759 | DNb01 | R | 12/12 |
| 10654 | DNb01 | L | 12/12 |
| 512006 | DNg13 | R | 12/12 |
| 11002 | DNae002 | L | 12/12 |
| 11563 | DNae001 | R | 12/12 |
| 531898 | DNp04 | L | 12/12 |
| 11505 | DNp54 | R | 12/12 |
| 519771 | DNp71 | L | 12/12 |
| 10249 | DNp31 | L | 12/12 |
| 10038 | pIP1 | R | 12/12 |
| 512071 | DNae001 | L | 12/12 |
| 10223 | DNpe056 | L | 12/12 |

IDs/types come from MaleCNS metadata. The list summarizes training-fold selection frequency; no IDs were hardcoded into the classifier.

## Permutation significance

Condition labels were independently shuffled **within each seed**. The aggregate and nested-bin nulls repeated their grouped folds, with inner bin choice repeated for nested bins. The exploratory window null took the maximum of ten grouped window accuracies per shuffle, correcting its observed scan conservatively. The aggregate selected-DN null **refit training-only DN ranks** and took the maximum across five population sizes; the temporal selected-DN null also took the maximum across four temporal bin sizes. There were **1,000 shuffles** for each of the first four comparisons and **500** for the temporal selected-DN search. These nulls test screen identity above shuffled labels; they do not, by themselves, test temporal improvement over aggregate.

| Comparison | Observed | Null mean | Null 95th | Empirical p |
| --- | ---: | ---: | ---: | ---: |
| All-DN aggregate | 40.0% | 20.1% | 26.7% | 0.0010 |
| All-DN nested temporal | 41.7% | 20.1% | 28.3% | 0.0010 |
| Exploratory selected window | 53.3% | 26.5% | 31.7% | 0.0010 |
| Selected-DN aggregate | 98.3% | 24.4% | 33.3% | 0.0010 |
| Selected-DN temporal | 90.0% | 29.2% | 36.7% | 0.0020 |

Full draws and shuffle descriptions are in [permutation-results.json](permutation-results.json). Both selected-DN readouts remain above conservative size-search nulls, while their paired comparison shows no time-binning gain.

## Static input and no-vision baseline

The strongest **static aggregate selected-DN** result was **100.0%**. The all-DN selected first window scored **53.3%** static and **53.3%** with the changing frame sequence. These controls do not show a benefit from Experiment 3's short temporal visual sequence. The no-vision baseline was replayed with `eye_drive=None` and no injection; its full DN totals matched Experiment 3. Baseline separation below measures visual presence, not Pokémon screen identity.

| Screen | Aggregate mean-vector distance to no vision | Matched-seed distance |
| --- | ---: | ---: |
| intro | 49.4 | 74.2 |
| title | 29.7 | 57.6 |
| new game menu | 17.4 | 53.1 |
| oak dialogue | 26.6 | 55.4 |
| bedroom | 45.9 | 73.4 |

## Performance and debug artifacts

Median MaleCNS simulation step **4.820 ms**; DN capture overhead **0.154 ms per step**. Offline binning all 132 trials took **4.1–15.4 ms per representation**. Grouped classifiers, distances and initial population analysis took **10.9 s**; the initial two 1,000-shuffle tests took **311.1 s**, window correction **68.3 s**, aggregate subset validation **20.0 s**, and temporal subset validation **597.7 s**. These are offline analysis costs, not live inference costs.

- [Classification by bin size](../../captures/experiment-04/classification-by-bin-size.svg)
- [Accuracy versus DN population size](../../captures/experiment-04/accuracy-vs-dn-population.svg)
- [Information versus simulation time](../../captures/experiment-04/information-vs-time.svg)
- [Permutation null versus observed](../../captures/experiment-04/permutation-null-vs-observed.svg)
- [All-DN confusion matrices](../../captures/experiment-04/confusion-matrices.svg)
- [Selected-DN confusion matrices](../../captures/experiment-04/subset-confusion-matrices.svg)
- [Selected-DN permutation nulls](../../captures/experiment-04/subset-permutation-null.svg)

## Conclusion and next bottleneck

**FAIL for the temporal-structure hypothesis; primary bottleneck C.** An early DN response is visible, but selecting its window yields an uncertain +13.3-point advantage over aggregate all-DN counts. The much stronger result is **98.3% grouped accuracy from training-selected aggregate DNs**, with permutation p **0.0010** and **97.9%** on the four non-Title screens. Time-resolved selected-DN readout scored **90.0%**. Pokémon screen information is present in the MaleCNS DN population, but the 200-step aggregation did **not** hide it; including many uninformative DNs in the distance/classifier hid it. A next experiment should validate the same subset across more seeds, screen instances and game states before any behavioral use. No reward, RL, RAM sensory input or action policy was added.

Reproduce from the repository root with `redfly-benchmark/.venv/bin/python run_temporal_dn_experiment.py` and `redfly-benchmark/.venv/bin/python analyze_temporal_dn.py`. Experiment 4 reads saved Experiment 3 vectors and does not need the ROM.
