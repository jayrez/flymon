# Experiment 6 — visual-instance generalization

```text
Experiment 6 verdict: PASS

Classes: bedroom, dialogue, menu, title, intro
Instances per class: {'bedroom': 5, 'dialogue': 8, 'menu': 5, 'title': 8, 'intro': 6}
Total unique images: 32 instance first frames; 60 unique sequence framebuffers

Primary outer protocol: unseen image instance + unseen neural seed
Chance accuracy: 20.0% (uniform five-way); largest-class baseline 25.0%

All-DN held-out-instance accuracy: 63.59%
Nested selected-DN held-out-instance accuracy: 86.25%
Frozen Experiment-4-DN accuracy: 87.50%

95% instance-bootstrap interval: 79.06%–94.06%
Permutation p-value: 0.000999

Exact-image instance classification: all-DN 11.72%; selected-DN 35.94%; chance 3.125%
Semantic class classification: 86.25%; balanced accuracy 82.40%

Same-instance / different-seed distance: 13.8562
Different-instance / same-class distance: 14.6838
Different-class distance: 25.6824
(Distances above: independent pairwise cross-fitted top50 DNs.)

Encoder generalization margin: 11.9992
Projection-neuron generalization margin: 735.9516
All-DN generalization margin: 2.2419
Selected-DN generalization margin: 10.9985

DN population size selected by inner CV: {250: 25, 500: 1, 100: 6}
DN selection stability across folds: selected-set Jaccard 0.6794; top50 Jaccard 0.8658

Most confused classes: bedroom / menu (69 errors)
Best-generalizing class: dialogue (100.00%)
Worst-generalizing class: bedroom (41.00%)

Primary result: E

Leakage checks:
image overlap: PASS
instance overlap: PASS
seed overlap: PASS
DN-selection leakage: PASS
permutation leakage: PASS
population-size selection: PASS
bootstrap instance unit: PASS

Does the Experiment 4 DN signal generalize beyond the five exact screens? YES

```

## Inference and limits

The nested classifier scored 86.25% on 640 image/seed trials from 32 visual instances. Every image and seed was excluded from its outer training fold. The restricted instance-level null mean was 20.72%, its 95th percentile 35.01%, and p=0.000999 over 1000 complete nested repetitions. The null preserves aligned-block composition; it is not an unrestricted image-label permutation. The bootstrap uses images, not the 640 neural repeats, and is conditional on the fitted folds.

Encoder class accuracy was 96.88%, projection-population accuracy 96.88%. Excluding Title and repeating nested selection gave 80.83% (25% chance). Trivial image-statistics accuracy was 96.88%; high performance here limits claims about sophisticated or semantic neural representation.

These captures cover narrow visual families: one canonical room, Oak introduction text, one pause-menu layout, one intro animation and changing title sprites. They are naturally different images, not independent game sessions or diverse world states. Distinct hashes prove pixel separation, not semantic independence. The preflight flags one near-duplicate pair; see dataset-diagnostics.json. Menu here differs from the original New Game menu, so the frozen E4 subset is a transfer diagnostic across that shift as well.

Euclidean distances depend on dimension. The selected-DN margin uses separately cross-fitted fixed top50 features, excluding both compared instance blocks and test seeds. It is not the distance in the variable-size primary classifier. Stability overlaps are descriptive and partly reflect overlap among training folds. Exact-image identification uses 32 labels, so its absolute accuracy is not directly comparable to five-way class accuracy.

Post-hoc near-duplicate sensitivity: merging both aligned blocks containing menu-03 and menu-09, so neither image can train a model testing the other, gives 85.78% nested class accuracy. This stricter diagnostic does not change the frozen primary analysis or its p-value. The five menu captures should not be described as five independent background layouts: this pair changes only a small cursor region.

## Per-class recall

| Class | Recall |
| --- | ---: |
| bedroom | 41.00% |
| dialogue | 100.00% |
| menu | 71.00% |
| title | 100.00% |
| intro | 100.00% |

## Distance decomposition

| Stage | Same image/new seed | New image/same class | Different class | Margin |
| --- | ---: | ---: | ---: | ---: |
| encoder | 0.0000 | 3.9394 | 15.9385 | 11.9992 |
| projection | 42.3904 | 223.9386 | 959.8902 | 735.9516 |
| all_dn | 162.1792 | 162.2538 | 164.4958 | 2.2419 |
| selected_dn | 13.8562 | 14.6838 | 25.6824 | 10.9985 |

## Controls and exploratory geometry

| Probe | Nearest-centroid distance / primary test distance |
| --- | ---: |
| baseline_none | 1.119 |
| shuffled-bedroom-01 | 1.025 |
| shuffled-dialogue-01 | 1.019 |
| shuffled-menu-01 | 1.016 |
| shuffled-title-01 | 0.973 |
| shuffled-intro-01 | 1.025 |
| ood-options-01 | 1.007 |

| Shuffled source | Original class accuracy | Shuffled source-label retention |
| --- | ---: | ---: |
| shuffled-bedroom-01 | 100.00% | 0.00% |
| shuffled-dialogue-01 | 100.00% | 100.00% |
| shuffled-menu-01 | 30.00% | 0.00% |
| shuffled-title-01 | 100.00% | 100.00% |
| shuffled-intro-01 | 100.00% | 100.00% |

Shuffled-source comparisons use only models that exclude the original source instance and the test seeds. Label retention is descriptive, not correctness on a semantically valid screen.

These ratios use frozen outer models. They are descriptive, not a calibrated OOD detector; no closest-class assignment is counted as correct. The shuffled controls preserve each frame’s pixel histogram, and none of the controls entered training. Full assigned-label distributions are in classification.json.

Spatial destruction preserved the dialogue, title and intro source labels for all 20 test seeds each, while bedroom and menu source-label retention fell to zero. Thus the strongest class separation does not require intact screen structure and may largely reflect luminance statistics; the controls do not support a claim of sophisticated spatial or semantic recognition. The options probe has a nearest-centroid distance ratio of about 1.007, so this diagnostic provides no clear evidence of novelty separation from all trained classes.

| Class excluded from feature ranking | Held-class within distance | Distance to other classes | Margin |
| --- | ---: | ---: | ---: |
| bedroom | 14.476 | 19.896 | 5.421 |
| dialogue | 4.058 | 23.679 | 19.622 |
| menu | 6.759 | 19.275 | 12.516 |
| title | 4.367 | 18.407 | 14.040 |
| intro | 3.915 | 24.209 | 20.294 |

This leave-one-class-out geometry is exploratory and predicts no unseen label.

## Scientific conclusion

Outcome E: A stable selected DN representation transfers to unseen images and unseen CNS seeds within these narrow capture families. The earlier 98.3% result is therefore not solely recognition of five fixed images, but this experiment does not establish broad Pokémon-state understanding.

The evidence supports considering a separate, carefully scoped Experiment 7 behavioral-interface study, with the simple-statistics and narrow-dataset limitations above.
No gameplay control, action decoding, reward modeling or reinforcement learning was implemented. Experiments 1–5 were not modified.

## Reproduction

```bash
POKEMON_ROM=/external/pokemon-red.gb redfly-benchmark/.venv/bin/python capture_visual_instance.py --collect
OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python run_generalization_experiment.py
OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python analyze_generalization_experiment.py --permutations 1000 --workers 4
```

Dataset/protocol hashes, frame hashes, full inner/outer index splits, per-trial counts, exact frozen FlyWire IDs, prediction arrays, permutation draws, selection frequencies and diagnostics are stored beside this report.
