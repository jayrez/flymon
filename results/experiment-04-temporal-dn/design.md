# Experiment 4 fixed analysis plan

Experiment 3's `nearest_centroid` and report generator both use leave-one-seed-out
validation: for a held-out seed, the centroid for each condition is computed
from its other 11 seeds, and all five held-out conditions are tested together.
No seed occurs in both train and test. This was inspected before Experiment 4.
The published 40% five-screen accuracy is already grouped by seed. Experiment 4
will independently recompute it from Experiment 3's saved DN vectors.

Experiment 3 preserved only 200-step per-DN totals, not per-DN temporal
activity. We will rerun the exact 460-element voltage vectors saved in its
`trials.json` for every condition/window, including the five static controls
and no-vision baseline. We will not recapture PyBoy or alter its encoder.
MaleCNS, 12 seeds, 200 steps, reset order and injection grouping are unchanged.
Every rerun's 200-step per-DN vector must exactly equal its Experiment 3
counterpart or analysis stops. DN spikes are accumulated into 40 consecutive
5-step bins; 10-, 20-, and 50-step bins and whole-run totals are exact sums
of these bins. `dn-bins-5.npz` stores condition × seed × time-bin × DN counts,
not raw spike indices. This is compact and fully sufficient for the analyses.

The primary classifier is Euclidean nearest centroid on raw binned counts,
flattened in chronological order. Features are never normalized using test
samples. Outer cross-validation holds out all trials of one seed. The aggregate
and each prespecified temporal bin size are evaluated with grouped outer
folds. The selected temporal bin size for the primary temporal claim is
chosen by leave-one-seed-out validation **inside each outer training fold**;
only then is the outer held-out seed predicted. This nested rule is repeated
inside label permutations. Chance is 1/5 or 1/4. For significance, condition
labels are independently permuted within each seed, preserving seed grouping,
and the entire grouped/nested evaluation is repeated at least 500 times.

Population-size comparisons use `all`, top 500, 250, 100 and 50 DNs. Neuron
scores are between-condition squared variation of class means divided by
within-condition squared variation, summed over temporal bins. Scores and
neuron selection are recomputed from **training seeds only** in each outer
fold. They are exploratory; the primary temporal claim uses all DNs and the
nested bin-size selector. A training-only inner selector will also report a
population size, but no test-driven winner will be used to claim success.

We will compare raw Euclidean and cosine distances within and between visual
conditions, same-seed/different-condition distances, and different-seed/
same-condition distances. Static and no-vision controls get the same readout.
Independent 20-step windows are evaluated with the same grouped classifier to
locate transient information. Diagnostic neuron IDs/types are reported from
training-fold ranks, with selection frequency across folds, not selected once
from the full dataset for classification.

A rerun check found intermittent CUDA trajectory divergence despite resetting the
same seed and injecting the same vectors: seed 109/New Game menu matched the
saved Experiment 3 aggregate in two of three initial single-trial repeats.
In a further eight repeats, seven matched the saved aggregate and all seven
matching runs had the **same SHA-256 hash of the entire 5-step DN tensor**;
the one nonmatching run had another hash. Therefore the capture runner uses
at most eight logged attempts per trial and retains only a run whose full
per-DN aggregate exactly matches Experiment 3. Attempts and timing are saved.
This conditioning is a reproducibility constraint and a possible limitation;
we will report how many retries occurred. If any trial fails eight attempts,
the experiment stops. No sensory or analysis parameters are changed by a
retry.

After the prespecified fixed-bin analysis, the exploratory 20-step window scan
found a first-window peak. To assess it without reusing the outer test seed
for window choice, an added analysis chooses among all ten windows by inner
seed-grouped validation and evaluates each outer held-out seed once. Because
this check was prompted by the observed scan, its permutation null is
conservative: for each of 1,000 within-seed label shuffles, it records the
maximum grouped accuracy across all ten windows. A paired, seed-level
bootstrap and exact sign-flip test compare its accuracy gain with the
aggregate readout. This is labeled exploratory and cannot retroactively
become the preregistered primary bin-size result.

During final validation, the exploratory population classifier's “all DN” row
failed to reproduce the primary all-DN classifier. The cause was NumPy
advanced-index axis reordering in the subset path. Selection was changed to
explicit `np.take(..., axis=-1)`, and all five all-DN confusion matrices now
match the primary classifier exactly. Only offline population analysis was
recomputed; recorded neural activity, Experiment 3 results, prespecified
all-DN bin analysis and its permutations were unchanged. This correction
revealed a strong training-selected aggregate DN subset. We added an
independent aggregate subset permutation test that refits rankings within
training seeds and corrects for five population sizes. A further temporal
subset null corrects across four temporal bin sizes and five sizes. These
post hoc subset results are kept separate from the prespecified all-DN
comparison.
