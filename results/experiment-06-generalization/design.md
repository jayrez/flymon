# Experiment 6 — frozen visual-instance generalization protocol

Frozen before neural simulation or outer-test evaluation. Dataset manifest SHA-256:
`8818669a58d9bd24e1aaafd28e78b4ffd0ec45974bccdb0d1999e9673b6e4624`. E3 encoder source SHA-256:
`44b7a88892d91c7aec49b240c8f67c52bc87940a492cc75d8a1cbab251f3e6a7`. No Experiments 1–5 result is changed.

## Dataset and independence

Classes and accepted sequence counts: {'bedroom': 5, 'dialogue': 8, 'menu': 5, 'title': 8, 'intro': 6}. Each instance contains ten actual
RGBA framebuffer samples, 4 emulator frames apart; 20 neural steps/sample, 200 total.
An instance is a deterministic natural capture trajectory, not an augmented image.
All its frames and all neural repeats stay grouped. Distinct instances share no
identical framebuffer hashes, even across classes. First-frame changed fractions
below 1% are flagged, not silently removed based on neural results.

Bedroom means the canonical `states/bedroom.state` room view, with naturally changed
player position/facing and camera framing. The savestate filename is retained as
the class label; it is not independently inferred from RAM. Dialogue means Oak's
opening text-box pages (both Oak and Pokémon illustration backgrounds). Menu means
the in-room pause-menu family, varying cursor and underlying room framing. It does
not pool options, new-game menus, inventory or unrelated interfaces. Intro means
natural poses in the opening Gengar/Nidorino animation; title means the fixed logo
with naturally changing Pokémon sprites. These are narrow visual families, not all
possible Pokémon dialogue, rooms or menus. All captures are deterministic and can
be temporally/visually correlated: distinct pixels do not imply independent game
sessions or broad semantic coverage. One attempted menu capture (`menu-06`) had no
menu on inspection and is excluded before neural evaluation, retained for audit.
The options screen is an OOD probe only. No ROM bytes or RAM-derived labels/input.

## Pixel-only preflight

Mean-frame pixel distance: {'different_instance_same_class': 30.48326100526757, 'different_class': 94.10831993359155, 'generalization_margin': 63.62505892832398}.
Encoder sequence distance: {'different_instance_same_class': 3.939350149798633, 'different_class': 15.938514854111224, 'generalization_margin': 11.99916470431259}.
Near-duplicate pairs below 1% first-frame changed pixels: 1.
Inspect the contact sheet and recorded pixel/histogram/encoder distances. Means
may hide local changes; all original frames and hashes are retained. No label is
changed after neural results. Diagnostic image classifiers use mean, standard
deviation, dark-pixel fraction and horizontal/vertical luminance profiles only.

## Model and readouts

Unchanged Experiment 3 18×20 spatial_grid / temporal_features / SpatialProjection,
same LC10a/LPLC2 indices, gains, quantization and ten-frame timing. Bit-exact encoder
regression on all five original E3 sequences must pass. MaleCNS v1.0, 166700 neurons,
25582938 connections, flybrain 0.1.0, CUDA P40. Primary DN readout is 200-step counts.
LC10a/LPLC2 counts are recorded separately. No gameplay policy, reward or action decoder.
Twenty new CNS seeds per image: [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220]. No outcome-based replay filtering.

## Outer and inner splits (fixed before testing)

Within each class, sort accepted instance IDs. Aligned instance fold j holds out
the jth instance of every class having one. With unequal class counts, later folds
have fewer classes; every instance is tested exactly once per seed block. Cross
these instance folds with four seed blocks: 201–205, 206–210, 211–215, 216–220.
Outer training excludes ALL held-out instances at ANY seed and ALL held-out seeds
at ANY instance. Every primary (instance,seed) trial receives one outer prediction.
Report trial-weighted accuracy and balanced accuracy, per-class recall, confusion.
Chance is 20% for five-way classification even in smaller test folds.

Within each outer training set, leave out each remaining aligned instance fold,
and hold out one of the three remaining seed blocks, cycling by inner fold order.
Re-rank DNs on that inner training subset, and score candidates [1314, 500, 250, 100, 50].
Choose maximal pooled inner accuracy; ties prefer the earlier/larger size. Re-rank
on all outer training data, freeze size and IDs, then predict once. ANOVA-like
between-class / within-class residual score with epsilon 1e-6, stable index tie
break. Euclidean nearest centroid on raw counts; no normalization or fitted metric.
The all-DN and each fixed candidate-size curve use the same outer splits.
Frozen E4 diagnostic uses the exact 20 FlyWire IDs in its previously recorded
`informative_dn_top20` table; no E6-driven choice of that subset.

## Other diagnostics

Exact-instance identity: seed-block outer holdout with all instance identities in
training; inner seed-block CV repeats DN ranking and population-size choice. This
diagnostic intentionally sees the same images and is not category generalization.
Encoder/pixel statistics: same outer splits, no DN selection. Projection counts:
same outer splits and raw Euclidean nearest centroid. Four-class no-title nested
analysis repeats selection from scratch; interpret as secondary.

Distance conditions use DIFFERENT seeds: same image, different image/same class,
different class. Margin = mean different-class minus mean within-class/different-
instance distance. Encoder margin is deterministic over image pairs. DN/projection
raw distances include all seed pairs with unequal seeds. Selected-DN geometry uses
a preregistered fixed top 50 to compare distances in equal dimension: for each
instance pair and each seed block, rank using data excluding BOTH aligned instance
folds and that entire seed block, then measure distances only on the withheld
seeds/images. This separate cross-fitted geometry cannot tune the primary model.
Its matrix, margin and noise distance explicitly refer to top50, not an adaptive
union of outer-selected IDs. Stability: outer selected-set and top50 Jaccard.

Leave-one-class-out geometry is exploratory: select top50 using four classes only,
then describe held-class within/between distances; no prediction of unseen labels.
No-vision baseline at every seed; spatial destruction control for first accepted
instance of each class at every seed, one fixed pixel permutation (seed 606), same
permutation for all frames, preserves exact per-frame luminance histograms. Controls
and the OOD options screen never enter training, selection, permutations or bootstrap.
Evaluate their distances to outer-training centroids with frozen selected features.

## Inference and leakage audits

1000 permutations, RNG 6062026: permute class labels among instances WITHIN each
aligned instance-fold block. This preserves block class composition and every
seed replicate of an image receives the same permuted label. This restricted,
conditional instance-level null preserves the prespecified stratified fold design;
later incomplete blocks have fewer exchangeable labels. Repeat the FULL inner
ranking, size selection, outer ranking and prediction for every permutation.
p=(1+null accuracy >= observed)/(1001); report mean and 95th percentile.
10000 bootstrap resamples, RNG 6062027: resample whole image instances WITHIN class,
carrying all their out-of-fold seed predictions. Report percentile 95% accuracy CI;
this is conditional on fitted folds, not a retraining/independent-game-session CI.
Assert zero image-hash, instance and seed overlap in every outer AND inner split.
Audit ranking inputs; perturb outer-test activity/labels and verify unchanged fold
feature selection. Verify permutation labels are constant within instance and
full nesting executes again. Verify bootstrap resamples instance units.

## Interpretation frozen before test results

PASS needs substantial above-chance nested double-holdout accuracy, instance-level
p<.05, most classes informative without title dependence, positive cross-fitted
selected-DN margin, and either frozen E4 transfer or meaningful cross-fold stability.
70% is a strong guide, not an arbitrary statistical cutoff. WEAK means modest,
class-dependent or uncertain generalization or unstable selected neurons. FAIL
means near-chance generalization or representation failure, even if image identity
works. Report A (fixed-image recognition), B (encoder succeeds/projection fails),
C (projection succeeds/DNs fail), D (some DN transfer/unstable subset), or E (stable
DN generalization), choosing the best-supported case and stating limitations.
If encoder itself fails, A is the closest outcome with explicit upstream caveat.
Stop at Experiment 6; only discuss whether an Experiment 7 study is justified.
