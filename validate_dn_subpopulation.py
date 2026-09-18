"""Validate Experiment 4's training-selected aggregate DN subpopulation."""
from __future__ import annotations

import json
import itertools
import time

import numpy as np

from analyze_temporal_dn import (RESULTS, NAMES, SEEDS, POPULATIONS, classification_record,
    fixed_population_cv, nested_population_cv, neuron_scores, predict_selected, representation)


def shuffled_population_accuracies(data, labels):
    """Group-held-out classification; train-only F ranks for all population sizes."""
    classes = labels.shape[1]
    correct = np.zeros(len(POPULATIONS), np.int32)
    for held in range(len(SEEDS)):
        train_seed = [seed for seed in range(len(SEEDS)) if seed != held]
        train_data = data[train_seed]
        order = np.argsort(labels[train_seed], axis=1)
        ordered = np.take_along_axis(train_data, order[:, :, None, None], axis=1)
        ranking = np.argsort(-neuron_scores(ordered), kind="stable")
        centroid = np.take(ordered.mean(axis=0), ranking, axis=-1)
        test = np.take(data[held], ranking, axis=-1)
        squared = np.square(test[:, None] - centroid[None]).sum(axis=2)
        cumulative = np.cumsum(squared, axis=-1)
        for i, size in enumerate(POPULATIONS):
            predicted = np.argmin(cumulative[:, :, size - 1], axis=1)
            correct[i] += np.count_nonzero(predicted == labels[held])
    return correct / (len(SEEDS) * classes)


def main():
    started = time.perf_counter()
    cpath = RESULTS / "classification.json"
    ppath = RESULTS / "permutation-results.json"
    c = json.loads(cpath.read_text())
    p = json.loads(ppath.read_text())
    archive = np.load(RESULTS / "dn-bins-5.npz")
    names = list(archive["conditions"])
    dynamic = archive["counts"][[names.index(n) for n in NAMES]].transpose(1, 0, 2, 3)
    static = archive["counts"][[names.index(n + "_static") for n in NAMES]].transpose(1, 0, 2, 3)
    x = {200: representation(dynamic, 200)}
    sx = {200: representation(static, 200)}
    choices = [{"bin_steps": 200} for _ in SEEDS]
    confusion, selected, freq, f_counts = nested_population_cv(x, choices)
    c["aggregate_nested_population"] = classification_record(confusion, NAMES)
    c["aggregate_nested_population_static_same_sizes"] = classification_record(
        fixed_population_cv(sx, selected), NAMES)
    c["aggregate_nested_population_choices"] = selected
    c["aggregate_informative_dn_count_f_gt_1_training_folds"] = f_counts
    c["aggregate_informative_dn_top20"] = [dict(dn_slot=int(slot),
        flywire_id=int(archive["flywire_ids"][slot]), cell_type=str(archive["cell_types"][slot]),
        side=str(archive["side"][slot]), training_fold_top50_frequency=int(count))
        for slot, count in freq.most_common(20)]
    four = x[200][:, [0, 2, 3, 4]]
    four_cf, four_choice, _, _ = nested_population_cv({200: four}, choices)
    c["aggregate_nested_population_four_screen"] = classification_record(
        four_cf, [n for n in NAMES if n != "title"])
    # Paired comparison with the temporal bin-and-population selector on the
    # same held-out seed groups; this compares readouts, not only chance.
    temporal_reps = {steps: representation(dynamic, steps) for steps in (5, 10, 20, 50)}
    paired = []
    for held in range(len(SEEDS)):
        train = [seed for seed in range(len(SEEDS)) if seed != held]
        agg_choice = selected[held]
        temp_choice = c["nested_population_choices"][held]
        agg_pred, _ = predict_selected(x[200], train, held, agg_choice["population"])
        temp_pred, _ = predict_selected(temporal_reps[temp_choice["bin_steps"]], train,
                                       held, temp_choice["population"])
        paired.append(dict(seed=int(SEEDS[held]), aggregate_correct=int(np.count_nonzero(agg_pred == np.arange(5))),
                           temporal_correct=int(np.count_nonzero(temp_pred == np.arange(5)))))
    delta = np.array([item["temporal_correct"] - item["aggregate_correct"] for item in paired])
    boot_rng = np.random.default_rng(2031)
    draws = np.array([boot_rng.choice(delta, len(delta), replace=True).sum() / 60 for _ in range(10000)])
    signs = np.array(list(itertools.product((-1, 1), repeat=len(delta))))
    null_delta = (signs * delta).sum(axis=1) / 60
    observed_delta = float(delta.sum() / 60)
    c["selected_temporal_vs_aggregate"] = dict(accuracy_difference=observed_delta,
        seed_group_bootstrap_95=[float(v) for v in np.quantile(draws, (.025, .975))],
        paired_sign_flip_two_sided_p=float(np.mean(np.abs(null_delta) >= abs(observed_delta))),
        paired_by_seed=paired)
    labels = np.tile(np.arange(5), (len(SEEDS), 1))
    observed_sizes = shuffled_population_accuracies(x[200], labels)
    expected_sizes = np.array([c["population_by_bin"]["200"][str(size)]["accuracy"] for size in POPULATIONS])
    if not np.allclose(observed_sizes, expected_sizes):
        raise RuntimeError(f"Population permutation evaluator mismatch: {observed_sizes} vs {expected_sizes}")
    rng = np.random.default_rng(20260921)
    null = []
    for iteration in range(1000):
        shuffled = np.stack([rng.permutation(row) for row in labels])
        null.append(float(shuffled_population_accuracies(x[200], shuffled).max()))
        if (iteration + 1) % 250 == 0:
            print("Population null", iteration + 1, "of 1000", flush=True)
    null = np.array(null)
    observed = c["aggregate_nested_population"]["accuracy"]
    p["aggregate_selected_population"] = dict(observed=observed, null_mean=float(null.mean()),
        null_95th=float(np.quantile(null, .95)),
        empirical_p=float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1)),
        permutations=1000, null_accuracies=null.tolist(),
        shuffle="condition labels permuted within each seed; DN F ranking refit from training seeds; max accuracy over all five sizes per shuffle (conservative family-wise null)")
    c["performance"]["aggregate_population_validation_ms"] = (time.perf_counter() - started) * 1000
    cpath.write_text(json.dumps(c, indent=2) + "\n")
    ppath.write_text(json.dumps(p, indent=2) + "\n")
    print("Aggregate nested population", observed, "static", c["aggregate_nested_population_static_same_sizes"]["accuracy"],
          "four screen", c["aggregate_nested_population_four_screen"]["accuracy"],
          "max-size permutation p", p["aggregate_selected_population"]["empirical_p"], flush=True)


if __name__ == "__main__":
    main()
