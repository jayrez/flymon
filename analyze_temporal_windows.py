"""Exploratory, seed-grouped selection of informative DN time windows."""
from __future__ import annotations

import itertools
import json
import time

import numpy as np

from analyze_temporal_dn import (RESULTS, NAMES, SEEDS, accuracy, classification_record,
    fixed_choices_cv, gram, grouped_gram, nested_bin_cv, predict_gram, representation)


def main():
    started = time.perf_counter()
    cpath = RESULTS / "classification.json"
    ppath = RESULTS / "permutation-results.json"
    c = json.loads(cpath.read_text())
    p = json.loads(ppath.read_text())
    z = np.load(RESULTS / "dn-bins-5.npz")
    conditions = list(z["conditions"])
    dynamic = z["counts"][[conditions.index(n) for n in NAMES]].transpose(1, 0, 2, 3)
    static = z["counts"][[conditions.index(n + "_static") for n in NAMES]].transpose(1, 0, 2, 3)
    x = representation(dynamic, 20)
    sx = representation(static, 20)
    kernels = {window: gram(x[:, :, window:window+1]) for window in range(10)}
    static_kernels = {window: gram(sx[:, :, window:window+1]) for window in range(10)}
    labels = np.tile(np.arange(5), (len(SEEDS), 1))
    confusion, choices = nested_bin_cv(kernels, labels, tuple(range(10)))
    record = classification_record(confusion, NAMES)
    static_record = classification_record(fixed_choices_cv(static_kernels, labels, choices), NAMES)
    four = x[:, [0, 2, 3, 4]]
    four_static = sx[:, [0, 2, 3, 4]]
    four_kernels = {window: gram(four[:, :, window:window+1]) for window in range(10)}
    four_static_kernels = {window: gram(four_static[:, :, window:window+1]) for window in range(10)}
    labels4 = np.tile(np.arange(4), (len(SEEDS), 1))
    confusion4, choices4 = nested_bin_cv(four_kernels, labels4, tuple(range(10)))
    four_names = [n for n in NAMES if n != "title"]
    c["nested_window"] = record
    c["nested_window_static_same_choices"] = static_record
    c["nested_window_choices"] = choices
    c["nested_window_four_screen"] = classification_record(confusion4, four_names)
    c["nested_window_four_screen_static_same_choices"] = classification_record(
        fixed_choices_cv(four_static_kernels, labels4, choices4), four_names)
    c["nested_window_four_screen_choices"] = choices4
    aggregate_kernel = gram(representation(dynamic, 200))
    paired = []
    for held, item in enumerate(choices):
        train = [seed for seed in range(len(SEEDS)) if seed != held]
        agg = predict_gram(aggregate_kernel, labels, train, held)
        win = predict_gram(kernels[item["bin_steps"]], labels, train, held)
        paired.append(dict(seed=int(SEEDS[held]), aggregate_correct=int(np.count_nonzero(agg == labels[held])),
                           window_correct=int(np.count_nonzero(win == labels[held]))))
    delta = np.array([item["window_correct"] - item["aggregate_correct"] for item in paired])
    rng = np.random.default_rng(2030)
    bootstrap = np.array([rng.choice(delta, len(delta), replace=True).sum() / 60 for _ in range(10000)])
    signs = np.array(list(itertools.product((-1, 1), repeat=len(delta))))
    null_gain = (signs * delta).sum(axis=1) / 60
    observed_gain = float(delta.sum() / 60)
    c["window_gain_vs_aggregate"] = dict(accuracy_difference=observed_gain,
        seed_group_bootstrap_95=[float(v) for v in np.quantile(bootstrap, (.025, .975))],
        paired_sign_flip_two_sided_p=float(np.mean(np.abs(null_gain) >= abs(observed_gain))),
        paired_by_seed=paired)
    # Family-wise null: each shuffle evaluates all ten windows and records the
    # maximum grouped accuracy. This is conservative for the nested selector.
    rng = np.random.default_rng(20260920)
    null = []
    for i in range(1000):
        shuffled = np.stack([rng.permutation(row) for row in labels])
        null.append(max(accuracy(grouped_gram(kernels[window], shuffled)) for window in range(10)))
        if (i + 1) % 250 == 0:
            print("Window null", i + 1, "of 1000", flush=True)
    null = np.array(null)
    p["nested_window"] = dict(observed=record["accuracy"], null_mean=float(null.mean()),
        null_95th=float(np.quantile(null, .95)),
        empirical_p=float((1 + np.count_nonzero(null >= record["accuracy"])) / (len(null) + 1)),
        permutations=1000, null_accuracies=null.tolist(),
        shuffle="labels independently permuted within seed; max of ten grouped window accuracies per shuffle (family-wise conservative null)")
    c["performance"]["window_selection_and_null_ms"] = (time.perf_counter() - started) * 1000
    cpath.write_text(json.dumps(c, indent=2) + "\n")
    ppath.write_text(json.dumps(p, indent=2) + "\n")
    print("Nested window", record["accuracy"], "static", static_record["accuracy"],
          "4-screen", c["nested_window_four_screen"]["accuracy"],
          "gain", observed_gain, "paired p", c["window_gain_vs_aggregate"]["paired_sign_flip_two_sided_p"],
          "family-wise null p", p["nested_window"]["empirical_p"], flush=True)


if __name__ == "__main__":
    main()
