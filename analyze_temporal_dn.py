"""Experiment 4: grouped temporal-DN readout, controls, permutations and plots."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import statistics
import time
from xml.sax.saxutils import escape

import numpy as np
from scipy.spatial.distance import cdist

from run_spatiotemporal_experiment import ROOT, NAMES, SEEDS

RESULTS = ROOT / "results/experiment-04-temporal-dn"
CAPTURES = ROOT / "captures/experiment-04"
BINS = (5, 10, 20, 50, 200)
TEMPORAL_PREFERENCE = (50, 20, 10, 5)  # tie: prefer fewer temporal features
POPULATIONS = (1314, 500, 250, 100, 50)
PERMUTATIONS = 1000


def representation(fine: np.ndarray, steps: int) -> np.ndarray:
    """[seed, condition, 40, DN] -> [seed, condition, bins, DN]."""
    if steps % 5 or 200 % steps:
        raise ValueError(steps)
    return fine.reshape(*fine.shape[:2], 200 // steps, steps // 5, fine.shape[-1]).sum(axis=3, dtype=np.float32)


def gram(data: np.ndarray) -> np.ndarray:
    flat = data.reshape(data.shape[0] * data.shape[1], -1).astype(np.float64)
    return flat @ flat.T


def predict_gram(kernel, labels, train_seeds, held_seed):
    classes = labels.shape[1]
    test = np.arange(held_seed * classes, (held_seed + 1) * classes)
    train_flat = np.concatenate([np.arange(seed * classes, (seed + 1) * classes)
                                 for seed in train_seeds])
    y = labels.ravel()
    dist = np.empty((classes, classes), np.float64)
    for cls in range(classes):
        idx = train_flat[y[train_flat] == cls]
        if len(idx) != len(train_seeds):
            raise RuntimeError("Each training seed must contain one of each condition")
        norm_centroid = kernel[np.ix_(idx, idx)].mean()
        dist[:, cls] = kernel[test, test] - 2 * kernel[np.ix_(test, idx)].mean(axis=1) + norm_centroid
    return np.argmin(dist, axis=1)


def grouped_gram(kernel, labels, allowed=None):
    seeds = range(labels.shape[0]) if allowed is None else allowed
    seeds = list(seeds)
    confusion = np.zeros((labels.shape[1], labels.shape[1]), np.int32)
    for held in seeds:
        train = [seed for seed in seeds if seed != held]
        predicted = predict_gram(kernel, labels, train, held)
        for truth, guess in zip(labels[held], predicted):
            confusion[truth, guess] += 1
    return confusion


def nested_bin_cv(kernels, labels, preference=TEMPORAL_PREFERENCE):
    confusion = np.zeros((labels.shape[1], labels.shape[1]), np.int32)
    choices = []
    for held in range(labels.shape[0]):
        train = [seed for seed in range(labels.shape[0]) if seed != held]
        scores = {steps: accuracy(grouped_gram(kernels[steps], labels, train))
                  for steps in preference}
        chosen = max(preference, key=lambda steps: scores[steps])
        choices.append(dict(seed=int(SEEDS[held]), bin_steps=chosen,
                            inner_accuracy={str(k): v for k, v in scores.items()}))
        predicted = predict_gram(kernels[chosen], labels, train, held)
        for truth, guess in zip(labels[held], predicted):
            confusion[truth, guess] += 1
    return confusion, choices


def fixed_choices_cv(kernels, labels, choices):
    confusion = np.zeros((labels.shape[1], labels.shape[1]), np.int32)
    for held, item in enumerate(choices):
        train = [seed for seed in range(labels.shape[0]) if seed != held]
        predicted = predict_gram(kernels[item["bin_steps"]], labels, train, held)
        for truth, guess in zip(labels[held], predicted):
            confusion[truth, guess] += 1
    return confusion


def accuracy(confusion):
    return float(np.trace(confusion) / confusion.sum())


def classification_record(confusion, names):
    recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    return dict(accuracy=accuracy(confusion), balanced_accuracy=float(recall.mean()),
                confusion=confusion.tolist(), labels=list(names), chance=1 / len(names))


def neuron_scores(training):
    """Training-only class-mean versus residual variation, summed over bins."""
    mean = training.mean(axis=0)  # [condition, bin, DN]
    grand = mean.mean(axis=0)
    between = training.shape[0] * np.square(mean - grand).sum(axis=(0, 1)) / (training.shape[1] - 1)
    within = np.square(training - mean[None]).sum(axis=(0, 1, 2)) / (training.shape[1] * (training.shape[0] - 1))
    return between / (within + 1e-6)


def predict_selected(data, train_seeds, held_seed, size):
    training = data[train_seeds]
    ranking = np.argsort(-neuron_scores(training), kind="stable")
    selected = ranking if size >= data.shape[-1] else ranking[:size]
    centroids = np.take(training, selected, axis=-1).mean(axis=0).reshape(data.shape[1], -1)
    test = np.take(data[held_seed], selected, axis=-1).reshape(data.shape[1], -1)
    predicted = np.argmin(cdist(test, centroids, metric="sqeuclidean"), axis=1)
    return predicted, ranking


def grouped_selected(data, size, allowed=None):
    seeds = list(range(data.shape[0])) if allowed is None else list(allowed)
    confusion = np.zeros((data.shape[1], data.shape[1]), np.int32)
    rankings = {}
    for held in seeds:
        train = [seed for seed in seeds if seed != held]
        predicted, ranking = predict_selected(data, train, held, size)
        rankings[held] = ranking
        for truth, guess in enumerate(predicted):
            confusion[truth, guess] += 1
    return confusion, rankings


def nested_population_cv(representations, bin_choices):
    nclass = next(iter(representations.values())).shape[1]
    confusion = np.zeros((nclass, nclass), np.int32)
    choices = []
    top50_frequency = Counter()
    counts_f_gt_1 = []
    for held, bin_item in enumerate(bin_choices):
        steps = bin_item["bin_steps"]
        data = representations[steps]
        train = [seed for seed in range(data.shape[0]) if seed != held]
        inner = {size: accuracy(grouped_selected(data, size, train)[0]) for size in POPULATIONS}
        chosen = max(POPULATIONS, key=lambda size: inner[size])
        pred, ranking = predict_selected(data, train, held, chosen)
        scores = neuron_scores(data[train])
        counts_f_gt_1.append(int((scores > 1).sum()))
        top50_frequency.update(ranking[:50].tolist())
        choices.append(dict(seed=int(SEEDS[held]), bin_steps=steps, population=chosen,
                            inner_accuracy={str(k): v for k, v in inner.items()}))
        for truth, guess in enumerate(pred):
            confusion[truth, guess] += 1
    return confusion, choices, top50_frequency, counts_f_gt_1


def distances(data):
    """Euclidean trial distances and mean-vector cosine for five visual classes."""
    flat = data.reshape(data.shape[0], data.shape[1], -1).astype(np.float64)
    within = {}
    for i, name in enumerate(NAMES[:data.shape[1]]):
        matrix = cdist(flat[:, i], flat[:, i])
        values = matrix[np.triu_indices(len(SEEDS), 1)]
        within[name] = dict(median=float(np.median(values)), mean=float(np.mean(values)))
    between = {}
    for i, j in combinations(range(data.shape[1]), 2):
        a, b = flat[:, i], flat[:, j]
        ma, mb = a.mean(axis=0), b.mean(axis=0)
        matrix = cdist(a, b)
        cosine = float(np.dot(ma, mb) / (np.linalg.norm(ma) * np.linalg.norm(mb)))
        between[f"{NAMES[i]}__{NAMES[j]}"] = dict(
            median_cross=float(np.median(matrix)), median_same_seed=float(np.median(np.linalg.norm(a-b, axis=1))),
            mean_vector_distance=float(np.linalg.norm(ma-mb)), mean_vector_cosine=cosine)
    w = float(np.mean([v["median"] for v in within.values()]))
    b = float(np.mean([v["median_cross"] for v in between.values()]))
    paired = float(np.mean([v["median_same_seed"] for v in between.values()]))
    return dict(within=within, between=between, mean_within=w, mean_between=b,
                between_within_ratio=b / w, mean_same_seed_different_condition=paired,
                mean_different_seed_same_condition=w,
                mean_pairwise_cosine=float(np.mean([v["mean_vector_cosine"] for v in between.values()])))


def permutation_test(kernels, kind, observed, count=PERMUTATIONS):
    rng = np.random.default_rng(20260918 + (0 if kind == "aggregate" else 1))
    labels = np.tile(np.arange(5), (len(SEEDS), 1))
    null = []
    for _ in range(count):
        shuffled = np.stack([rng.permutation(row) for row in labels])
        if kind == "aggregate":
            score = accuracy(grouped_gram(kernels[200], shuffled))
        elif kind == "nested_temporal":
            score = accuracy(nested_bin_cv(kernels, shuffled)[0])
        else:
            raise ValueError(kind)
        null.append(score)
    null = np.array(null)
    return dict(observed=observed, null_mean=float(null.mean()), null_95th=float(np.quantile(null, .95)),
                empirical_p=float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1)),
                permutations=count, null_accuracies=null.tolist(),
                shuffle="independent condition-label permutation within each seed; grouped/nested evaluation repeated")


def fixed_population_cv(representations, choices):
    classes = next(iter(representations.values())).shape[1]
    confusion = np.zeros((classes, classes), np.int32)
    for held, item in enumerate(choices):
        data = representations[item["bin_steps"]]
        train = [seed for seed in range(data.shape[0]) if seed != held]
        predicted, _ = predict_selected(data, train, held, item["population"])
        for truth, guess in enumerate(predicted):
            confusion[truth, guess] += 1
    return confusion


def baseline_distances(visual_reps, baseline_reps, steps):
    a = visual_reps[steps].reshape(len(SEEDS), len(NAMES), -1).astype(np.float64)
    base = baseline_reps[steps].reshape(len(SEEDS), -1).astype(np.float64)
    return {name: dict(mean_vector_euclidean=float(np.linalg.norm(a[:, i].mean(axis=0) - base.mean(axis=0))),
                       median_matched_seed=float(np.median(np.linalg.norm(a[:, i] - base, axis=1))),
                       mean_vector_cosine=float(np.dot(a[:, i].mean(axis=0), base.mean(axis=0)) /
                                                (np.linalg.norm(a[:, i].mean(axis=0)) * np.linalg.norm(base.mean(axis=0)))))
            for i, name in enumerate(NAMES)}


def main():
    started = time.perf_counter()
    source = json.loads((ROOT / "results/experiment-03-spatiotemporal/trials.json").read_text())
    capture = json.loads((RESULTS / "capture-metrics.json").read_text())
    archive = np.load(RESULTS / "dn-bins-5.npz")
    conditions = list(archive["conditions"])
    raw = archive["counts"]
    if raw.shape != (11, 12, 40, 1314) or list(archive["seeds"]) != SEEDS:
        raise RuntimeError("Unexpected DN capture shape or seeds")
    dynamic = raw[[conditions.index(n) for n in NAMES]].transpose(1, 0, 2, 3)
    static = raw[[conditions.index(n + "_static") for n in NAMES]].transpose(1, 0, 2, 3)
    baseline = raw[conditions.index("baseline_none")][:, None]
    # Recompute the Experiment 3 grouped classifier from its saved aggregate,
    # independent of the newly captured bins.
    original = np.array([[t["dn_counts"] for t in source["trials"][name]] for name in NAMES], np.float32)
    original = original.transpose(1, 0, 2)[:, :, None, :]
    labels5 = np.tile(np.arange(5), (len(SEEDS), 1))
    original_confusion = grouped_gram(gram(original), labels5)
    if not np.array_equal(original_confusion, source["dn_classifier"]["confusion"]):
        raise RuntimeError("Original Experiment 3 classifier not reproduced")
    reps, static_reps, baseline_reps, binning_ms = {}, {}, {}, {}
    for steps in BINS:
        t0 = time.perf_counter()
        reps[steps] = representation(dynamic, steps)
        static_reps[steps] = representation(static, steps)
        baseline_reps[steps] = representation(baseline, steps)
        binning_ms[str(steps)] = (time.perf_counter() - t0) * 1000
    if not np.array_equal(reps[200][:, :, 0], original[:, :, 0]):
        raise RuntimeError("New aggregate does not match Experiment 3")
    kernels = {steps: gram(reps[steps]) for steps in BINS}
    static_kernels = {steps: gram(static_reps[steps]) for steps in BINS}
    grouped = {str(steps): classification_record(grouped_gram(kernels[steps], labels5), NAMES)
               for steps in BINS}
    static_grouped = {str(steps): classification_record(grouped_gram(static_kernels[steps], labels5), NAMES)
                      for steps in BINS}
    nested_confusion, bin_choices = nested_bin_cv(kernels, labels5)
    nested_static = fixed_choices_cv(static_kernels, labels5, bin_choices)
    nested = classification_record(nested_confusion, NAMES)
    static_nested = classification_record(nested_static, NAMES)
    if grouped["200"]["confusion"] != original_confusion.tolist():
        raise RuntimeError("Grouped aggregate confusion mismatch")
    # Four non-Title conditions; all seeds remain grouped.
    four_names = [name for name in NAMES if name != "title"]
    four_reps = {steps: data[:, [0, 2, 3, 4]] for steps, data in reps.items()}
    four_static = {steps: data[:, [0, 2, 3, 4]] for steps, data in static_reps.items()}
    labels4 = np.tile(np.arange(4), (len(SEEDS), 1))
    four_kernels = {steps: gram(data) for steps, data in four_reps.items()}
    four_static_kernels = {steps: gram(data) for steps, data in four_static.items()}
    four_grouped = {str(steps): classification_record(grouped_gram(four_kernels[steps], labels4), four_names)
                    for steps in BINS}
    four_nested_confusion, four_choices = nested_bin_cv(four_kernels, labels4)
    four_nested = classification_record(four_nested_confusion, four_names)
    four_static_nested = classification_record(fixed_choices_cv(four_static_kernels, labels4, four_choices), four_names)
    separation = {str(steps): distances(reps[steps]) for steps in BINS}
    static_separation = {str(steps): distances(static_reps[steps]) for steps in BINS}
    baseline_sep = {str(steps): baseline_distances(reps, baseline_reps, steps) for steps in BINS}
    time_windows = []
    for window in range(10):
        view = reps[20][:, :, window:window + 1, :]
        record = classification_record(grouped_gram(gram(view), labels5), NAMES)
        record.update(start_step=window * 20, end_step=(window + 1) * 20,
                      separation=distances(view))
        time_windows.append(record)
    population = {}
    for steps in BINS:
        population[str(steps)] = {}
        for size in POPULATIONS:
            confusion, _ = grouped_selected(reps[steps], size)
            population[str(steps)][str(size)] = classification_record(confusion, NAMES)
    nested_pop_confusion, population_choices, top50_freq, f_gt_1 = nested_population_cv(reps, bin_choices)
    nested_pop = classification_record(nested_pop_confusion, NAMES)
    static_nested_pop = classification_record(fixed_population_cv(static_reps, population_choices), NAMES)
    top = []
    for dn_slot, frequency in top50_freq.most_common(20):
        top.append(dict(dn_slot=dn_slot, flywire_id=int(archive["flywire_ids"][dn_slot]),
                        cell_type=str(archive["cell_types"][dn_slot]), side=str(archive["side"][dn_slot]),
                        training_fold_top50_frequency=frequency))
    analysis_before_perm_ms = (time.perf_counter() - started) * 1000
    print("Grouped accuracy by bin:", {key: round(val["accuracy"], 3) for key, val in grouped.items()}, flush=True)
    print("Nested bin-selected accuracy:", nested["accuracy"], "static", static_nested["accuracy"], flush=True)
    print("Population nested accuracy:", nested_pop["accuracy"], "static", static_nested_pop["accuracy"], flush=True)
    t0 = time.perf_counter()
    perm_aggregate = permutation_test(kernels, "aggregate", grouped["200"]["accuracy"])
    print("Aggregate permutation p:", perm_aggregate["empirical_p"], flush=True)
    perm_temporal = permutation_test(kernels, "nested_temporal", nested["accuracy"])
    print("Nested temporal permutation p:", perm_temporal["empirical_p"], flush=True)
    permutation_ms = (time.perf_counter() - t0) * 1000
    attempts = [trial["attempts"] for group in capture["timings"].values() for trial in group]
    step_ms = [trial["step_ms"] for group in capture["timings"].values() for trial in group]
    dn_capture_ms = [trial["dn_capture_ms"] for group in capture["timings"].values() for trial in group]
    classification = dict(protocol=dict(seeds=SEEDS, conditions=NAMES, held_out_group="seed",
        classifier="raw-count Euclidean nearest centroid", outer_folds=12,
        inner_selection="11-seed leave-one-seed-out within outer training seeds; prefer coarser bin on ties",
        bin_steps=BINS, population_sizes=POPULATIONS),
        original_experiment3=classification_record(original_confusion, NAMES),
        grouped_experiment3=classification_record(original_confusion, NAMES),
        aggregate=grouped["200"], dynamic_by_bin=grouped, static_by_bin=static_grouped,
        nested_dynamic=nested, nested_static_same_bins=static_nested,
        nested_bin_choices=bin_choices, four_screen_by_bin=four_grouped,
        four_screen_nested=four_nested, four_screen_static_same_bins=four_static_nested,
        four_screen_bin_choices=four_choices, separation=separation, static_separation=static_separation,
        no_vision_separation=baseline_sep, time_windows=time_windows,
        population_by_bin=population, nested_population=nested_pop,
        nested_population_static_same_choices=static_nested_pop,
        nested_population_choices=population_choices, informative_dn_top20=top,
        informative_dn_count_f_gt_1_training_folds=f_gt_1,
        performance=dict(median_step_ms=float(statistics.median(step_ms)),
            median_dn_capture_ms=float(statistics.median(dn_capture_ms)), binning_ms=binning_ms,
            analysis_before_permutations_ms=analysis_before_perm_ms, permutations_ms=permutation_ms,
            total_analysis_ms=(time.perf_counter() - started) * 1000,
            cuda_replay_extra_attempts=sum(a - 1 for a in attempts),
            cuda_replay_trials_requiring_retry=sum(a > 1 for a in attempts)))
    permutation = dict(aggregate=perm_aggregate, nested_temporal=perm_temporal,
                       description="Condition labels shuffled independently within each seed; all grouped folds and inner bin selection rerun")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "classification.json").write_text(json.dumps(classification, indent=2) + "\n")
    (RESULTS / "permutation-results.json").write_text(json.dumps(permutation, indent=2) + "\n")
    print("Saved base Experiment 4 analysis; aggregate", grouped["200"]["accuracy"],
          "nested temporal", nested["accuracy"], "p", perm_temporal["empirical_p"], flush=True)


def svg_document(width, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#ffffff"/>'
            f'<g font-family="DejaVu Sans,Arial,sans-serif">{body}</g></svg>')


def svg_text(x, y, value, size=14, anchor="start", color="#18212f", weight="normal"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}">{escape(str(value))}</text>')


def bar_plot(title, labels, series, path, chance):
    width, height, left, top, bottom = 900, 440, 90, 65, 360
    body = svg_text(width / 2, 30, title, 19, "middle", weight="bold")
    for value in (0, .2, .4, .6, .8, 1):
        y = bottom - value * (bottom - top)
        body += f'<line x1="{left}" y1="{y}" x2="840" y2="{y}" stroke="#e6e9ef"/>'
        body += svg_text(left - 12, y + 5, f"{value:.0%}", 12, "end")
    xstep = (840 - left) / len(labels)
    width_bar = min(32, xstep / (len(series) + 1))
    colors = ("#2563eb", "#dc6b19", "#0f9d88", "#9b59b6")
    for i, (label, values) in enumerate(series.items()):
        for j, val in enumerate(values):
            x = left + j * xstep + xstep * .17 + i * width_bar
            y = bottom - val * (bottom - top)
            body += f'<rect x="{x:.1f}" y="{y:.1f}" width="{width_bar:.1f}" height="{bottom-y:.1f}" fill="{colors[i]}"/>'
            body += svg_text(x + width_bar / 2, y - 5, f"{val:.0%}", 11, "middle")
        body += f'<rect x="{left + i * 160}" y="405" width="12" height="12" fill="{colors[i]}"/>'
        body += svg_text(left + 18 + i * 160, 416, label, 12)
    for j, label in enumerate(labels):
        body += svg_text(left + (j + .5) * xstep, bottom + 25, label, 12, "middle")
    cy = bottom - chance * (bottom - top)
    body += f'<line x1="{left}" y1="{cy}" x2="840" y2="{cy}" stroke="#a83232" stroke-dasharray="6 4"/>'
    body += svg_text(842, cy + 4, "chance", 11)
    path.write_text(svg_document(width, height, body))


def line_plot(title, xlabels, series, path, chance):
    width, height, left, top, bottom = 900, 440, 80, 60, 355
    body = svg_text(width / 2, 30, title, 19, "middle", weight="bold")
    for value in (0, .2, .4, .6, .8, 1):
        y = bottom - value * (bottom - top)
        body += f'<line x1="{left}" y1="{y}" x2="820" y2="{y}" stroke="#e6e9ef"/>'
        body += svg_text(left - 10, y + 4, f"{value:.0%}", 12, "end")
    colors = ("#2563eb", "#dc6b19", "#0f9d88", "#9b59b6", "#45556c")
    for i, (name, values) in enumerate(series.items()):
        coords = [(left + j * (740 / max(1, len(xlabels) - 1)), bottom - val * (bottom - top))
                  for j, val in enumerate(values)]
        body += '<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in coords) + f'" fill="none" stroke="{colors[i]}" stroke-width="3"/>'
        for x, y in coords:
            body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[i]}"/>'
        body += f'<rect x="{left + i * 160}" y="405" width="12" height="12" fill="{colors[i]}"/>'
        body += svg_text(left + 18 + i * 160, 416, name, 12)
    for j, label in enumerate(xlabels):
        body += svg_text(left + j * (740 / max(1, len(xlabels) - 1)), bottom + 25, label, 12, "middle")
    cy = bottom - chance * (bottom - top)
    body += f'<line x1="{left}" y1="{cy}" x2="820" y2="{cy}" stroke="#a83232" stroke-dasharray="6 4"/>'
    path.write_text(svg_document(width, height, body))


def permutation_plot(permutation, path):
    keys = [key for key in ("aggregate", "nested_temporal", "nested_window",
                             "aggregate_selected_population", "temporal_selected_population") if key in permutation]
    width, height = 40 + 485 * len(keys), 470
    body = svg_text(width / 2, 30, "Grouped-seed label permutation nulls", 19, "middle", weight="bold")
    for panel, key in enumerate(keys):
        item = permutation[key]
        left = 70 + panel * 485
        top, bottom = 75, 370
        values = np.array(item["null_accuracies"])
        hist, edges = np.histogram(values, bins=np.arange(0, 1.01, 1 / 30))
        hmax = max(hist.max(), 1)
        for j, h in enumerate(hist):
            if h == 0:
                continue
            x = left + edges[j] * 420 / 1.0
            w = (edges[j+1] - edges[j]) * 420 / 1.0
            y = bottom - h / hmax * (bottom - top)
            body += f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bottom-y:.1f}" fill="#7fa8df"/>'
        ox = left + item["observed"] * 420 / 1.0
        body += f'<line x1="{ox:.1f}" y1="{top}" x2="{ox:.1f}" y2="{bottom}" stroke="#b42318" stroke-width="3"/>'
        body += svg_text(left + 210, 405, f"{key.replace('_',' ')}: observed {item['observed']:.1%}, p={item['empirical_p']:.4f}", 13, "middle")
        body += svg_text(left + 210, 65, f"null mean {item['null_mean']:.1%}; 95th {item['null_95th']:.1%}", 12, "middle")
        for tick in (0, .2, .4, .6, .8, 1.0):
            body += svg_text(left + tick * 420 / 1.0, bottom + 22, f"{tick:.0%}", 11, "middle")
    path.write_text(svg_document(width, height, body))


def confusion_plot(items, path):
    width, height = 55 + 350 * len(items), 425
    body = svg_text(width / 2, 30, "Held-out seed confusion matrices", 19, "middle", weight="bold")
    for panel, (title, record) in enumerate(items):
        matrix = np.array(record["confusion"])
        left = 55 + 350 * panel
        top = 100
        cell = 44
        body += svg_text(left + 130, 65, f"{title} ({record['accuracy']:.1%})", 14, "middle", weight="bold")
        for i in range(matrix.shape[0]):
            body += svg_text(left + 25, top + i * cell + 28, NAMES[i].replace("new_game_menu", "menu").replace("oak_dialogue", "dialogue"), 10, "end")
            body += svg_text(left + 58 + i * cell, top - 12, str(i + 1), 11, "middle")
            for j in range(matrix.shape[1]):
                value = int(matrix[i, j])
                opacity = .12 + .80 * value / 12
                x, y = left + 36 + j * cell, top + i * cell
                body += f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="#2563eb" opacity="{opacity:.2f}"/>'
                body += svg_text(x + (cell-2)/2, y + 28, value, 14, "middle")
    body += svg_text(width / 2, 375, "Columns 1–5: Intro, Title, New Game menu, Oak dialogue, Bedroom", 12, "middle")
    path.write_text(svg_document(width, height, body))


def make_plots(classification, permutation):
    CAPTURES.mkdir(parents=True, exist_ok=True)
    bins = ["200", "50", "20", "10", "5"]
    bar_plot("DN classification by bin size (12 grouped seed folds)",
             ["aggregate" if b == "200" else b + " steps" for b in bins],
             {"spatiotemporal": [classification["dynamic_by_bin"][b]["accuracy"] for b in bins],
              "static input": [classification["static_by_bin"][b]["accuracy"] for b in bins]},
             CAPTURES / "classification-by-bin-size.svg", .2)
    line_plot("Accuracy versus training-fold-selected DN population size", [str(x) for x in POPULATIONS],
              {("aggregate" if b == "200" else b + " steps"):
               [classification["population_by_bin"][b][str(size)]["accuracy"] for size in POPULATIONS]
               for b in bins}, CAPTURES / "accuracy-vs-dn-population.svg", .2)
    line_plot("DN identity information by simulation window", [f"{i*20}-{(i+1)*20}" for i in range(10)],
              {"grouped accuracy": [x["accuracy"] for x in classification["time_windows"]]},
              CAPTURES / "information-vs-time.svg", .2)
    broad_keys = ("aggregate", "nested_temporal", "nested_window")
    permutation_plot({key: permutation[key] for key in broad_keys if key in permutation},
                     CAPTURES / "permutation-null-vs-observed.svg")
    if "nested_window" in permutation:
        permutation_plot({"nested_window": permutation["nested_window"]}, CAPTURES / "window-permutation-null.svg")
    subset_keys = ("aggregate_selected_population", "temporal_selected_population")
    if all(key in permutation for key in subset_keys):
        permutation_plot({key: permutation[key] for key in subset_keys}, CAPTURES / "subset-permutation-null.svg")
    confusion_plot([("Aggregate", classification["aggregate"]),
                    ("Nested temporal", classification["nested_dynamic"]),
                    ("Selected window", classification["nested_window"] if "nested_window" in classification else classification["nested_static_same_bins"])],
                   CAPTURES / "confusion-matrices.svg")
    if "aggregate_nested_population" in classification:
        confusion_plot([("Aggregate subset", classification["aggregate_nested_population"]),
                        ("Temporal subset", classification["nested_population"])],
                       CAPTURES / "subset-confusion-matrices.svg")


def make_report(c, permutations):
    """Build the Experiment 4 report from validated, grouped results."""
    bin_mode = Counter(item["bin_steps"] for item in c["nested_bin_choices"]).most_common(1)[0][0]
    aggregate_pop_mode = Counter(item["population"] for item in c["aggregate_nested_population_choices"]).most_common(1)[0][0]
    temporal_pop_mode = Counter(item["population"] for item in c["nested_population_choices"]).most_common(1)[0][0]
    aggregate_all = c["aggregate"]["accuracy"]
    temporal_all = c["nested_dynamic"]["accuracy"]
    early_window = c["nested_window"]["accuracy"]
    aggregate_selected = c["aggregate_nested_population"]["accuracy"]
    temporal_selected = c["nested_population"]["accuracy"]
    static_selected = c["nested_population_static_same_choices"]["accuracy"]
    gain = c["selected_temporal_vs_aggregate"]
    perf = c["performance"]
    lines = []
    add = lines.append
    add("# Experiment 4 — temporal DN readout versus informative DN subsets\n")
    add(f"**Verdict: FAIL for the temporal-readout hypothesis. Primary finding: C — screen identity survives in a small DN subpopulation and is masked by population-wide noise.** The prespecified all-DN nested bin readout changed five-screen accuracy from **{aggregate_all:.1%}** aggregate to **{temporal_all:.1%}**. In contrast, training-selected DNs using **aggregate counts** reached **{aggregate_selected:.1%}**, versus **{temporal_selected:.1%}** for nested temporal bins with selected DNs. Chance is **20%**.\n")
    add("## Experiment 3 classifier verification and seed isolation\n")
    add("Experiment 3 used leave-one-**seed**-out Euclidean nearest centroid: each fold trained on all five conditions from 11 seeds and tested all five conditions of the twelfth. No seed was shared between training and test. We independently reproduced the original **40.0% (24/60)** and its exact confusion matrix from the saved Experiment 3 DN vectors. The Experiment 4 replay also reproduced that aggregate vector and classifier exactly. The original 40% was **not seed leakage**. The four-screen aggregate result excluding Title was **45.8%** versus 25% chance.\n")
    add("## Controlled replay and DN recording\n")
    add("Experiment 3 saved only 200-step per-DN totals, so we replayed its exact saved 460-element injection vectors. The five visual conditions, five static controls, no-vision baseline, canonical bedroom sequence, encoder, 460 projection cells, 1,314 DN cells, 12 seeds (`101..112`), reset procedure and 200 simulation steps were unchanged. No ROM or Pokémon RAM was accessed during Experiment 4. [DN bins](dn-bins-5.npz) contains compressed **11 conditions × 12 seeds × 40 five-step bins × 1,314 DNs**. The 10-, 20-, 50- and 200-step representations are exact sums, with no raw spike dumps.\n")
    add(f"Every trial's complete per-DN aggregate was required to match Experiment 3. Intermittent CUDA replay divergence required **{perf['cuda_replay_extra_attempts']} extra attempts in {perf['cuda_replay_trials_requiring_retry']} of 132 trials**; the fixed eight-attempt limit and attempts are recorded in [capture metrics](capture-metrics.json). In repeated checks, matching aggregate runs had the same entire five-step tensor hash. This conditional replay is a limitation, and no sensory or readout parameter was changed to make a run match.\n")
    add("## Raw DN separability: aggregate versus time bins\n")
    add("A representation is the chronological raw firing count for each DN in each bin, flattened for Euclidean nearest centroid; no test-derived normalization is used. Within is the average of each screen's median **different-seed** distance. Between is the average of ten screens' median cross-seed distance. Same-seed/different-screen distances hold the stochastic seed constant and are listed separately. Cosines compare condition mean vectors. Absolute Euclidean scales differ across dimensions, so ratios are compared within each row.\n")
    add("| Readout | Dimensions | Within | Between | Between/within | Mean cosine | Same-seed different-screen |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for steps in (200, 50, 20, 10, 5):
        item = c["separation"][str(steps)]
        name = "Aggregate" if steps == 200 else f"{steps}-step bins"
        add(f"| {name} | 1,314 × {200//steps} | {item['mean_within']:.1f} | {item['mean_between']:.1f} | {item['between_within_ratio']:.3f} | {item['mean_pairwise_cosine']:.5f} | {item['mean_same_seed_different_condition']:.1f} |")
    add("\nNone of the all-DN binned representations raises between-screen distance above different-seed within-screen distance. Per-pair Euclidean and cosine values are in [classification.json](classification.json).\n")
    add("## Grouped-by-seed classification\n")
    add("Every outer fold holds out **all conditions of one seed**. Aggregate and each fixed bin size use raw-count Euclidean nearest centroid. The prespecified nested temporal method selects a bin size using grouped inner validation on the 11 training seeds; only then is the outer seed predicted. Static classifiers use static training examples and the bin chosen on dynamic training data. Five-screen chance is 20%; four-screen chance excluding Title is 25%. Balanced accuracy equals ordinary accuracy because each class has 12 trials.\n")
    add("| All-DN readout | Dynamic five-screen | Static five-screen | Dynamic four-screen |\n| --- | ---: | ---: | ---: |")
    for steps in (200, 50, 20, 10, 5):
        label = "Aggregate" if steps == 200 else f"{steps}-step bins"
        add(f"| {label} | {c['dynamic_by_bin'][str(steps)]['accuracy']:.1%} | {c['static_by_bin'][str(steps)]['accuracy']:.1%} | {c['four_screen_by_bin'][str(steps)]['accuracy']:.1%} |")
    add(f"| **Nested bin choice** | **{temporal_all:.1%}** | **{c['nested_static_same_bins']['accuracy']:.1%}** | **{c['four_screen_nested']['accuracy']:.1%}** |")
    add(f"| Exploratory selected 20-step window | {early_window:.1%} | {c['nested_window_static_same_choices']['accuracy']:.1%} | {c['nested_window_four_screen']['accuracy']:.1%} |\n")
    add(f"Nested bin choice selected **{dict(sorted(Counter(x['bin_steps'] for x in c['nested_bin_choices']).items()))}** across 12 outer folds (modal **{bin_mode} steps**). Its **{temporal_all:.1%}** versus **{aggregate_all:.1%}** all-DN difference is only **{(temporal_all-aggregate_all)*100:+.1f} percentage points**. A fixed 10-step result is descriptive, since that fixed bin's test accuracy was inspected among candidates.\n")
    add("### All-DN aggregate confusion matrix\n")
    add(confusion_table(c["aggregate"]) + "\n")
    add("### Prespecified nested temporal-bin confusion matrix\n")
    add(confusion_table(c["nested_dynamic"]) + "\n")
    add("## Time localization\n")
    add("The single-window scan was exploratory. Each row uses only one 20-step window and the same grouped classifier. A follow-up selector chose among all ten windows using **inner training seeds only**; the outer test seed never chose a window.\n")
    add("| Steps | Five-screen accuracy | Between/within |\n| --- | ---: | ---: |")
    for item in c["time_windows"]:
        add(f"| {item['start_step']}–{item['end_step']} | {item['accuracy']:.1%} | {item['separation']['between_within_ratio']:.3f} |")
    window_choices = [item["bin_steps"] for item in c["nested_window_choices"]]
    wg = c["window_gain_vs_aggregate"]
    add(f"\nThe inner selector chose **steps 0–20 in all 12 outer folds** (`{window_choices}`), scoring **{early_window:.1%}**. The identical first frame in the static control also scored **{c['nested_window_static_same_choices']['accuracy']:.1%}**, so this is an **initial static visual response**, not evidence that changing game frames helps. Its +{wg['accuracy_difference']*100:.1f}-point gain over aggregate has seed-group bootstrap 95% interval **{wg['seed_group_bootstrap_95'][0]*100:+.1f} to {wg['seed_group_bootstrap_95'][1]*100:+.1f} points** and paired sign-flip p **{wg['paired_sign_flip_two_sided_p']:.3f}**. The advantage over aggregate is uncertain.\n")
    add("## DN population-size analysis: the main finding\n")
    add("Within **every training fold**, DNs were ranked by an ANOVA-like score: between-condition variation of training class means divided by within-condition residual variation, summed across bins. Top 500/250/100/50 were reselected independently for each held-out seed. Test labels and test spikes never entered the ranking. This is a screen-identity diagnostic, not an action decoder.\n")
    add("| Readout | All 1,314 | Top 500 | Top 250 | Top 100 | Top 50 |\n| --- | ---: | ---: | ---: | ---: | ---: |")
    for steps in (200, 50, 20, 10, 5):
        label = "Aggregate" if steps == 200 else f"{steps}-step bins"
        add("| " + label + " | " + " | ".join(f"{c['population_by_bin'][str(steps)][str(n)]['accuracy']:.1%}" for n in POPULATIONS) + " |")
    add(f"\nAggregate **population size** was also chosen by inner grouped validation for each outer fold. It chose **{dict(sorted(Counter(x['population'] for x in c['aggregate_nested_population_choices']).items()))}**, modal **{aggregate_pop_mode} DNs**, and achieved **{aggregate_selected:.1%} (59/60)**. The same selected sizes with static training reached **{c['aggregate_nested_population_static_same_sizes']['accuracy']:.1%}**; four-screen aggregate reached **{c['aggregate_nested_population_four_screen']['accuracy']:.1%}**. A separate nested bin-plus-population procedure chose modal **{temporal_pop_mode} DNs** and reached **{temporal_selected:.1%}** dynamic versus **{static_selected:.1%}** static. Thus **time binning is unnecessary for the strongest grouped result**.\n")
    add("### Aggregate selected-DN confusion matrix\n")
    add(confusion_table(c["aggregate_nested_population"]) + "\n")
    add("### Temporal-bin selected-DN confusion matrix\n")
    add(confusion_table(c["nested_population"]) + "\n")
    add(f"Temporal selected-DN accuracy minus aggregate selected-DN accuracy is **{gain['accuracy_difference']*100:+.1f} points**, with seed-group bootstrap 95% interval **{gain['seed_group_bootstrap_95'][0]*100:+.1f} to {gain['seed_group_bootstrap_95'][1]*100:+.1f}** and paired sign-flip p **{gain['paired_sign_flip_two_sided_p']:.3f}**. This does not support a temporal advantage. The median number of aggregate training-fold DNs with F>1 was **{statistics.median(c['aggregate_informative_dn_count_f_gt_1_training_folds']):.0f}**; that threshold is descriptive, not a validated cutoff.\n")
    add("| FlyWire ID | DN type | Side | Training folds in top 50 |\n| ---: | --- | :---: | ---: |")
    for item in c["aggregate_informative_dn_top20"][:15]:
        add(f"| {item['flywire_id']} | {item['cell_type']} | {item['side']} | {item['training_fold_top50_frequency']}/12 |")
    add("\nIDs/types come from MaleCNS metadata. The list summarizes training-fold selection frequency; no IDs were hardcoded into the classifier.\n")
    add("## Permutation significance\n")
    add("Condition labels were independently shuffled **within each seed**. The aggregate and nested-bin nulls repeated their grouped folds, with inner bin choice repeated for nested bins. The exploratory window null took the maximum of ten grouped window accuracies per shuffle, correcting its observed scan conservatively. The aggregate selected-DN null **refit training-only DN ranks** and took the maximum across five population sizes; the temporal selected-DN null also took the maximum across four temporal bin sizes. There were **1,000 shuffles** for each of the first four comparisons and **500** for the temporal selected-DN search. These nulls test screen identity above shuffled labels; they do not, by themselves, test temporal improvement over aggregate.\n")
    add("| Comparison | Observed | Null mean | Null 95th | Empirical p |\n| --- | ---: | ---: | ---: | ---: |")
    for key, label in (("aggregate", "All-DN aggregate"), ("nested_temporal", "All-DN nested temporal"),
                       ("nested_window", "Exploratory selected window"),
                       ("aggregate_selected_population", "Selected-DN aggregate"),
                       ("temporal_selected_population", "Selected-DN temporal")):
        p = permutations[key]
        add(f"| {label} | {p['observed']:.1%} | {p['null_mean']:.1%} | {p['null_95th']:.1%} | {p['empirical_p']:.4f} |")
    add("\nFull draws and shuffle descriptions are in [permutation-results.json](permutation-results.json). Both selected-DN readouts remain above conservative size-search nulls, while their paired comparison shows no time-binning gain.\n")
    add("## Static input and no-vision baseline\n")
    add(f"The strongest **static aggregate selected-DN** result was **{c['aggregate_nested_population_static_same_sizes']['accuracy']:.1%}**. The all-DN selected first window scored **{c['nested_window_static_same_choices']['accuracy']:.1%}** static and **{early_window:.1%}** with the changing frame sequence. These controls do not show a benefit from Experiment 3's short temporal visual sequence. The no-vision baseline was replayed with `eye_drive=None` and no injection; its full DN totals matched Experiment 3. Baseline separation below measures visual presence, not Pokémon screen identity.\n")
    add("| Screen | Aggregate mean-vector distance to no vision | Matched-seed distance |\n| --- | ---: | ---: |")
    for name in NAMES:
        x = c["no_vision_separation"]["200"][name]
        add(f"| {name.replace('_',' ')} | {x['mean_vector_euclidean']:.1f} | {x['median_matched_seed']:.1f} |")
    add("\n## Performance and debug artifacts\n")
    add(f"Median MaleCNS simulation step **{perf['median_step_ms']:.3f} ms**; DN capture overhead **{perf['median_dn_capture_ms']:.3f} ms per step**. Offline binning all 132 trials took **{min(perf['binning_ms'].values()):.1f}–{max(perf['binning_ms'].values()):.1f} ms per representation**. Grouped classifiers, distances and initial population analysis took **{perf['analysis_before_permutations_ms']/1000:.1f} s**; the initial two 1,000-shuffle tests took **{perf['permutations_ms']/1000:.1f} s**, window correction **{perf['window_selection_and_null_ms']/1000:.1f} s**, aggregate subset validation **{perf['aggregate_population_validation_ms']/1000:.1f} s**, and temporal subset validation **{perf['temporal_population_validation_ms']/1000:.1f} s**. These are offline analysis costs, not live inference costs.\n")
    for label, path in (("Classification by bin size", "classification-by-bin-size.svg"),
                        ("Accuracy versus DN population size", "accuracy-vs-dn-population.svg"),
                        ("Information versus simulation time", "information-vs-time.svg"),
                        ("Permutation null versus observed", "permutation-null-vs-observed.svg"),
                        ("All-DN confusion matrices", "confusion-matrices.svg"),
                        ("Selected-DN confusion matrices", "subset-confusion-matrices.svg"),
                        ("Selected-DN permutation nulls", "subset-permutation-null.svg")):
        add(f"- [{label}](../../captures/experiment-04/{path})")
    add("\n## Conclusion and next bottleneck\n")
    add(f"**FAIL for the temporal-structure hypothesis; primary bottleneck C.** An early DN response is visible, but selecting its window yields an uncertain +13.3-point advantage over aggregate all-DN counts. The much stronger result is **{aggregate_selected:.1%} grouped accuracy from training-selected aggregate DNs**, with permutation p **{permutations['aggregate_selected_population']['empirical_p']:.4f}** and **{c['aggregate_nested_population_four_screen']['accuracy']:.1%}** on the four non-Title screens. Time-resolved selected-DN readout scored **{temporal_selected:.1%}**. Pokémon screen information is present in the MaleCNS DN population, but the 200-step aggregation did **not** hide it; including many uninformative DNs in the distance/classifier hid it. A next experiment should validate the same subset across more seeds, screen instances and game states before any behavioral use. No reward, RL, RAM sensory input or action policy was added.\n")
    add("Reproduce from the repository root with `redfly-benchmark/.venv/bin/python run_temporal_dn_experiment.py` and `redfly-benchmark/.venv/bin/python analyze_temporal_dn.py`. Experiment 4 reads saved Experiment 3 vectors and does not need the ROM.\n")
    (RESULTS / "analysis.md").write_text("\n".join(lines))
    print(f"Experiment 3 original/grouped {aggregate_all:.1%}/{aggregate_all:.1%}; aggregate all DNs {aggregate_all:.1%}; best validated temporal selected DNs {temporal_selected:.1%}; chance 20%; selected-DN aggregate permutation p={permutations['aggregate_selected_population']['empirical_p']:.4f}; modal temporal bin {bin_mode}; best aggregate DN population {aggregate_pop_mode}; static temporal selected DNs {static_selected:.1%}; primary C; verdict FAIL")


def confusion_table(record):
    names = [name.replace("new_game_menu", "menu").replace("oak_dialogue", "dialogue") for name in record["labels"]]
    rows = ["| Actual \\ predicted | " + " | ".join(names) + " |",
            "| --- |" + " ---: |" * len(names)]
    for name, row in zip(names, record["confusion"]):
        rows.append("| " + name + " | " + " | ".join(map(str, row)) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
    from analyze_temporal_windows import main as window_main
    window_main()
    from validate_dn_subpopulation import main as population_main
    population_main()
    from validate_temporal_subpopulation import main as temporal_population_main
    temporal_population_main()
    complete = json.loads((RESULTS / "classification.json").read_text())
    nulls = json.loads((RESULTS / "permutation-results.json").read_text())
    make_plots(complete, nulls)
    make_report(complete, nulls)
