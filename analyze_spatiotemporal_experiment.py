"""Rebuild Experiment 3's Markdown analysis from saved trials; never edits prior data."""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "experiment-03-spatiotemporal"
NAMES = ["intro", "title", "new_game_menu", "oak_dialogue", "bedroom"]
DISPLAY = {"intro": "Intro", "title": "Title", "new_game_menu": "New Game menu",
           "oak_dialogue": "Oak dialogue", "bedroom": "Bedroom"}


def avg(values):
    return float(np.mean(list(values)))


def distance_summary(comparison):
    within = avg(comparison["within"][n]["median_euclidean"] for n in NAMES)
    between = avg(comparison["between"][f"{a}__{b}"]["median_cross_trial_euclidean"]
                  for a, b in combinations(NAMES, 2))
    return within, between, between / within


def classification(trials, names=NAMES):
    x = np.array([[t["dn_counts"] for t in trials[name]] for name in names], np.float32)
    confusion = np.zeros((len(names), len(names)), int)
    for held in range(x.shape[1]):
        centroids = (x.sum(axis=1) - x[:, held]) / (x.shape[1] - 1)
        for actual in range(len(names)):
            predicted = int(np.argmin(np.linalg.norm(centroids - x[actual, held], axis=1)))
            confusion[actual, predicted] += 1
    return np.trace(confusion) / confusion.sum(), confusion


def permutation_p(trials, iterations=1000):
    x = np.array([[t["dn_counts"] for t in trials[name]] for name in NAMES], np.float32)
    rng = np.random.default_rng(20260918)

    def score(a):
        sums = a.sum(axis=1)
        correct = 0
        for held in range(a.shape[1]):
            centroids = (sums - a[:, held]) / (a.shape[1] - 1)
            delta = centroids[:, None, :] - a[None, :, held, :]
            predicted = np.argmin(np.einsum("ijk,ijk->ij", delta, delta), axis=0)
            correct += int(np.count_nonzero(predicted == np.arange(a.shape[0])))
        return correct / (a.shape[0] * a.shape[1])

    observed = score(x)
    null = []
    for _ in range(iterations):
        shuffled = np.stack([x[rng.permutation(len(NAMES)), seed]
                             for seed in range(x.shape[1])], axis=1)
        null.append(score(shuffled))
    return observed, (1 + sum(v >= observed for v in null)) / (iterations + 1), avg(null), float(np.quantile(null, 0.95))


def pair_matrix(comparison, metric, precision=1):
    lines = ["| | " + " | ".join(DISPLAY[n] for n in NAMES) + " |",
             "| --- |" + " ---: |" * len(NAMES)]
    for a in NAMES:
        values = []
        for b in NAMES:
            if a == b:
                value = comparison["within"][a]["median_euclidean"] if metric == "median_cross_trial_euclidean" else (1.0 if metric == "mean_vector_cosine" else 0.0)
            else:
                key = f"{a}__{b}" if NAMES.index(a) < NAMES.index(b) else f"{b}__{a}"
                value = comparison["between"][key][metric]
            values.append(f"{value:.{precision}f}")
        lines.append("| " + DISPLAY[a] + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


def encoder_matrix(distances):
    lines = ["| | " + " | ".join(DISPLAY[n] for n in NAMES) + " |",
             "| --- |" + " ---: |" * len(NAMES)]
    for a in NAMES:
        values = []
        for b in NAMES:
            key = f"{a}__{b}" if NAMES.index(a) < NAMES.index(b) else f"{b}__{a}"
            values.append("0.0" if a == b else f"{distances[key]:.1f}")
        lines.append("| " + DISPLAY[a] + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


def stats(trials, key):
    values = [trial[key] for trial in trials]
    return f"{np.mean(values):,.0f} ± {np.std(values, ddof=1):,.0f}"


def main():
    d = json.loads((RESULTS / "trials.json").read_text())
    exp1 = json.loads((ROOT / "results/experiment-01-photoreceptors/trials.json").read_text())
    exp2 = json.loads((ROOT / "results/experiment-02-feature-detectors/trials.json").read_text())
    dn_w, dn_b, dn_ratio = distance_summary(d["dn_comparison"])
    vp_w, vp_b, vp_ratio = distance_summary(d["vp_comparison"])
    static_w, static_b, static_ratio = distance_summary(d["static_dn_comparison"])
    p1_w, p1_b, p1_ratio = distance_summary(exp1["comparison"])
    p2_w, p2_b, p2_ratio = distance_summary(exp2["comparison"])
    observed, p, null_mean, null95 = permutation_p(d["trials"])
    p1_class, _ = classification(exp1["trials"])
    p2_class, _ = classification(exp2["trials"])
    four = [n for n in NAMES if n != "title"]
    four2, _ = classification(exp2["trials"], four)
    four3, _ = classification(d["trials"], four)
    lines = []
    add = lines.append
    add("# Experiment 3 — 2-D spatial + temporal projection into MaleCNS\n")
    add("**Verdict: WEAK.** All five injected sequences are distinct and visual-projection neurons separate them strongly. DN trial distances do not separate beyond stochastic variation: between **%.1f**, within **%.1f**, ratio **%.3f**. Held-out five-way DN classification is **%.1f%%** versus **20%%** chance, but it equals Experiment 2's aggregate accuracy and has concentrated confusions. Temporal sequences did not outperform the static control.\n" % (dn_b, dn_w, dn_ratio, 100 * observed))
    add("## Experiment 2 collapse diagnosis\n")
    add("Experiment 2 reduced each full frame to the largest dark component `(dx, sqrt(area))`. Upstream [`FeatureDetectors`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/eyes.py) then selected only L/R side and capped both active channels at 0.8. Intro, New Game menu, Oak dialogue and bedroom all selected L; their 460-element onset hashes are `d3777aff4d9c80dd7b7fad2f75c8fae1e2e1fcdf0ecdb20de86d66ce5d53999c`, and steady hashes are `d97869e217b46f2ae5818022fc70de7073c92a8da3fc744eda94137d9865098d`. Pairwise distance among those four is exactly **0** in both phases. Their onset vector min/max/mean/SD/nonzero is **0/0.8/0.3983/0.4000/229**; steady is **0/0.8/0.2348/0.3643/135**. Title's opposite-side onset and steady hashes differ, with distances **17.16** and **13.27**. The inputs had distinct images and component geometries, so the collapse occurred in single-object spatial compression, L/R aggregation, and detector saturation, before the CNS. See the preregistered [design and diagnosis](design.md).\n")
    collapse = json.loads((RESULTS / "experiment-02-collapse.json").read_text())
    add("| Experiment 2 screen | Phase | SHA-256 prefix | Min/max | Mean/SD | Nonzero of 460 |\n| --- | --- | --- | --- | --- | ---: |")
    for name in NAMES:
        for phase in ("onset", "steady"):
            x = collapse["stats"][name][phase]
            add(f"| {DISPLAY[name]} | {phase} | `{x['sha256'][:16]}…` | {x['min']:.2f}/{x['max']:.2f} | {x['mean']:.4f}/{x['std']:.4f} | {x['nonzero']} |")
    add("\nPairwise exact distances for all onset and steady vectors are in [the collapse diagnostic](experiment-02-collapse.json). All six pairings among the four left-side screens are zero; each Title pairing is 17.16 onset and 13.27 steady. Full hashes are retained in that JSON.\n")
    add("## Spatial and temporal encoder\n")
    add("Each actual 144×160 RGBA PyBoy frame is converted to Rec.709 luminance, then mean-pooled in nonoverlapping 8×8 blocks to an **18×20 two-dimensional grid**. Each cell retains x and y. LC10a receives `0.55 × (1 − luminance)` as a dark-object/contrast proxy; LPLC2 receives `0.75 × abs(current − previous luminance)` as a change/looming-like proxy. First-frame change is zero. Voltage is rounded to 0.05 increments and capped at 0.8; observed gain limits prevent detector saturation. These are explicit engineering approximations, not a learned model or a claim that static game pixels are natural fly prey or true looming.\n")
    add("[`Eyes`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/eyes.py) supplies the biological luminance-plus-change idea, but accepts only a 1-D panorama. Upstream `FeatureDetectors` chooses biological classes but offers only one L/R object value. [`FlyBrain.step`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/brain.py) accepts the per-group voltage injections used here. `reservoir.Trace` was not used because exact firing-count vectors are the endpoint. MaleCNS metadata provides no projection-neuron receptive-field coordinates or visual elevation. Soma positions were not treated as receptive fields. We evenly assigned distinct grid samples within each half-screen to metadata-selected cells in stable neuron order. This preserves source x/y in a **surrogate mapping**, not biological retinotopy. The exact 460 indices and grid coordinates are in `trials.json`.\n")
    add("| Class/side | Neurons | Grid source | Feature |\n| --- | ---: | --- | --- |")
    for key, count in d["projection_counts"].items():
        kind, side = key.rsplit("_", 1)
        add(f"| {kind} {side} | {count} | {side} half of 18×20 | {'absolute change' if kind == 'LPLC2' else 'darkness'} |")
    add("\n**460 unique visual-projection neurons** receive potential drive; **1,314 metadata-selected descending neurons** are measured. LC4/LPLC1 were excluded because no validated threat/projectile signal was extracted. The MaleCNS v1.0 dataset has 166,700 neurons and 25,582,938 connections.\n")
    add("Ten frames were captured four PyBoy ticks apart, then presented for 20 MaleCNS steps each: **200 steps per trial**, matching Experiments 1/2. The initial checkpoint matches the Experiment 1 framebuffer pixel-for-pixel. Bedroom begins from canonical `states/bedroom.state` (SHA-256 `%s`) and holds Right for eight game frames between captured frames 3 and 5. Other conditions use no buttons during capture. Capture happens once; its exact sequence is reused for all 12 seeds (`101..112`). The static control repeats each condition's initial grid for every window; no-vision injects nothing and uses `eye_drive=None`. No Pokémon RAM or ROM copy is used.\n" % d["bedroom_state"]["sha256"])
    add("## Encoder separation and debug artifacts\n")
    add("The exact quantized injected 460-cell vector for every frame is stored in `trials.json`, with SHA-256, min/max/mean/SD and nonzero count. This was checked **before** CNS trials. There are **%d unique condition sequences**. Repeated trials replay identical captured sequences, so encoder within-condition distance is **0**; its between/within ratio is **undefined**, rather than infinite or an estimated stochastic separation. Mean between-condition encoder distance is **%.2f** (Euclidean over 10×460 values).\n" % (d["encoder_comparison"]["unique_outputs"], d["encoder_comparison"]["mean_between_condition_distance"]))
    add("### Pairwise encoder Euclidean distance\n")
    add(encoder_matrix(d["encoder_comparison"]["between_condition_distance"]) + "\n")
    add("| Condition | Unique framebuffer frames | Unique injected windows | Dynamic–static encoder distance | Vector min/max | Mean/SD | Nonzero entries over 10 windows | Debug strip |\n| --- | ---: | ---: | ---: | --- | --- | ---: | --- |")
    for name in NAMES:
        m = d["frame_metrics"][name]
        flat = np.array(m["dynamic_vectors"], np.float32).ravel()
        distinct = len({v["sha256"] for v in m["dynamic_stats"]})
        add(f"| {DISPLAY[name]} | {m['unique_frames']} | {distinct} | {d['encoder_comparison']['dynamic_static_distance'][name]:.2f} | {flat.min():.2f}/{flat.max():.2f} | {flat.mean():.3f}/{flat.std():.3f} | {np.count_nonzero(flat)} | [contact sheet](../../captures/experiment-03/{name}-contact-sheet.png) |")
    add("\nEach condition's [capture directory](../../captures/experiment-03/) contains all ten original frames, downsampled grids, temporal-difference maps, darkness feature maps and projected-drive images. In projected-drive images, red is LC10a darkness and blue is LPLC2 change; the contact sheet stacks original, grid, change and projection by time. Title and New Game menu did not animate within the 36-elapsed-game-frame capture, so their dynamic and static inputs are identical.\n")
    add("## Neural response and information-loss stage\n")
    add("Means ± sample SD across 12 seeds, counting spikes across 200 steps. Every trial passed a finite-state and spike-accounting check. The no-vision DN vector matched Experiment 1 exactly for each seed. Per-window CNS/visual-projection/DN counts and all per-neuron DN/visual-projection vectors are in `trials.json`.\n")
    add("| Condition | CNS spikes | Visual-projection spikes | DN spikes |\n| --- | ---: | ---: | ---: |")
    for name in ["baseline_none", *NAMES, *(f"{n}_static" for n in NAMES)]:
        label = "No vision" if name == "baseline_none" else DISPLAY[name.removesuffix("_static")] + (" static" if name.endswith("_static") else " temporal")
        group = d["trials"][name]
        add(f"| {label} | {stats(group, 'cns_spikes')} | {stats(group, 'visual_projection_spikes')} | {stats(group, 'dn_spikes')} |")
    add("\nVisual-projection firing-count vectors strongly distinguish all conditions: mean within-screen Euclidean **%.1f**, mean between-screen **%.1f**, ratio **%.2f**, and held-out nearest-centroid accuracy **%.1f%%**. Thus the encoder distinction survives through the stimulated biological projection cells.\n" % (vp_w, vp_b, vp_ratio, 100 * d["vp_classifier"]["accuracy"]))
    add("### Projection-neuron cross-trial Euclidean distance\n")
    add(pair_matrix(d["vp_comparison"], "median_cross_trial_euclidean") + "\n")
    add("DN vectors are less separated: mean within-screen **%.1f**, between-screen **%.1f**, ratio **%.3f**. Pairwise mean-vector cosines remain close to one; total DN spike differences alone cannot establish identity separation.\n" % (dn_w, dn_b, dn_ratio))
    add("### DN cross-trial Euclidean distance (diagonal = within-screen)\n")
    add(pair_matrix(d["dn_comparison"], "median_cross_trial_euclidean") + "\n")
    add("### DN mean-vector cosine\n")
    add(pair_matrix(d["dn_comparison"], "mean_vector_cosine", 5) + "\n")
    add("| Pair | Mean-vector Euclidean | Mean DN total difference (first − second) |\n| --- | ---: | ---: |")
    for a, b in combinations(NAMES, 2):
        item = d["dn_comparison"]["between"][f"{a}__{b}"]
        add(f"| {DISPLAY[a]} / {DISPLAY[b]} | {item['mean_vector_euclidean']:.1f} | {item['dn_total_mean_difference']:+.1f} |")
    add("\nThe stage classification is **D: some distinctions survive into descending neurons**, but weakly and mainly as a few condition clusters. Stage A is excluded by five distinct encoder outputs; B is excluded by 100% projection-vector held-out classification. Stage C as a complete DN collapse is excluded by significant held-out DN classification, although stochastic DN distance still exceeds aggregate between-screen distance. This is **not** a PASS.\n")
    add("## Static control and no-vision baseline\n")
    add("Static DN within/between distance is **%.1f/%.1f** (ratio **%.3f**), with **%.1f%%** held-out accuracy. It is as good as or slightly better than temporal input, so this experiment does **not** show a temporal benefit. Dynamic–static encoder distance is 0 for Title and menu, 0.48 for dialogue, 4.41 for Intro and 6.38 for bedroom.\n" % (static_w, static_b, static_ratio, 100 * d["static_dn_classifier"]["accuracy"]))
    add("| Condition | DN mean-vector distance from no vision | Matched-seed DN distance from no vision | DN mean-vector dynamic–static distance |\n| --- | ---: | ---: | ---: |")
    base = np.array([t["dn_counts"] for t in d["trials"]["baseline_none"]], float)
    for name in NAMES:
        visual = np.array([t["dn_counts"] for t in d["trials"][name]], float)
        static = np.array([t["dn_counts"] for t in d["trials"][name + "_static"]], float)
        add(f"| {DISPLAY[name]} | {np.linalg.norm(visual.mean(0)-base.mean(0)):.1f} | {statistics.median(np.linalg.norm(visual-base,axis=1)):.1f} | {np.linalg.norm(visual.mean(0)-static.mean(0)):.1f} |")
    add("\n## Held-out classification diagnostic\n")
    add("Leave-one-**seed**-out nearest centroid uses 11 seeds per condition for training and predicts the five held-out trials for the twelfth seed, repeated over all 12 seeds. No policy or action decoder is fitted. Five-way chance is **20%%**. Observed DN accuracy is **%.1f%% (24/60)**; a 1,000-shuffle label permutation within each seed gives p = **%.4f**, null mean %.1f%%, null 95th percentile %.1f%%. This shows recoverable DN information, but modest accuracy and confusions prevent a PASS.\n" % (100 * observed, p, 100 * null_mean, 100 * null95))
    add("| Actual \\ predicted | " + " | ".join(DISPLAY[n] for n in NAMES) + " |")
    add("| --- |" + " ---: |" * len(NAMES))
    for name, row in zip(NAMES, d["dn_classifier"]["confusion"]):
        add("| " + DISPLAY[name] + " | " + " | ".join(map(str, row)) + " |")
    add("\nFor the four screens merged by Experiment 2 (excluding Title), held-out DN accuracy rises from **%.1f%%** (chance 25%%; Experiment 2) to **%.1f%%**. Full five-way accuracy remains **40.0%%** in both experiments. This is partial recovery of screen information, not robust five-screen discrimination.\n" % (100 * four2, 100 * four3))
    add("## Performance\n")
    add("Frame capture includes PyBoy ticks plus framebuffer copy. Spatial preprocessing includes grayscale and grid pooling; temporal feature calculation and projection encoding were measured separately per captured frame. Injection-group construction is amortized per 20-step window. `FlyBrain.step` includes grouped voltage application and CUDA synchronization; DN aggregation also counts the 460 projection cells. Medians are shown; runs were not simultaneous with previous experiments.\n")
    add("| Condition | Capture/frame ms | Spatial/frame ms | Change/frame ms | Encode/frame ms | Group/window ms | MaleCNS step ms | Aggregation/step ms |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in NAMES:
        m = d["frame_metrics"][name]
        group = d["trials"][name]
        add(f"| {DISPLAY[name]} | {statistics.median(d['capture_ms'][name]):.3f} | {statistics.median(m['spatial_preprocess_ms']):.3f} | {m['temporal_feature_ms']:.3f} | {m['projection_encode_ms']:.3f} | {statistics.median(t['projection_grouping_ms'] for t in group):.3f} | {statistics.median(t['step_ms'] for t in group):.3f} | {statistics.median(t['dn_aggregate_ms'] for t in group):.3f} |")
    add("\nThe visual step ranged about **3.0–5.8 ms**, above the **2.37 ms** Experiment 1 and **2.42 ms** Experiment 2 visual steps and the earlier bare CUDA **~1.93 ms**. The 0.05-V grouping limits injection calls but still makes this surrogate much costlier than the prior one-value-per-side detector. No optimization was attempted.\n")
    add("## Comparison with Experiments 1 and 2\n")
    add("| Metric | Exp. 1 photoreceptors | Exp. 2 feature detector | Exp. 3 spatial + temporal |\n| --- | ---: | ---: | ---: |")
    add(f"| Within-screen DN distance | {p1_w:.1f} | {p2_w:.1f} | {dn_w:.1f} |")
    add(f"| Between-screen DN distance | {p1_b:.1f} | {p2_b:.1f} | {dn_b:.1f} |")
    add(f"| Between/within ratio | {p1_ratio:.3f} | {p2_ratio:.3f} | {dn_ratio:.3f} |")
    add(f"| Held-out five-way DN accuracy | {100*p1_class:.1f}% | {100*p2_class:.1f}% | {100*observed:.1f}% |")
    add("| Distinct direct feature injections | 5 receptor drives | 2 projection patterns | 5 projection sequences |")
    add("| Verdict | FAIL | WEAK | **WEAK** |\n")
    add("Experiment 3 materially repairs the **encoder/projection collapse** and improves four-screen DN recoverability, but does **not** materially improve aggregate five-screen DN accuracy or the primary between/within distance criterion. Its ratio (0.957) remains below one. Most measured improvement comes from spatial coding, as the static control matches temporal performance. The previous experiment files and results were not modified.\n")
    add("## Conclusion and recommended Experiment 4\n")
    add("**WEAK.** Case **D**, with limited downstream survival: strong distinct encoder and projection signals, a statistically detectable but low-accuracy DN identity signal, and no aggregate DN distance advantage over stochastic variation. Title/menu had no temporal changes in the chosen window. These observations do not justify RL or action decoding.\n")
    add("Experiment 4 should isolate **temporal contribution** with longer, deliberately motion-rich but fixed visual sequences and a matched static control. It should also test an independently justified retinotopic mapping if MaleCNS visual receptive-field metadata or connectome-derived optic-column correspondence becomes available; this surrogate neuron order must not be treated as anatomy. Keep held-out classification and both projection/DN variance gates.\n")
    add("Reproduce with `POKEMON_ROM=/local/path/pokemon-red.gb redfly-benchmark/.venv/bin/python run_spatiotemporal_experiment.py` followed by `redfly-benchmark/.venv/bin/python analyze_spatiotemporal_experiment.py`. The ROM stays external.\n")
    (RESULTS / "analysis.md").write_text("\n".join(lines))
    print(f"Experiment 1: FAIL; Experiment 2: WEAK; Experiment 3: WEAK\nEncoder between/within: undefined (within=0); DN between/within: {dn_ratio:.3f}\nHeld-out DN classification: {100*observed:.1f}% versus 20% chance; information-loss stage: D (limited DN survival)\nSpatial mapping repaired encoder collapse; no convincing aggregate five-screen or temporal gain")


if __name__ == "__main__":
    main()
