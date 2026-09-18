"""Reconstruct Experiment 2's exact class-ordered injection vectors from saved data."""
from __future__ import annotations

import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/experiment-02-feature-detectors/trials.json"
TARGET = ROOT / "results/experiment-03-spatiotemporal/experiment-02-collapse.json"


def main():
    data = json.loads(SOURCE.read_text())
    order = [("LPLC2", "loom", "L"), ("LPLC2", "loom", "R"),
             ("LC10a", "chase", "L"), ("LC10a", "chase", "R")]
    vectors = {}
    stats = {}
    for name, metrics in data["frame_metrics"].items():
        stats[name] = {}
        for phase in ("onset", "steady"):
            amounts = metrics[phase + "_amounts"]
            vector = np.concatenate([
                np.full(data["target_counts"][kind][side], amounts[channel + side], np.float32)
                for kind, channel, side in order])
            vectors[name, phase] = vector
            stats[name][phase] = dict(sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
                                      min=float(vector.min()), max=float(vector.max()),
                                      mean=float(vector.mean()), std=float(vector.std()),
                                      nonzero=int(np.count_nonzero(vector)))
    distances = {}
    for a, b in combinations(data["frame_metrics"], 2):
        distances[f"{a}__{b}"] = {
            phase: float(np.linalg.norm(vectors[a, phase] - vectors[b, phase]))
            for phase in ("onset", "steady")}
    TARGET.write_text(json.dumps(dict(class_order=[f"{kind}_{side}" for kind, _, side in order],
                                      neuron_count=460, stats=stats, pairwise_distances=distances), indent=2) + "\n")
    print("Saved", TARGET)


if __name__ == "__main__":
    main()
