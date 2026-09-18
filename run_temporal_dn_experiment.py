"""Experiment 4 rerun of exact Experiment 3 inputs, recording 5-step DN bins."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cupy as cp
import numpy as np
from flybrain import FlyBrain

from flymon.spatiotemporal import SpatiotemporalConfig, SpatialProjection
from run_spatiotemporal_experiment import DATA, ROOT, NAMES, SEEDS, STEPS

SOURCE = ROOT / "results/experiment-03-spatiotemporal/trials.json"
RESULTS = ROOT / "results/experiment-04-temporal-dn"
CONDITIONS = ("baseline_none", *NAMES, *(f"{n}_static" for n in NAMES))
FINE_BIN = 5


def run_trial(brain, seed, vectors, projection, dn_slot, dn_count, config):
    brain.reset(seed=seed)
    cp.cuda.Stream.null.synchronize()
    bins = np.zeros((STEPS // FINE_BIN, dn_count), dtype=np.uint8)
    step_seconds = capture_seconds = grouping_seconds = 0.0
    frame_vectors = [None] * config.frames if vectors is None else vectors
    for frame_index, vector in enumerate(frame_vectors):
        t0 = time.perf_counter()
        pairs = [] if vector is None else [(cp.asarray(ids), voltage)
                                            for ids, voltage in projection.injection_pairs(vector)]
        grouping_seconds += time.perf_counter() - t0
        for step_in_frame in range(config.neural_steps_per_frame):
            t0 = time.perf_counter()
            fired = brain.step(eye_drive=None, inject=pairs)
            step_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()
            slots = dn_slot[fired]
            active = slots[slots >= 0]
            slot = (frame_index * config.neural_steps_per_frame + step_in_frame) // FINE_BIN
            bins[slot] += np.bincount(active, minlength=dn_count).astype(np.uint8)
            capture_seconds += time.perf_counter() - t0
    if not bool(cp.isfinite(brain.v).all().get()):
        raise RuntimeError("NaN/Inf in MaleCNS state")
    return bins, dict(step_ms=step_seconds * 1000 / STEPS,
                      dn_capture_ms=capture_seconds * 1000 / STEPS,
                      projection_grouping_ms=grouping_seconds * 1000 / config.frames)


def main():
    source_bytes = SOURCE.read_bytes()
    old = json.loads(source_bytes)
    config = SpatiotemporalConfig(**old["config"])
    if old["seeds"] != SEEDS or old["steps"] != STEPS or config.frames * config.neural_steps_per_frame != STEPS:
        raise RuntimeError("Experiment 3 protocol mismatch")
    brain = FlyBrain(data=DATA, device="cuda", batch=1)
    if brain.device != "cuda" or brain.n != 166700 or brain._W.nnz != 25582938:
        raise RuntimeError("Wrong MaleCNS or CUDA device")
    projection = SpatialProjection(brain, config)
    if projection.ids.tolist() != old["projection_neuron_indices"]:
        raise RuntimeError("Projection neuron mapping changed")
    dn = brain.cells(["descending_neuron"])
    if len(dn) != old["dn_count"] or not np.all(brain.superclass[dn] == "descending_neuron"):
        raise RuntimeError("DN metadata changed")
    slot = np.full(brain.n, -1, np.int32)
    slot[dn] = np.arange(len(dn), dtype=np.int32)
    vectors = {"baseline_none": None}
    for name in NAMES:
        vectors[name] = np.array(old["frame_metrics"][name]["dynamic_vectors"], np.float32)
        vectors[name + "_static"] = np.array(old["frame_metrics"][name]["static_vectors"], np.float32)
    for name, frames in vectors.items():
        if frames is not None and frames.shape != (config.frames, len(projection.ids)):
            raise RuntimeError(f"Saved Experiment 3 vector shape mismatch: {name}")
    brain.reset(seed=0)
    for _ in range(20):
        brain.step(eye_drive=None)
    data = np.empty((len(CONDITIONS), len(SEEDS), STEPS // FINE_BIN, len(dn)), np.uint8)
    metrics = {name: [] for name in CONDITIONS}
    for seed_idx, seed in enumerate(SEEDS):
        for condition_idx, name in enumerate(CONDITIONS):
            expected = np.array(old["trials"][name][seed_idx]["dn_counts"], np.int32)
            for attempt in range(1, 9):
                bins, timing = run_trial(brain, seed, vectors[name], projection, slot, len(dn), config)
                if np.array_equal(bins.sum(axis=0, dtype=np.int32), expected):
                    break
                print(f"CUDA replay mismatch: {name} seed {seed} attempt {attempt}; retrying", flush=True)
            else:
                raise RuntimeError(f"Aggregate DN mismatch after 8 attempts: {name} seed {seed}")
            if int(bins.sum()) != old["trials"][name][seed_idx]["dn_spikes"]:
                raise RuntimeError(f"Total DN spike mismatch: {name} seed {seed}")
            data[condition_idx, seed_idx] = bins
            metrics[name].append(dict(seed=seed, attempts=attempt, **timing))
        print("Matched Experiment 3 exactly for seed", seed, flush=True)
    meta = np.load(DATA / "brain.npz")
    RESULTS.mkdir(parents=True, exist_ok=True)
    target = RESULTS / "dn-bins-5.npz"
    np.savez_compressed(target, counts=data, conditions=np.array(CONDITIONS), seeds=np.array(SEEDS),
                        dn_indices=dn.astype(np.int32), flywire_ids=meta["ids"][dn],
                        cell_types=brain.cell_type[dn], side=brain.side[dn])
    (RESULTS / "capture-metrics.json").write_text(json.dumps(dict(source_experiment3_sha256=hashlib.sha256(source_bytes).hexdigest(),
        model=old["model"], device="cuda", config=asdict(config), fine_bin_steps=FINE_BIN,
        exact_aggregate_match_for_all_trials=True, replay_policy="up to 8 attempts; retain only runs matching Experiment 3 per-DN totals", conditions=CONDITIONS,
        seeds=SEEDS, dn_count=len(dn), timings=metrics), indent=2) + "\n")
    print("Saved", target, "shape", data.shape, flush=True)


if __name__ == "__main__":
    main()
