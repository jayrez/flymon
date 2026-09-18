"""Experiment 2: static Pokémon frames into upstream FlyBrain FeatureDetectors.

Experiment 1 analysis and trial files are never written by this script.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cupy as cp
import numpy as np
from flybrain import FlyBrain
from flybrain.eyes import FeatureDetectors

from flymon.features import (FeatureConfig, detector_for, extract_from_gray,
                             save_feature_artifacts, target_populations)
from flymon.vision import preprocess
from run_visual_experiment import DATA, ROOT, SEEDS, STEPS, capture_conditions, compare

CAPTURES = ROOT / "captures" / "experiment-02"
RESULTS = ROOT / "results" / "experiment-02-feature-detectors"
EXPERIMENT_1_CAPTURES = ROOT / "captures" / "vision"
STATE = ROOT / "states" / "bedroom.state"


def prepare_conditions(frames, brain):
    observations = {}
    metrics = {}
    CAPTURES.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        # The same framebuffer must be used in both experiments, not just the
        # same scene name. The bedroom is freshly loaded from its canonical state.
        old = EXPERIMENT_1_CAPTURES / f"{name}-original.png"
        from PIL import Image
        previous_frame = np.asarray(Image.open(old).convert("RGBA"))
        if not np.array_equal(frame, previous_frame):
            raise RuntimeError(f"Experiment 2 frame differs from Experiment 1: {name}")
        t0 = time.perf_counter()
        gray, _ = preprocess(frame)
        preprocess_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        observation = extract_from_gray(gray, FeatureConfig())
        extraction_ms = (time.perf_counter() - t0) * 1000
        if observation.opp is None:
            raise RuntimeError(f"No feature object selected for {name}")
        detector = detector_for(brain, observation)
        detector.inject(opp=observation.opp)
        onset = dict(detector.last)
        detector.inject(opp=observation.opp)
        steady = dict(detector.last)
        save_feature_artifacts(CAPTURES / name, frame, observation, onset, steady)
        observations[name] = observation
        metrics[name] = dict(preprocess_ms=preprocess_ms, feature_extraction_ms=extraction_ms,
                             source_frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                             area=observation.area, bbox=observation.bbox,
                             center_x=observation.center_x, dx=observation.dx,
                             size=observation.size, side=observation.side,
                             dark_fraction=float(observation.dark_mask.mean()),
                             onset_amounts=onset, steady_amounts=steady)
    return observations, metrics


def run_trial(brain, seed, observation, dn_slot, target_masks):
    brain.reset(seed=seed)
    detector = None if observation is None else detector_for(brain, observation)
    cp.cuda.Stream.null.synchronize()
    dn_counts = np.zeros(int((dn_slot >= 0).sum()), np.int32)
    counts = dict(cns_spikes=0, feature_projection_spikes=0, dn_spikes=0,
                  class_spikes={name: 0 for name in target_masks if name != "all"})
    projection_seconds = step_seconds = aggregate_seconds = 0.0
    for _ in range(STEPS):
        t0 = time.perf_counter()
        injection = () if detector is None else detector.inject(opp=observation.opp)
        projection_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        fired = brain.step(eye_drive=None, inject=injection)
        step_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        counts["cns_spikes"] += len(fired)
        counts["feature_projection_spikes"] += int(target_masks["all"][fired].sum())
        for name in counts["class_spikes"]:
            counts["class_spikes"][name] += int(target_masks[name][fired].sum())
        slots = dn_slot[fired]
        active = slots[slots >= 0]
        counts["dn_spikes"] += len(active)
        dn_counts += np.bincount(active, minlength=len(dn_counts)).astype(np.int32)
        aggregate_seconds += time.perf_counter() - t0
    if not bool(cp.isfinite(brain.v).all().get()):
        raise RuntimeError("NaN/Inf in neural state")
    if int(dn_counts.sum()) != counts["dn_spikes"]:
        raise RuntimeError("DN spike accounting mismatch")
    return dict(seed=seed, **counts, dn_counts=dn_counts.tolist(),
                feature_projection_ms=projection_seconds * 1000 / STEPS,
                step_ms=step_seconds * 1000 / STEPS,
                dn_aggregate_ms=aggregate_seconds * 1000 / STEPS)


def main():
    reference = json.loads((ROOT / "results" / "experiment-01-photoreceptors" / "trials.json").read_text())
    if reference["seeds"] != SEEDS or reference["steps"] != STEPS:
        raise RuntimeError("Experiment 1 seed/step protocol changed")
    brain = FlyBrain(data=DATA, device="cuda", batch=1)
    if brain.device != "cuda" or brain.n != 166700 or brain._W.nnz != 25582938:
        raise RuntimeError("Wrong device or MaleCNS data")
    dn = brain.cells(["descending_neuron"])
    if len(dn) != reference["dn_count"] or not np.all(brain.superclass[dn] == "descending_neuron"):
        raise RuntimeError("DN population differs from Experiment 1")
    detector = FeatureDetectors(brain)
    populations = target_populations(detector)
    selected = np.concatenate(list(populations.values()))
    if len(np.unique(selected)) != len(selected) or not np.all(brain.superclass[selected] == "visual_projection"):
        raise RuntimeError("Feature target overlap or metadata mismatch")
    counts_by_side = {kind: {side: len(detector.cells[channel][side]) for side in "LR"}
                      for kind, channel in (("LPLC2", "loom"), ("LC10a", "chase"),
                                            ("LC4", "threat"), ("LPLC1", "shot"))}
    print("Feature neuron counts:", counts_by_side, flush=True)
    print("Unique selected targets:", len(selected), "DNs:", len(dn), flush=True)
    frames = capture_conditions()
    if set(frames) != set(reference["frame_metrics"]):
        raise RuntimeError("Visual conditions differ from Experiment 1")
    observations, frame_metrics = prepare_conditions(frames, brain)
    target_masks = {}
    for kind, ids in populations.items():
        mask = np.zeros(brain.n, bool)
        mask[ids] = True
        target_masks[kind] = mask
    mask = np.zeros(brain.n, bool)
    mask[selected] = True
    target_masks["all"] = mask
    dn_slot = np.full(brain.n, -1, np.int32)
    dn_slot[dn] = np.arange(len(dn), dtype=np.int32)
    brain.reset(seed=0)
    warm = detector_for(brain, next(iter(observations.values())))
    for _ in range(20):
        brain.step(eye_drive=None, inject=warm.inject(opp=next(iter(observations.values())).opp))
    conditions = {"baseline_none": None, **observations}
    trials = {name: [] for name in conditions}
    for seed in SEEDS:
        for name, observation in conditions.items():
            trials[name].append(run_trial(brain, seed, observation, dn_slot, target_masks))
        print(f"Completed seed {seed}", flush=True)
    output = dict(model="flybrain 0.1.0 MaleCNS v1.0", pathway="FeatureDetectors.inject",
                  device="cuda", steps=STEPS, seeds=SEEDS, dn_count=len(dn),
                  target_counts=counts_by_side, selected_target_count=len(selected),
                  config=FeatureConfig().__dict__,
                  bedroom_state={"path": str(STATE.relative_to(ROOT)),
                                 "sha256": hashlib.sha256(STATE.read_bytes()).hexdigest()},
                  frame_metrics=frame_metrics, trials=trials, comparison=compare(trials))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "trials.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
