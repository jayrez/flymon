"""First framebuffer -> photoreceptor -> MaleCNS response experiment.

No game RAM, rewards, action policy, or trained decoder are used.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

import cupy as cp
import numpy as np
from PIL import Image
from flybrain import FlyBrain

from flymon.emulator import PokemonEmulator
from flymon.vision import PanoramaEyes, VisionConfig, preprocess, save_debug_artifacts

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "redfly-benchmark" / "data"
CAPTURES = ROOT / "captures" / "vision"
RESULTS = ROOT / "results"
STEPS = 200
SEEDS = list(range(101, 113))


def capture_conditions() -> dict[str, np.ndarray]:
    """Reach four distinct built-in screens with only timed intro and Start/A taps."""
    frames = {}
    with PokemonEmulator() as game:
        game.tick(900)
        frames["intro"] = game.framebuffer()
        game.tick(600)
        frames["title"] = game.framebuffer()
        game.tap("start", 3, 3)
        game.tick(90)
        game.tap("a", 3, 3)
        game.tick(90)
        frames["new_game_menu"] = game.framebuffer()
        for _ in range(5):
            game.tap("a", 3, 3)
            game.tick(90)
        frames["oak_dialogue"] = game.framebuffer()
    return frames


def prepare_drives(frames, brain):
    outputs = {}
    timings = {}
    CAPTURES.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        t0 = time.perf_counter()
        gray, panorama = preprocess(frame)
        preprocess_ms = (time.perf_counter() - t0) * 1000
        encoder = PanoramaEyes(brain.azimuth, VisionConfig())
        encoder.reset()
        t0 = time.perf_counter()
        onset = encoder.drive(panorama)
        steady = encoder.drive(panorama)
        project_ms = (time.perf_counter() - t0) * 1000 / 2
        if onset.shape != (len(brain.visual),) or steady.shape != onset.shape:
            raise RuntimeError("Receptor drive length mismatch")
        save_debug_artifacts(CAPTURES / name, frame, gray, panorama,
                             brain.azimuth, onset, steady)
        outputs[name] = (onset, steady)
        timings[name] = dict(preprocess_ms=preprocess_ms, projection_ms=project_ms,
                             gray_mean=float(gray.mean()), panorama_min=float(panorama.min()),
                             panorama_max=float(panorama.max()),
                             onset_drive_mean=float(onset.mean()),
                             steady_drive_mean=float(steady.mean()))
    return outputs, timings


def run_trial(brain, seed, drive_pair, dn_slot, visual_mask):
    brain.reset(seed=seed)
    cp.cuda.Stream.null.synchronize()
    dn_counts = np.zeros(int((dn_slot >= 0).sum()), np.int32)
    total_spikes = visual_spikes = dn_spikes = 0
    step_seconds = aggregate_seconds = 0.0
    for step in range(STEPS):
        drive = None if drive_pair is None else drive_pair[0 if step == 0 else 1]
        t0 = time.perf_counter()
        fired = brain.step(eye_drive=drive)
        step_seconds += time.perf_counter() - t0  # step() transfers spikes and synchronizes CUDA
        t0 = time.perf_counter()
        total_spikes += len(fired)
        visual_spikes += int(visual_mask[fired].sum())
        slots = dn_slot[fired]
        active = slots[slots >= 0]
        dn_spikes += len(active)
        dn_counts += np.bincount(active, minlength=len(dn_counts)).astype(np.int32)
        aggregate_seconds += time.perf_counter() - t0
    if not bool(cp.isfinite(brain.v).all().get()):
        raise RuntimeError("NaN/Inf in neural state")
    if int(dn_counts.sum()) != dn_spikes:
        raise RuntimeError("DN spike accounting mismatch")
    return dict(seed=seed, cns_spikes=total_spikes, visual_spikes=visual_spikes,
                dn_spikes=dn_spikes, dn_counts=dn_counts.tolist(),
                step_ms=step_seconds * 1000 / STEPS,
                dn_aggregate_ms=aggregate_seconds * 1000 / STEPS)


def distance(a, b):
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))


def cosine(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else None


def compare(trials):
    names = list(trials)
    vectors = {name: np.array([t["dn_counts"] for t in group], dtype=np.float64)
               for name, group in trials.items()}
    means = {name: value.mean(axis=0) for name, value in vectors.items()}
    within = {}
    for name, group in vectors.items():
        pairs = [distance(group[i], group[j]) for i in range(len(group))
                 for j in range(i + 1, len(group))]
        within[name] = dict(median_euclidean=float(statistics.median(pairs)),
                            min_euclidean=float(min(pairs)), max_euclidean=float(max(pairs)))
    pairs = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = vectors[left], vectors[right]
            all_distances = [distance(x, y) for x in a for y in b]
            paired = [distance(x, y) for x, y in zip(a, b)]
            key = f"{left}__{right}"
            pairs[key] = dict(mean_vector_cosine=cosine(means[left], means[right]),
                              mean_vector_euclidean=distance(means[left], means[right]),
                              median_cross_trial_euclidean=float(statistics.median(all_distances)),
                              median_paired_seed_euclidean=float(statistics.median(paired)),
                              dn_total_mean_difference=float(np.mean([t["dn_spikes"] for t in trials[left]])
                                                             - np.mean([t["dn_spikes"] for t in trials[right]])))
    return dict(within=within, between=pairs, names=names)


def main():
    if not DATA.joinpath("brain.npz").is_file() or not DATA.joinpath("weights.npz").is_file():
        raise FileNotFoundError("MaleCNS data missing; run the documented flybrain download")
    brain = FlyBrain(data=DATA, device="cuda", batch=1)
    if brain.device != "cuda" or brain.n != 166700 or brain._W.nnz != 25582938:
        raise RuntimeError("Wrong device or MaleCNS dataset")
    dn = brain.cells(["descending_neuron"])
    if len(dn) == 0 or not np.all(brain.superclass[dn] == "descending_neuron"):
        raise RuntimeError("Descending-neuron metadata selection failed")
    if len(np.unique(brain.visual)) != len(brain.visual) or len(brain.azimuth) != len(brain.visual):
        raise RuntimeError("Visual receptor metadata mismatch")
    print(f"CUDA MaleCNS: {brain.n} neurons, {brain._W.nnz} connections", flush=True)
    print(f"Visual receptors: {len(brain.visual)}; descending neurons: {len(dn)}", flush=True)
    frames = capture_conditions()
    drives, prep = prepare_drives(frames, brain)
    visual_mask = np.zeros(brain.n, bool)
    visual_mask[brain.visual] = True
    dn_slot = np.full(brain.n, -1, np.int32)
    dn_slot[dn] = np.arange(len(dn), dtype=np.int32)
    # Warm CUDA kernels once outside measurements; each trial still starts from a reset seed.
    brain.reset(seed=0)
    for _ in range(20):
        brain.step(eye_drive=next(iter(drives.values()))[1])
    conditions = {"baseline_none": None, **drives}
    trials = {name: [] for name in conditions}
    for seed in SEEDS:
        for name, drive in conditions.items():
            trials[name].append(run_trial(brain, seed, drive, dn_slot, visual_mask))
        print(f"Completed seed {seed}", flush=True)
    output = dict(model="flybrain 0.1.0 MaleCNS v1.0", device="cuda", steps=STEPS,
                  seeds=SEEDS, receptor_count=len(brain.visual), dn_count=len(dn),
                  config=VisionConfig().__dict__, frame_metrics=prep, trials=trials,
                  comparison=compare(trials))
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "visual-response-trials.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
