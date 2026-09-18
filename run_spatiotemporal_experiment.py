"""Experiment 3: short PyBoy sequences into spatial visual-projection drive."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import cupy as cp
import numpy as np
from PIL import Image
from flybrain import FlyBrain

from flymon.emulator import PokemonEmulator
from flymon.spatiotemporal import (SpatiotemporalConfig, SpatialProjection,
                                   spatial_grid, temporal_features, vector_stats)
from run_visual_experiment import DATA, ROOT, SEEDS, STEPS, compare

RESULTS = ROOT / "results" / "experiment-03-spatiotemporal"
CAPTURES = ROOT / "captures" / "experiment-03"
PRIOR_CAPTURES = ROOT / "captures" / "vision"
BEDROOM_STATE = ROOT / "states" / "bedroom.state"
NAMES = ("intro", "title", "new_game_menu", "oak_dialogue", "bedroom")
CONFIG = SpatiotemporalConfig()


def move_to_condition(game: PokemonEmulator, name: str) -> None:
    if name == "bedroom":
        game.load_state(BEDROOM_STATE)
        game.tick(1)
        return
    game.tick(900)
    if name == "intro":
        return
    game.tick(600)
    if name == "title":
        return
    game.tap("start", 3, 3)
    game.tick(90)
    game.tap("a", 3, 3)
    game.tick(90)
    if name == "new_game_menu":
        return
    for _ in range(5):
        game.tap("a", 3, 3)
        game.tick(90)
    if name == "oak_dialogue":
        return
    raise ValueError(name)


def capture_sequences(config: SpatiotemporalConfig):
    """Reconstruct prior checkpoints, then capture actual PyBoy frame evolution."""
    sequences, timings = {}, {}
    for name in NAMES:
        with PokemonEmulator() as game:
            move_to_condition(game, name)
            frames, seconds = [], []
            for i in range(config.frames):
                t0 = time.perf_counter()
                if i:
                    if name == "bedroom" and i == 4:
                        game.press("right")
                    if name == "bedroom" and i == 6:
                        game.release("right")
                    game.tick(config.game_frame_interval)
                frames.append(game.framebuffer())
                seconds.append(time.perf_counter() - t0)
            if name == "bedroom":
                game.release("right")
        previous = np.asarray(Image.open(PRIOR_CAPTURES / f"{name}-original.png").convert("RGBA"))
        if not np.array_equal(frames[0], previous):
            raise RuntimeError(f"Initial {name} frame differs from Experiments 1/2")
        sequences[name] = np.stack(frames)
        timings[name] = [s * 1000 for s in seconds]
        print(f"Captured {name}: {config.frames} frames; unique={len({f.tobytes() for f in frames})}", flush=True)
    return sequences, timings


def as_gray_image(a: np.ndarray, width=160, height=144):
    im = Image.fromarray(np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8), "L")
    return im.resize((width, height), Image.Resampling.NEAREST)


def projection_image(projection: SpatialProjection, vector: np.ndarray):
    h, w = projection.config.grid_height, projection.config.grid_width
    channel = np.zeros((h, w, 3), np.float32)
    offset = 0
    for (kind, _), ids in projection.populations.items():
        y, x = projection.grid_indices[(kind, _)]
        values = vector[offset:offset + len(ids)] / projection.config.voltage_cap
        channel[y, x, 0 if kind == "LC10a" else 2] = values
        offset += len(ids)
    image = Image.fromarray(np.rint(channel * 255).astype(np.uint8), "RGB")
    return image.resize((160, 144), Image.Resampling.NEAREST)


def save_artifacts(name, frames, grids, darkness, change, vectors, projection):
    folder = CAPTURES / name
    folder.mkdir(parents=True, exist_ok=True)
    thumbnails = []
    for i, (frame, grid, dark, delta, vector) in enumerate(zip(frames, grids, darkness, change, vectors)):
        Image.fromarray(frame, "RGBA").save(folder / f"frame-{i:02d}-original.png")
        as_gray_image(grid).save(folder / f"frame-{i:02d}-grid.png")
        as_gray_image(delta).save(folder / f"frame-{i:02d}-temporal-difference.png")
        as_gray_image(dark).save(folder / f"frame-{i:02d}-feature-map.png")
        projected = projection_image(projection, vector)
        projected.save(folder / f"frame-{i:02d}-projected-drive.png")
        thumbnails.append((Image.fromarray(frame, "RGBA").convert("RGB"), as_gray_image(grid).convert("RGB"),
                           as_gray_image(delta).convert("RGB"), projected))
    sheet = Image.new("RGB", (160 * len(frames), 144 * 4))
    for col, items in enumerate(thumbnails):
        for row, im in enumerate(items):
            sheet.paste(im, (160 * col, 144 * row))
    sheet.save(CAPTURES / f"{name}-contact-sheet.png")


def prepare_sequences(frames_by_name, projection, config):
    prepared, diagnostics = {}, {}
    for name, frames in frames_by_name.items():
        grays, grids, spatial_ms = [], [], []
        for frame in frames:
            t0 = time.perf_counter()
            gray, grid = spatial_grid(frame, config)
            spatial_ms.append((time.perf_counter() - t0) * 1000)
            grays.append(gray)
            grids.append(grid)
        grids = np.stack(grids)
        t0 = time.perf_counter()
        darkness, change = temporal_features(grids)
        temporal_ms = (time.perf_counter() - t0) * 1000 / len(frames)
        t0 = time.perf_counter()
        vectors = np.stack([projection.encode(d, c) for d, c in zip(darkness, change)])
        projection_ms = (time.perf_counter() - t0) * 1000 / len(frames)
        static_dark, static_change = temporal_features(np.repeat(grids[:1], config.frames, axis=0))
        static_vectors = np.stack([projection.encode(d, c) for d, c in zip(static_dark, static_change)])
        save_artifacts(name, frames, grids, darkness, change, vectors, projection)
        prepared[name] = dict(dynamic=vectors, static=static_vectors)
        diagnostics[name] = dict(frame_sha256=[hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames],
                                 unique_frames=len({frame.tobytes() for frame in frames}),
                                 spatial_preprocess_ms=spatial_ms, temporal_feature_ms=temporal_ms,
                                 projection_encode_ms=projection_ms,
                                 frame_luminance_mean=[float(g.mean()) for g in grids],
                                 temporal_mean=[float(c.mean()) for c in change],
                                 dynamic_stats=[vector_stats(v) for v in vectors],
                                 static_stats=[vector_stats(v) for v in static_vectors],
                                 dynamic_vectors=vectors.tolist(), static_vectors=static_vectors.tolist())
    return prepared, diagnostics


def encoder_comparison(prepared):
    names = list(prepared)
    vectors = {name: prepared[name]["dynamic"].ravel().astype(np.float64) for name in names}
    static = {name: prepared[name]["static"].ravel().astype(np.float64) for name in names}
    distances = {f"{a}__{b}": float(np.linalg.norm(vectors[a] - vectors[b]))
                 for i, a in enumerate(names) for b in names[i + 1:]}
    cosines = {f"{a}__{b}": float(np.dot(vectors[a], vectors[b]) /
               (np.linalg.norm(vectors[a]) * np.linalg.norm(vectors[b])))
               for i, a in enumerate(names) for b in names[i + 1:]}
    static_distances = {f"{a}__{b}": float(np.linalg.norm(static[a] - static[b]))
                        for i, a in enumerate(names) for b in names[i + 1:]}
    within = {name: 0.0 for name in names}  # one deterministic capture, replayed for each seed
    unique = len({prepared[name]["dynamic"].tobytes() for name in names})
    return dict(unique_outputs=unique, within_condition_distance=within,
                mean_within_condition_distance=0.0,
                between_condition_distance=distances,
                mean_between_condition_distance=float(np.mean(list(distances.values()))),
                between_within_ratio=None, pairwise_cosine=cosines,
                static_between_condition_distance=static_distances,
                dynamic_static_distance={name: float(np.linalg.norm(vectors[name] - static[name]))
                                         for name in names})


def run_trial(brain, seed, vectors, projection, dn_slot, vp_slot, config):
    brain.reset(seed=seed)
    cp.cuda.Stream.null.synchronize()
    dn_counts = np.zeros(int((dn_slot >= 0).sum()), np.int32)
    vp_counts = np.zeros(len(projection.ids), np.int32)
    total = vp_total = dn_total = 0
    step_seconds = aggregate_seconds = grouping_seconds = 0.0
    windows = []
    if vectors is None:
        frame_vectors = [None] * config.frames
    else:
        frame_vectors = vectors
    for vector in frame_vectors:
        t0 = time.perf_counter()
        pairs = [] if vector is None else [(cp.asarray(ids), voltage)
                                            for ids, voltage in projection.injection_pairs(vector)]
        grouping_seconds += time.perf_counter() - t0
        window_total = window_vp = window_dn = 0
        for _ in range(config.neural_steps_per_frame):
            t0 = time.perf_counter()
            fired = brain.step(eye_drive=None, inject=pairs)
            step_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()
            n = len(fired)
            vp_active = vp_slot[fired]
            vp_active = vp_active[vp_active >= 0]
            dn_active = dn_slot[fired]
            dn_active = dn_active[dn_active >= 0]
            dn_counts += np.bincount(dn_active, minlength=len(dn_counts)).astype(np.int32)
            vp_counts += np.bincount(vp_active, minlength=len(vp_counts)).astype(np.int32)
            total += n
            vp_total += len(vp_active)
            dn_total += len(dn_active)
            window_total += n
            window_vp += len(vp_active)
            window_dn += len(dn_active)
            aggregate_seconds += time.perf_counter() - t0
        windows.append(dict(cns_spikes=window_total, visual_projection_spikes=window_vp,
                            dn_spikes=window_dn))
    if int(dn_counts.sum()) != dn_total or int(vp_counts.sum()) != vp_total:
        raise RuntimeError("Spike accounting mismatch")
    if not bool(cp.isfinite(brain.v).all().get()):
        raise RuntimeError("NaN/Inf in neural state")
    return dict(seed=seed, cns_spikes=total, visual_projection_spikes=vp_total,
                dn_spikes=dn_total, dn_counts=dn_counts.tolist(), vp_counts=vp_counts.tolist(),
                windows=windows, projection_grouping_ms=grouping_seconds * 1000 / config.frames,
                step_ms=step_seconds * 1000 / STEPS,
                dn_aggregate_ms=aggregate_seconds * 1000 / STEPS)


def vector_comparison(trials, key):
    converted = {name: [dict(dn_counts=trial[key], dn_spikes=sum(trial[key]), seed=trial["seed"])
                        for trial in group] for name, group in trials.items()}
    return compare(converted)


def nearest_centroid(trials, key="dn_counts"):
    names = list(NAMES)
    matrix = np.array([[trial[key] for trial in trials[name]] for name in names], dtype=np.float64)
    confusion = np.zeros((len(names), len(names)), int)
    for held in range(len(SEEDS)):
        centroids = np.delete(matrix, held, axis=1).mean(axis=1)
        for actual in range(len(names)):
            distances = np.linalg.norm(centroids - matrix[actual, held], axis=1)
            predicted = int(np.argmin(distances))
            confusion[actual, predicted] += 1
    return dict(method="leave-one-seed-out nearest centroid, Euclidean DN counts",
                chance_accuracy=1 / len(names), accuracy=float(np.trace(confusion) / confusion.sum()),
                confusion=confusion.tolist(), labels=names)


def main():
    prior1 = json.loads((ROOT / "results/experiment-01-photoreceptors/trials.json").read_text())
    prior2 = json.loads((ROOT / "results/experiment-02-feature-detectors/trials.json").read_text())
    if prior1["seeds"] != SEEDS or prior2["seeds"] != SEEDS or STEPS != CONFIG.frames * CONFIG.neural_steps_per_frame:
        raise RuntimeError("Prior seed protocol or neural step count changed")
    brain = FlyBrain(data=DATA, device="cuda", batch=1)
    if brain.device != "cuda" or brain.n != 166700 or brain._W.nnz != 25582938:
        raise RuntimeError("Wrong device or MaleCNS data")
    projection = SpatialProjection(brain, CONFIG)
    dn = brain.cells(["descending_neuron"])
    if len(dn) != prior1["dn_count"] or not np.all(brain.superclass[dn] == "descending_neuron"):
        raise RuntimeError("DN metadata selection differs from prior experiments")
    counts = {f"{kind}_{side}": len(ids) for (kind, side), ids in projection.populations.items()}
    print("Projection populations:", counts, "total", len(projection.ids), "DNs", len(dn), flush=True)
    sequences, capture_ms = capture_sequences(CONFIG)
    prepared, frame_metrics = prepare_sequences(sequences, projection, CONFIG)
    diagnostic = encoder_comparison(prepared)
    print("Encoder unique:", diagnostic["unique_outputs"], "of", len(NAMES), flush=True)
    print("Encoder pairwise Euclidean:", diagnostic["between_condition_distance"], flush=True)
    if diagnostic["unique_outputs"] < len(NAMES):
        print("WARNING: distinct conditions collapsed at encoder output; proceeding for loss-stage diagnosis", flush=True)
    mask_dn = np.full(brain.n, -1, np.int32)
    mask_dn[dn] = np.arange(len(dn), dtype=np.int32)
    mask_vp = np.full(brain.n, -1, np.int32)
    mask_vp[projection.ids] = np.arange(len(projection.ids), dtype=np.int32)
    brain.reset(seed=0)
    for _ in range(20):
        brain.step(eye_drive=None)
    trials = {name: [] for name in ("baseline_none", *NAMES, *(f"{n}_static" for n in NAMES))}
    for seed in SEEDS:
        trials["baseline_none"].append(run_trial(brain, seed, None, projection, mask_dn, mask_vp, CONFIG))
        for name in NAMES:
            trials[name].append(run_trial(brain, seed, prepared[name]["dynamic"], projection,
                                          mask_dn, mask_vp, CONFIG))
            trials[f"{name}_static"].append(run_trial(brain, seed, prepared[name]["static"], projection,
                                                       mask_dn, mask_vp, CONFIG))
        print("Completed seed", seed, flush=True)
    for name in ("baseline_none",):
        for trial, old in zip(trials[name], prior1["trials"][name]):
            if trial["dn_counts"] != old["dn_counts"]:
                raise RuntimeError("No-vision baseline differs from Experiment 1")
    visual = {name: trials[name] for name in NAMES}
    static = {name: trials[name + "_static"] for name in NAMES}
    output = dict(model="flybrain 0.1.0 MaleCNS v1.0", device="cuda", pathway="spatial surrogate into LPLC2/LC10a",
                  config=asdict(CONFIG), steps=STEPS, seeds=SEEDS, dn_count=len(dn),
                  projection_counts=counts, projection_neuron_indices=projection.ids.tolist(),
                  projection_grid_coordinates={f"{kind}_{side}": np.column_stack(projection.grid_indices[(kind, side)]).tolist()
                                               for kind, side in projection.populations},
                  bedroom_state=dict(path="states/bedroom.state", sha256=hashlib.sha256(BEDROOM_STATE.read_bytes()).hexdigest()),
                  capture_ms=capture_ms, frame_metrics=frame_metrics, encoder_comparison=diagnostic,
                  trials=trials, dn_comparison=compare(visual), static_dn_comparison=compare(static),
                  vp_comparison=vector_comparison(visual, "vp_counts"),
                  static_vp_comparison=vector_comparison(static, "vp_counts"),
                  dn_classifier=nearest_centroid(visual), vp_classifier=nearest_centroid(visual, "vp_counts"),
                  static_dn_classifier=nearest_centroid(static))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "trials.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print("Saved", path, flush=True)


if __name__ == "__main__":
    main()
