"""Experiment 14 frozen-controller, long-horizon autonomous progression attempt."""
from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from flybrain import FlyBrain

from flymon.emulator import PokemonEmulator
from flymon.exploration import (
    ExplorationConfig,
    FrozenExplorationController,
    FrozenInterfaceController,
    visual_metrics,
)
from flymon.interaction import PopulationEventController
from flymon.locomotion import estimate_population_baseline
from flymon.motor import (
    E7_CONFIG,
    MOTOR_TYPES,
    OnlineE3Encoder,
    resolve_dn_ids,
    resolve_motor_populations,
    run_neural_window,
)
from flymon.progression import (
    FFmpegVideoRecorder,
    ProgressionTracker,
    StreamingMetricsWriter,
    atomic_write_json,
    checkpoint_decisions,
    probe_video,
    save_checkpoint,
    sha256_file,
    validate_metrics,
)
from flymon.spatiotemporal import SpatialProjection
from flymon.steering import estimate_baseline
from run_population_event_experiment import (
    HOLD_FRAMES,
    NEURAL_STEPS,
    POPULATION_KEY,
    RELEASE_FRAMES,
    WINDOW_SECONDS,
    active_record,
    baseline_trial,
    load_population,
    population_config,
)
from run_visual_experiment import DATA, ROOT


BASE_COMMIT = "9027e4727e3f17b9a6f840a0c285d517d0cf3740"
RESULTS = ROOT / "results/experiment-14-progression"
CAPTURES = ROOT / "captures/experiment-14-progression"
STATE = ROOT / "states/bedroom.state"
MATCHED_SEEDS = tuple(range(921, 931))
CONDITIONS = ("live_full", "live_no_a", "frozen_full", "no_vision_full")
MATCHED_HORIZON = 1500
LONG_SEED = 950
LONG_HORIZON = 5000
CHECKPOINT_INTERVAL = 250
FPS = 5.0
SCHEMA_VERSION = "experiment-14.v1"
ALLOWED_ACTIONS = frozenset({None, "LEFT", "RIGHT", "UP", "A"})


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def load_frozen_configuration() -> tuple[ExplorationConfig, dict, float]:
    e8 = json.loads((ROOT / "results/experiment-08-steering/thresholds.json").read_text())
    e9 = json.loads((ROOT / "results/experiment-09-locomotion/thresholds.json").read_text())
    e13 = json.loads((ROOT / "results/experiment-13-population-event/thresholds.json").read_text())
    config = ExplorationConfig(
        e8["selected_multiplier"],
        e8["pooled_baseline_sd_floor_hz"],
        e8["left_evidence_side"],
        e9["forward"]["selected_multiplier"],
        e9["forward"]["sd_floor_hz"],
        e9["backward"]["selected_multiplier"],
        e9["backward"]["sd_floor_hz"],
        1,
    )
    selected = e13["selected"]
    expected = {
        "signal": "norm", "event_rule": "rising", "signal_window": 1,
        "evidence_window": 1, "threshold": 1.5,
    }
    if any(selected[key] != value for key, value in expected.items()):
        raise RuntimeError("Experiment 13 frozen A decoder differs from expected selection")
    if (float(e13["hysteresis"]), int(e13["refractory_decisions"]),
            int(e13["a_hold_frames"]), int(e13["a_release_frames"])) != (
            .5, 5, HOLD_FRAMES, RELEASE_FRAMES):
        raise RuntimeError("Experiment 13 timing or event-state parameters changed")
    return config, selected, float(e13["sd_floor_hz"])


def trial_paths(condition: str, seed: int) -> dict[str, Path]:
    stem = f"seed-{seed:04d}"
    return {
        "metrics": RESULTS / "metrics" / condition / f"{stem}.jsonl.gz",
        "video": CAPTURES / "videos" / condition / f"{stem}.mp4",
        "checkpoints": CAPTURES / "checkpoints" / condition / stem,
    }


def sensory_source(condition: str, current: np.ndarray,
                   initial: np.ndarray) -> np.ndarray | None:
    if condition == "no_vision_full":
        return None
    if condition == "frozen_full":
        return initial
    if condition in {"live_full", "live_no_a", "long_run"}:
        return current
    raise ValueError(f"unknown condition: {condition}")


def apply_action(game: PokemonEmulator, action: str | None) -> None:
    if action is not None:
        game.press(action.lower())
        game.tick(HOLD_FRAMES)
        game.release(action.lower())
    else:
        game.tick(HOLD_FRAMES)
    game.tick(RELEASE_FRAMES)


def metric_row(*, condition: str, seed: int, decision: int,
               elapsed: float, action: str | None, previous_action: str | None,
               rates: dict, state: dict, steering_baseline,
               locomotion_baseline, event_state: dict, rms_z: float,
               input_hash: str | None, visual: dict,
               luminance_mean: float, luminance_std: float,
               trajectory: dict, a_enabled: bool) -> dict:
    direction = state["direction"]
    steering_threshold = float(direction["thresholds_hz"]["LEFT"])
    forward_threshold = float(direction["thresholds_hz"]["UP"])
    steer = float(direction["steering_signal_hz"])
    forward = float(direction["forward_delta_hz"])
    if action == "A":
        reason = "A-event-first"
    else:
        reason = f"direction-{direction['reason']}"
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": 14,
        "condition": condition,
        "seed": seed,
        "decision": decision,
        "simulated_time_s": float((decision + 1) * WINDOW_SECONDS),
        "elapsed_run_s": float(elapsed),
        "action": action,
        "previous_action": previous_action,
        "controller": {
            "selected_action": state["selected_action"],
            "arbitration_reason": reason,
        },
        "DNa02": {
            "left_rate_hz": float(rates["DNa02_L"]),
            "right_rate_hz": float(rates["DNa02_R"]),
            "steering_signal_hz": steer,
            "baseline_left_hz": float(steering_baseline.left_mean_hz),
            "baseline_right_hz": float(steering_baseline.right_mean_hz),
            "threshold_hz": steering_threshold,
            "normalized_excess": float(abs(steer) / steering_threshold - 1.0),
        },
        "DNg100": {
            "rate_hz": float(rates["DNg100"]),
            "baseline_hz": float(locomotion_baseline.dng100_mean_hz),
            "delta_hz": forward,
            "threshold_hz": forward_threshold,
            "normalized_excess": float(forward / forward_threshold - 1.0),
        },
        "P20": {
            "population_size": 20,
            "rms_z": float(rms_z),
            "threshold": 1.5,
            "hysteresis": .5,
            "armed": event_state.get("armed"),
            "refractory_remaining": int(event_state["refractory_remaining"]),
            "event_fired": bool(event_state.get("selected_action") == "A"),
            "refractory_suppressions": int(event_state["refractory_suppressions"]),
        },
        "action_state": {
            "A_enabled": bool(a_enabled),
            "DOWN_enabled": False,
            "hold_frames": HOLD_FRAMES,
            "release_frames": RELEASE_FRAMES,
        },
        "visual_analysis": {
            "input_frame_sha256": input_hash,
            "resulting_frame_sha256": visual["frame_sha256"],
            "coarse_state_hash": visual["coarse_state_sha256"],
            "frame_to_previous_mae": float(visual["frame_to_previous_mae"]),
            "frame_to_initial_mae": float(visual["frame_to_initial_mae"]),
            "mean_luminance": float(luminance_mean),
            "luminance_std": float(luminance_std),
        },
        "trajectory": trajectory,
    }


def clean_incomplete(paths: dict[str, Path]) -> None:
    for key in ("metrics", "video"):
        path = paths[key]
        partials = [path.with_name(path.name + ".partial")]
        if key == "video":
            partials.append(path.with_name(path.stem + ".partial" + path.suffix))
        for partial in partials:
            if partial.exists():
                partial.unlink()
    if paths["checkpoints"].exists():
        shutil.rmtree(paths["checkpoints"])


def run_trial(game, brain, projection, populations, direction_config,
              event_spec, event_floor_hz, *, condition: str, seed: int,
              horizon: int) -> dict:
    paths = trial_paths(condition, seed)
    clean_incomplete(paths)
    for path in (paths["metrics"], paths["video"]):
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["checkpoints"].mkdir(parents=True, exist_ok=True)

    game.load_state(STATE)
    game.tick(1)
    initial = game.framebuffer()
    save_checkpoint(initial, paths["checkpoints"] / "decision-000000.png")

    vector_baseline, _, rate_windows, cell_windows, _, cache = baseline_trial(
        brain, projection, populations, seed)
    steering_baseline = estimate_baseline(rate_windows)
    locomotion_baseline = estimate_population_baseline(rate_windows, cell_windows)
    direction = FrozenExplorationController(
        steering_baseline, locomotion_baseline, direction_config, enable_down=False)
    a_enabled = condition != "live_no_a"
    event = PopulationEventController(
        vector_baseline, population_config(event_spec, event_floor_hz),
        ablated=not a_enabled)
    controller = FrozenInterfaceController(direction, None, event)

    encoder = OnlineE3Encoder(projection)
    metric_encoder = OnlineE3Encoder(projection)
    if condition == "no_vision_full":
        initial_vector = np.zeros(len(projection.ids), np.float32)
    else:
        initial_vector, _ = metric_encoder.encode(initial)
    checkpoints = set(checkpoint_decisions(horizon, CHECKPOINT_INTERVAL))
    tracker = ProgressionTracker()
    # The first decision's change is measured from the exact initial frame.
    # Historical logs produced before this fix are marked null at decision 0
    # because the exact first resulting pixels were intentionally not retained.
    previous_result = initial
    previous_action = None
    started = time.perf_counter()

    try:
        with StreamingMetricsWriter(paths["metrics"]) as writer, \
                FFmpegVideoRecorder(paths["video"], fps=FPS) as recorder:
            for decision in range(horizon):
                current = game.framebuffer()
                source = sensory_source(condition, current, initial)
                if source is None:
                    drive = np.zeros(len(projection.ids), np.float32)
                    inject = None
                    input_hash = None
                else:
                    drive, _ = encoder.encode(source)
                    inject = drive
                    input_hash = hashlib.sha256(source.tobytes()).hexdigest()

                neural, _ = run_neural_window(
                    brain, projection, inject, populations, NEURAL_STEPS, cache)
                population = active_record(neural, vector_baseline, event_floor_hz)
                z = np.asarray(population["z"], float)
                rms_z = float(np.sqrt(np.mean(np.square(z))))
                rates = neural["controller_rates_hz"] | {
                    "event_population_rates_hz": population["rates_hz"]
                }
                action, state = controller.decode(rates)
                if action not in ALLOWED_ACTIONS:
                    raise RuntimeError(f"forbidden controller action: {action}")
                if action == "DOWN":
                    raise RuntimeError("DOWN is disabled in Experiment 14")
                if not a_enabled and action == "A":
                    raise RuntimeError("A emitted in LIVE_NO_A")

                apply_action(game, action)
                resulting = game.framebuffer()
                recorder.write(resulting[..., :3])
                boundary = decision + 1
                if boundary in checkpoints:
                    save_checkpoint(
                        resulting,
                        paths["checkpoints"] / f"decision-{boundary:06d}.png")

                visual = visual_metrics(
                    resulting, initial, drive, initial_vector, previous_result)
                rgb = resulting[..., :3].astype(np.float32) / 255.0
                trajectory = tracker.update(visual["coarse_state_sha256"])
                row = metric_row(
                    condition=condition, seed=seed, decision=decision,
                    elapsed=time.perf_counter() - started, action=action,
                    previous_action=previous_action, rates=rates, state=state,
                    steering_baseline=steering_baseline,
                    locomotion_baseline=locomotion_baseline,
                    event_state=state["event"], rms_z=rms_z,
                    input_hash=input_hash, visual=visual,
                    luminance_mean=float(rgb.mean()),
                    luminance_std=float(rgb.std()), trajectory=trajectory,
                    a_enabled=a_enabled)
                writer.write(row)
                previous_result = resulting
                previous_action = action
                if boundary % 100 == 0 or boundary == horizon:
                    print(
                        f"experiment14 {condition} seed={seed} "
                        f"decision={boundary}/{horizon}", flush=True)
    except Exception:
        raise

    runtime = time.perf_counter() - started
    integrity = validate_metrics(
        paths["metrics"], horizon=horizon, condition=condition, seed=seed,
        allowed_actions=ALLOWED_ACTIONS, forbid_a=not a_enabled)
    video = probe_video(paths["video"], expected_frames=horizon)
    checkpoint_files = sorted(paths["checkpoints"].glob("decision-*.png"))
    expected_checkpoints = checkpoint_decisions(horizon, CHECKPOINT_INTERVAL)
    observed_checkpoints = [int(path.stem.split("-")[-1]) for path in checkpoint_files]
    if observed_checkpoints != expected_checkpoints:
        raise RuntimeError(
            f"checkpoint mismatch: {observed_checkpoints} != {expected_checkpoints}")
    return {
        "condition": condition,
        "seed": seed,
        "horizon": horizon,
        "completed_decisions": integrity["row_count"],
        "metrics_path": relative(paths["metrics"]),
        "metrics_sha256": file_sha256(paths["metrics"]),
        "metrics_row_count": integrity["row_count"],
        "metrics_compressed_size": paths["metrics"].stat().st_size,
        "metrics_integrity": integrity,
        "video_path": relative(paths["video"]),
        "video_size": paths["video"].stat().st_size,
        "video_sha256": file_sha256(paths["video"]),
        "video": video,
        "checkpoint_directory": relative(paths["checkpoints"]),
        "checkpoint_decisions": observed_checkpoints,
        "checkpoint_file_count": len(checkpoint_files),
        "checkpoint_total_size": sum(path.stat().st_size for path in checkpoint_files),
        "runtime_seconds": runtime,
        "completed_successfully": True,
        "error": None,
    }


def initial_manifest(direction_config, event_spec, event_floor_hz) -> dict:
    return {
        "schema_version": "experiment-14.manifest.v1",
        "experiment": 14,
        "base_commit": BASE_COMMIT,
        "matched_seeds": list(MATCHED_SEEDS),
        "matched_horizon": MATCHED_HORIZON,
        "conditions": list(CONDITIONS),
        "long_run_seed": LONG_SEED,
        "long_run_horizon": LONG_HORIZON,
        "controller_parameters_changed": False,
        "direction_config": asdict(direction_config),
        "event_config": {
            "selected": event_spec,
            "sd_floor_hz": event_floor_hz,
            "hold_frames": HOLD_FRAMES,
            "release_frames": RELEASE_FRAMES,
        },
        "trials": [],
    }


def complete_key(row: dict) -> tuple[str, int, int]:
    return row["condition"], int(row["seed"]), int(row["horizon"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restart", action="store_true",
        help="rerun completed Experiment 14 trials (controller remains frozen)")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    CAPTURES.mkdir(parents=True, exist_ok=True)
    FFmpegVideoRecorder.check_available()
    direction_config, event_spec, event_floor_hz = load_frozen_configuration()
    manifest_path = RESULTS / "trial-manifest.json"
    if manifest_path.exists() and not args.restart:
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = initial_manifest(direction_config, event_spec, event_floor_hz)
    completed = {
        complete_key(row) for row in manifest.get("trials", [])
        if row.get("completed_successfully")
    }

    population_document = load_population()
    if population_document["name"] != "E4_E6_frozen_P20" or population_document["count"] != 20:
        raise RuntimeError("Experiment 13 frozen population identity changed")
    brain = FlyBrain(data=DATA, device="cuda")
    ids = [row["flywire_malecns_id"] for row in population_document["neurons"]]
    resolved, frozen_indices = resolve_dn_ids(brain, DATA, ids)
    if [row["brain_index"] for row in resolved["neurons"]] != [
            row["brain_index"] for row in population_document["neurons"]]:
        raise RuntimeError("resolved P20 order differs from Experiment 13")
    _, motor_populations = resolve_motor_populations(brain, DATA, MOTOR_TYPES)
    populations = dict(motor_populations)
    populations[POPULATION_KEY] = frozen_indices
    projection = SpatialProjection(brain, E7_CONFIG)

    schedule = [
        (condition, seed, MATCHED_HORIZON)
        for seed in MATCHED_SEEDS for condition in CONDITIONS
    ] + [("long_run", LONG_SEED, LONG_HORIZON)]
    with PokemonEmulator() as game:
        for condition, seed, horizon in schedule:
            key = (condition, seed, horizon)
            if key in completed:
                print(f"experiment14 resume skip {condition} seed={seed}", flush=True)
                continue
            manifest["trials"] = [
                row for row in manifest.get("trials", [])
                if complete_key(row) != key
            ]
            try:
                trial = run_trial(
                    game, brain, projection, populations, direction_config,
                    event_spec, event_floor_hz, condition=condition,
                    seed=seed, horizon=horizon)
            except Exception as exc:
                manifest["trials"].append({
                    "condition": condition, "seed": seed, "horizon": horizon,
                    "completed_decisions": None, "completed_successfully": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                atomic_write_json(manifest_path, manifest)
                raise
            manifest["trials"].append(trial)
            manifest["trials"].sort(
                key=lambda row: (row["seed"], CONDITIONS.index(row["condition"])
                    if row["condition"] in CONDITIONS else len(CONDITIONS)))
            atomic_write_json(manifest_path, manifest)
            print(f"experiment14 completed {condition} seed={seed}", flush=True)

    print("Experiment 14 acquisition complete", flush=True)


if __name__ == "__main__":
    main()
