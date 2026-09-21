"""Experiment 13 frozen historical-DN population event decoding."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from flybrain import FlyBrain

from flymon.emulator import PokemonEmulator
from flymon.exploration import (
    ExplorationConfig,
    FrozenExplorationController,
    FrozenInterfaceController,
    visual_metrics,
)
from flymon.interaction import (
    EventDecoderConfig,
    FrozenVectorBaseline,
    PopulationEventConfig,
    PopulationEventController,
    TransparentEventController,
    estimate_frozen_population_baseline,
    estimate_frozen_vector_baseline,
    permute_population_cells,
    permute_population_time,
    population_z_vector,
)
from flymon.locomotion import estimate_population_baseline
from flymon.motor import (
    E7_CONFIG,
    MOTOR_TYPES,
    OnlineE3Encoder,
    resolve_dn_ids,
    resolve_motor_populations,
    run_neural_window,
    synthetic_frame,
)
from flymon.spatiotemporal import SpatialProjection
from flymon.steering import estimate_baseline
from run_visual_experiment import DATA, ROOT


RESULTS = ROOT / "results/experiment-13-population-event"
CAPTURES = ROOT / "captures/experiment-13"
STATE = ROOT / "states/bedroom.state"
CALIBRATION_SEEDS = list(range(861, 881))
HELDOUT_SEEDS = list(range(881, 901))
INTEGRATED_SEEDS = list(range(901, 921))
BASELINE_WINDOWS = 10
CONTROLLED_WINDOWS = 20
NATURAL_HORIZON = 60
NEURAL_STEPS = 10
WINDOW_SECONDS = .2
HOLD_FRAMES = 8
RELEASE_FRAMES = 4
POPULATION_KEY = "E4_frozen"
CONTROLLED_ONSETS = {
    "luminance_onset", "luminance_offset", "looming_onset",
    "left_onset", "right_onset",
}


def rgba(gray: np.ndarray) -> np.ndarray:
    frame = np.empty((144, 160, 4), np.uint8)
    frame[..., :3] = gray[..., None]
    frame[..., 3] = 255
    return frame


def controlled_sequences() -> dict[str, list[np.ndarray | None]]:
    dark = rgba(np.full((144, 160), 16, np.uint8))
    light = rgba(np.full((144, 160), 240, np.uint8))
    gray = np.full((144, 160), 192, np.uint8)
    loom = []
    for size in np.linspace(6, 100, CONTROLLED_WINDOWS, dtype=int):
        image = gray.copy()
        image[72-size//2:72-size//2+size, 80-size//2:80-size//2+size] = 16
        loom.append(rgba(image))
    onset = lambda a, b: [a] * 5 + [b] * 15
    return {
        "controlled_no_vision": [None] * CONTROLLED_WINDOWS,
        "uniform": [synthetic_frame("uniform")] * CONTROLLED_WINDOWS,
        "luminance_onset": onset(dark, light),
        "luminance_offset": onset(light, dark),
        "looming_onset": [synthetic_frame("uniform")] * 5 + loom[:15],
        "left_onset": [synthetic_frame("uniform")] * 5 + [synthetic_frame("left")] * 15,
        "right_onset": [synthetic_frame("uniform")] * 5 + [synthetic_frame("right")] * 15,
    }


def secondary_sequences() -> dict[str, list[np.ndarray]]:
    names = {
        "secondary_title": "title",
        "secondary_intro": "intro",
        "secondary_menu": "new_game_menu",
        "secondary_dialogue": "oak_dialogue",
    }
    sequences = {}
    for condition, directory in names.items():
        frames = [
            np.asarray(Image.open(
                ROOT / f"captures/experiment-03/{directory}/frame-{index:02d}-original.png"
            ).convert("RGBA"))
            for index in range(10)
        ]
        sequences[condition] = frames
    return sequences


def source_frame(condition: str, current: np.ndarray, initial: np.ndarray,
                 order: np.ndarray) -> np.ndarray | None:
    if condition == "bedroom_no_vision":
        return None
    if condition == "bedroom_frozen":
        return initial
    if condition == "bedroom_shuffled":
        return current.reshape(-1, 4)[order].reshape(current.shape).copy()
    return current


def load_population() -> dict:
    return json.loads((RESULTS / "population.json").read_text())


def baseline_trial(brain, projection, populations, seed):
    brain.reset(seed=seed)
    cache = {}
    rate_windows, per_cell_windows, population_counts = [], [], []
    for _ in range(BASELINE_WINDOWS):
        neural, _ = run_neural_window(
            brain, projection, None, populations, NEURAL_STEPS, cache)
        rate_windows.append(neural["controller_rates_hz"])
        per_cell_windows.append(neural["per_cell_counts"])
        population_counts.append(neural["per_cell_counts"][POPULATION_KEY])
    vector = estimate_frozen_vector_baseline(population_counts, WINDOW_SECONDS)
    named = estimate_frozen_population_baseline(
        rate_windows, per_cell_windows, tuple(populations), WINDOW_SECONDS)
    return vector, named, rate_windows, per_cell_windows, population_counts, cache


def active_record(neural, vector_baseline: FrozenVectorBaseline,
                  floor_hz: float | None = None) -> dict:
    counts = list(map(int, neural["per_cell_counts"][POPULATION_KEY]))
    rates = [float(value / WINDOW_SECONDS) for value in counts]
    means = list(vector_baseline.means_hz)
    record = {
        "counts": counts,
        "rates_hz": rates,
        "baseline_means_hz": means,
        "baseline_variances_hz2": list(vector_baseline.variances_hz2),
        "deltas_hz": [rate - mean for rate, mean in zip(rates, means)],
    }
    if floor_hz is not None:
        record["z"] = population_z_vector(rates, vector_baseline, floor_hz).tolist()
    dnp_counts = list(map(int, neural["per_cell_counts"]["DNp01"]))
    dnp_rates = [float(value / WINDOW_SECONDS) for value in dnp_counts]
    record["dnp01"] = {"counts": dnp_counts, "rates_hz": dnp_rates}
    return record


def baseline_payload(vector, named, rate_windows, population_counts) -> dict:
    return {
        "vector": asdict(vector),
        "named": asdict(named),
        "controller_rate_windows": rate_windows,
        "population_baseline_counts": population_counts,
    }


def acquire_calibration(brain, projection, populations):
    records, baselines = [], {}
    fixed_sequences = controlled_sequences() | secondary_sequences()
    for seed in CALIBRATION_SEEDS:
        for condition, frames in fixed_sequences.items():
            vector, named, rates, cells, counts, cache = baseline_trial(
                brain, projection, populations, seed)
            baselines[f"{seed}:{condition}"] = baseline_payload(
                vector, named, rates, counts)
            encoder = OnlineE3Encoder(projection)
            for decision, frame in enumerate(frames):
                drive = None if frame is None else encoder.encode(frame)[0]
                neural, _ = run_neural_window(
                    brain, projection, drive, populations, NEURAL_STEPS, cache)
                records.append({
                    "seed": seed,
                    "condition": condition,
                    "decision": decision,
                    "population": active_record(neural, vector),
                    "frame_sha256": None if frame is None else
                        hashlib.sha256(frame.tobytes()).hexdigest(),
                })
        print("controlled calibration", seed, flush=True)

    with PokemonEmulator() as game:
        for seed in CALIBRATION_SEEDS:
            for condition in (
                "bedroom_live", "bedroom_no_vision",
                "bedroom_frozen", "bedroom_shuffled",
            ):
                game.load_state(STATE)
                game.tick(1)
                initial = game.framebuffer()
                order = np.random.default_rng(130000 + seed).permutation(144 * 160)
                vector, named, rates, cells, counts, cache = baseline_trial(
                    brain, projection, populations, seed)
                baselines[f"{seed}:{condition}"] = baseline_payload(
                    vector, named, rates, counts)
                encoder = OnlineE3Encoder(projection)
                for decision in range(NATURAL_HORIZON):
                    current = game.framebuffer()
                    frame = source_frame(condition, current, initial, order)
                    drive = None if frame is None else encoder.encode(frame)[0]
                    neural, _ = run_neural_window(
                        brain, projection, drive, populations, NEURAL_STEPS, cache)
                    records.append({
                        "seed": seed,
                        "condition": condition,
                        "decision": decision,
                        "population": active_record(neural, vector),
                        "frame_sha256": hashlib.sha256(current.tobytes()).hexdigest(),
                    })
                    game.tick(HOLD_FRAMES + RELEASE_FRAMES)
            print("natural calibration", seed, flush=True)
    return records, baselines


def calibration_floor(baselines: dict) -> float:
    residuals = []
    for payload in baselines.values():
        counts = np.asarray(payload["population_baseline_counts"], float)
        rates = counts / WINDOW_SECONDS
        residuals.extend((rates - rates.mean(axis=0)).ravel().tolist())
    return max(.5, float(np.std(residuals, ddof=1)))


def vector_baseline(payload: dict, subset=None) -> FrozenVectorBaseline:
    value = payload["vector"]
    means = np.asarray(value["means_hz"], float)
    variances = np.asarray(value["variances_hz2"], float)
    if subset is not None:
        means, variances = means[subset], variances[subset]
    return FrozenVectorBaseline(
        tuple(map(float, means)), tuple(map(float, variances)), int(value["windows"]))


def population_config(spec: dict, floor_hz: float) -> PopulationEventConfig:
    return PopulationEventConfig(
        signal=spec["signal"],
        event_rule=spec["event_rule"],
        threshold=float(spec["threshold"]),
        sd_floor_hz=float(floor_hz),
        signal_window=int(spec["signal_window"]),
        evidence_window=int(spec["evidence_window"]),
        hysteresis=.5,
        refractory_decisions=5,
    )


def stable_condition_code(condition: str) -> int:
    return int.from_bytes(hashlib.sha256(condition.encode()).digest()[:4], "little")


def grouped_records(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["seed"], row["condition"]].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: row["decision"])
    return groups


def replay_population(records, baselines, spec, floor_hz, *, ablated=False,
                      cell_permutation=False, temporal_shuffle=False, subset=None):
    outputs = []
    for (seed, condition), rows in sorted(grouped_records(records).items()):
        rates = np.asarray([row["population"]["rates_hz"] for row in rows], float)
        if temporal_shuffle:
            rates = permute_population_time(
                rates, 1310000 + seed * 1000 + stable_condition_code(condition) % 997)
        if cell_permutation:
            rates = permute_population_cells(
                rates, 1320000 + seed * 1000 + stable_condition_code(condition) % 997)
        if subset is not None:
            rates = rates[:, subset]
        controller = PopulationEventController(
            vector_baseline(baselines[f"{seed}:{condition}"], subset),
            population_config(spec, floor_hz),
            ablated=ablated,
        )
        for decision, cell_rates in enumerate(rates):
            action, state = controller.decode({
                "event_population_rates_hz": cell_rates.tolist()
            })
            outputs.append({
                "seed": seed, "condition": condition, "decision": decision,
                "action": action, "state": state,
            })
    return outputs


def candidate_grid() -> list[dict]:
    return json.loads((RESULTS / "signal-candidates.json").read_text())["grid"]


def event_rate(rows) -> float:
    return float(np.mean([row["action"] == "A" for row in rows])) if rows else 0.


def longest_run(rows) -> int:
    longest = run = 0
    for row in rows:
        run = run + 1 if row["action"] == "A" else 0
        longest = max(longest, run)
    return longest


def select_decoder(records, baselines, floor_hz):
    scores = []
    for spec in candidate_grid():
        outputs = replay_population(records, baselines, spec, floor_hz)
        by_trial = grouped_records(outputs)
        effects, controlled_gains, consistency = [], [], []
        live_rates, no_rates = [], []
        for seed in CALIBRATION_SEEDS:
            no = event_rate(by_trial[seed, "bedroom_no_vision"])
            live = event_rate(by_trial[seed, "bedroom_live"])
            onset_rates = [event_rate(by_trial[seed, condition])
                           for condition in sorted(CONTROLLED_ONSETS)]
            effects.append(live - no)
            controlled_gains.append(float(np.mean(onset_rates)) - no)
            consistency.append(float(np.mean([rate > no for rate in onset_rates])))
            live_rates.append(live)
            no_rates.append(no)
        suppressions = 0
        max_burst = 0
        for trial in by_trial.values():
            suppressions += max(
                [row["state"]["refractory_suppressions"] for row in trial], default=0)
            max_burst = max(max_burst, longest_run(trial))
        suppression_fraction = suppressions / max(1, len(outputs))
        effect = float(np.mean(effects))
        seed_sd = float(np.std(effects, ddof=1))
        controlled_gain = float(np.mean(controlled_gains))
        controlled_consistency = float(np.mean(consistency))
        positive_seed_fraction = float(np.mean(np.asarray(effects) > 0))
        live = float(np.mean(live_rates))
        no = float(np.mean(no_rates))
        objective = (
            effect + .25 * controlled_gain
            + .01 * controlled_consistency + .01 * positive_seed_fraction
            - .25 * seed_sd
            - 2 * max(0., no - .05) - max(0., live - .10)
            - .01 * suppression_fraction
        )
        scores.append(spec | {
            "live_rate": live,
            "no_vision_rate": no,
            "live_no_effect": effect,
            "controlled_gain": controlled_gain,
            "controlled_consistency": controlled_consistency,
            "positive_seed_fraction": positive_seed_fraction,
            "seed_effect_sd": seed_sd,
            "suppression_fraction": suppression_fraction,
            "max_event_run": max_burst,
            "objective": objective,
            "eligible": bool(no <= .05 and live <= .10 and max_burst <= 1),
        })
    eligible = [score for score in scores if score["eligible"]]
    pool = eligible or scores
    selected = max(
        pool,
        key=lambda score: (
            score["objective"], -score["candidate_order"], score["threshold"]
        ),
    )
    return scores, selected, replay_population(
        records, baselines, selected, floor_hz)


def exact_sign_permutation(values) -> dict:
    values = np.asarray(values, float)
    observed = float(values.mean())
    extreme = 0
    total = 1 << len(values)
    for start in range(0, total, 32768):
        bits = np.arange(start, min(total, start + 32768), dtype=np.uint32)[:, None]
        signs = 1 - 2 * (
            (bits >> np.arange(len(values), dtype=np.uint32)) & 1
        ).astype(np.int8)
        null = (signs * values).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12))
    return {
        "effect": observed,
        "p_value": extreme / total,
        "seed_effects": values.tolist(),
        "permutations": total,
    }


def compare_outputs(outputs, other_condition: str, metric: str) -> dict:
    groups = grouped_records(outputs)
    effects = []
    for seed in HELDOUT_SEEDS:
        def trial_mean(condition):
            rows = groups[seed, condition]
            if metric == "event":
                return event_rate(rows)
            values = [
                row["state"]["evidence"] for row in rows
                if row["state"]["evidence"] is not None
            ]
            return float(np.mean(values)) if values else 0.
        effects.append(trial_mean("bedroom_live") - trial_mean(other_condition))
    return exact_sign_permutation(effects)


def heldout_trial(game, brain, projection, populations, seed, condition,
                  selected, floor_hz):
    game.load_state(STATE)
    game.tick(1)
    initial = game.framebuffer()
    order = np.random.default_rng(130000 + seed).permutation(144 * 160)
    vector, named, rate_windows, cell_windows, counts, cache = baseline_trial(
        brain, projection, populations, seed)
    baseline_data = baseline_payload(vector, named, rate_windows, counts)
    controller = PopulationEventController(
        vector, population_config(selected, floor_hz))
    encoder = OnlineE3Encoder(projection)
    metric_encoder = OnlineE3Encoder(projection)
    initial_vector, _ = metric_encoder.encode(initial)
    previous = None
    rows = []
    for decision in range(NATURAL_HORIZON):
        current = game.framebuffer()
        frame = source_frame(condition, current, initial, order)
        if frame is None:
            drive, injection = np.zeros(len(projection.ids), np.float32), None
        else:
            drive, _ = encoder.encode(frame)
            injection = drive
        neural, _ = run_neural_window(
            brain, projection, injection, populations, NEURAL_STEPS, cache)
        population = active_record(neural, vector, floor_hz)
        action, state = controller.decode({
            "event_population_rates_hz": population["rates_hz"]
        })
        if action:
            game.press("a")
            game.tick(HOLD_FRAMES)
            game.release("a")
        else:
            game.tick(HOLD_FRAMES)
        game.tick(RELEASE_FRAMES)
        rows.append({
            "seed": seed,
            "condition": condition,
            "decision": decision,
            "population": population,
            "action": action,
            "controller_state": state,
            "visual": visual_metrics(
                current, initial, drive, initial_vector, previous),
            "post_frame_sha256": hashlib.sha256(
                game.framebuffer().tobytes()).hexdigest(),
            "baseline": baseline_data if decision == 0 else None,
        })
        previous = current
    return rows


def secondary_trial(brain, projection, populations, seed, condition, frames,
                    selected, floor_hz):
    vector, named, rate_windows, cell_windows, counts, cache = baseline_trial(
        brain, projection, populations, seed)
    baseline_data = baseline_payload(vector, named, rate_windows, counts)
    controller = PopulationEventController(
        vector, population_config(selected, floor_hz))
    encoder = OnlineE3Encoder(projection)
    rows = []
    for decision, frame in enumerate(frames):
        drive, _ = encoder.encode(frame)
        neural, _ = run_neural_window(
            brain, projection, drive, populations, NEURAL_STEPS, cache)
        population = active_record(neural, vector, floor_hz)
        action, state = controller.decode({
            "event_population_rates_hz": population["rates_hz"]
        })
        rows.append({
            "seed": seed, "condition": condition, "decision": decision,
            "population": population, "action": action,
            "controller_state": state,
            "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "baseline": baseline_data if decision == 0 else None,
            "offline_sequence": True,
        })
    return rows


def extract_replay_data(trials):
    records, baselines = [], {}
    for row in trials:
        records.append({
            "seed": row["seed"],
            "condition": row["condition"],
            "decision": row["decision"],
            "population": {"rates_hz": row["population"]["rates_hz"]},
            "dnp01_rates_hz": row["population"]["dnp01"]["rates_hz"],
        })
        if row["baseline"] is not None:
            baselines[f"{row['seed']}:{row['condition']}"] = row["baseline"]
    return records, baselines


def replay_dnp01_reference(records, baselines):
    frozen = json.loads((
        ROOT / "results/experiment-12-dnp01-event/thresholds.json").read_text())
    selected = frozen["selected"]
    floor = float(frozen["sd_floor_hz"])
    outputs = []
    for (seed, condition), rows in sorted(grouped_records(records).items()):
        named = baselines[f"{seed}:{condition}"]["named"]
        mean = float(named["means_hz"]["DNp01"])
        variance = float(named["variances_hz2"]["DNp01"])
        scale = max(math.sqrt(max(0., variance)), floor)
        controller = TransparentEventController(EventDecoderConfig(
            selected["strategy"], "sum", mean, scale,
            selected["threshold_z"], selected["window"],
            frozen["hysteresis_z"], frozen["refractory_decisions"],
        ))
        for decision, row in enumerate(rows):
            action, state = controller.decode({
                "DNp01_cells_hz": row["dnp01_rates_hz"]
            })
            outputs.append({
                "seed": seed, "condition": condition, "decision": decision,
                "action": action, "state": state,
            })
    return outputs


def event_quality(outputs) -> dict:
    groups = grouped_records(outputs)
    live_counts, gaps = [], []
    longest = suppressions = 0
    for seed in HELDOUT_SEEDS:
        rows = groups[seed, "bedroom_live"]
        indices = [row["decision"] for row in rows if row["action"] == "A"]
        live_counts.append(len(indices))
        gaps.extend(b - a for a, b in zip(indices, indices[1:]))
        longest = max(longest, longest_run(rows))
        suppressions += max(
            [row["state"]["refractory_suppressions"] for row in rows], default=0)
    return {
        "events_per_trial": float(np.mean(live_counts)),
        "a_decision_fraction": event_rate([
            row for row in outputs if row["condition"] == "bedroom_live"
        ]),
        "median_inter_event": float(np.median(gaps)) if gaps else None,
        "mean_inter_event": float(np.mean(gaps)) if gaps else None,
        "minimum_inter_event": min(gaps, default=None),
        "longest_event_run": longest,
        "refractory_suppressions": suppressions,
        "trials_with_event_fraction": float(np.mean(np.asarray(live_counts) > 0)),
        "zero_event_trial_fraction": float(np.mean(np.asarray(live_counts) == 0)),
    }


def integration_run(brain, projection, populations, selected, floor_hz, passed):
    if not passed:
        return {
            "enabled": False,
            "reason": "Held-out population A validation did not pass",
            "seeds": [],
            "horizon": 500,
            "records": [],
        }
    e8 = json.loads((
        ROOT / "results/experiment-08-steering/thresholds.json").read_text())
    e9 = json.loads((
        ROOT / "results/experiment-09-locomotion/thresholds.json").read_text())
    config = ExplorationConfig(
        e8["selected_multiplier"], e8["pooled_baseline_sd_floor_hz"],
        e8["left_evidence_side"],
        e9["forward"]["selected_multiplier"], e9["forward"]["sd_floor_hz"],
        e9["backward"]["selected_multiplier"], e9["backward"]["sd_floor_hz"], 1,
    )
    records = []
    with PokemonEmulator() as game:
        for seed in INTEGRATED_SEEDS:
            game.load_state(STATE)
            game.tick(1)
            initial = game.framebuffer()
            vector, named, rate_windows, cell_windows, counts, cache = baseline_trial(
                brain, projection, populations, seed)
            steering = estimate_baseline(rate_windows)
            locomotion = estimate_population_baseline(rate_windows, cell_windows)
            direction = FrozenExplorationController(
                steering, locomotion, config, enable_down=False)
            event = PopulationEventController(
                vector, population_config(selected, floor_hz))
            controller = FrozenInterfaceController(direction, None, event)
            encoder = OnlineE3Encoder(projection)
            metric_encoder = OnlineE3Encoder(projection)
            initial_vector, _ = metric_encoder.encode(initial)
            previous = None
            for decision in range(500):
                current = game.framebuffer()
                drive, _ = encoder.encode(current)
                neural, _ = run_neural_window(
                    brain, projection, drive, populations, NEURAL_STEPS, cache)
                population = active_record(neural, vector, floor_hz)
                rates = neural["controller_rates_hz"] | {
                    "event_population_rates_hz": population["rates_hz"]
                }
                action, state = controller.decode(rates)
                if action:
                    game.press(action.lower())
                    game.tick(HOLD_FRAMES)
                    game.release(action.lower())
                else:
                    game.tick(HOLD_FRAMES)
                game.tick(RELEASE_FRAMES)
                records.append({
                    "seed": seed, "decision": decision, "action": action,
                    "population": population, "controller_state": state,
                    "visual": visual_metrics(
                        current, initial, drive, initial_vector, previous),
                })
                previous = current
            print("integrated", seed, flush=True)
    return {
        "enabled": True,
        "reason": "Held-out population A validation passed",
        "seeds": INTEGRATED_SEEDS,
        "horizon": 500,
        "records": records,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    CAPTURES.mkdir(parents=True, exist_ok=True)
    brain = FlyBrain(data=DATA, device="cuda")
    population_document = load_population()
    ids = [row["flywire_malecns_id"] for row in population_document["neurons"]]
    resolved, frozen_indices = resolve_dn_ids(brain, DATA, ids)
    assert resolved["count"] == population_document["count"]
    assert [row["brain_index"] for row in resolved["neurons"]] == [
        row["brain_index"] for row in population_document["neurons"]
    ]
    motor_metadata, motor_populations = resolve_motor_populations(
        brain, DATA, MOTOR_TYPES)
    populations = dict(motor_populations)
    populations[POPULATION_KEY] = frozen_indices
    projection = SpatialProjection(brain, E7_CONFIG)

    calibration, calibration_baselines = acquire_calibration(
        brain, projection, populations)
    floor_hz = calibration_floor(calibration_baselines)
    scores, selected, selected_outputs = select_decoder(
        calibration, calibration_baselines, floor_hz)
    calibration_artifact = {
        "metadata": {
            "seeds": CALIBRATION_SEEDS,
            "population": population_document["name"],
            "conditions": sorted({row["condition"] for row in calibration}),
        },
        "records": calibration,
        "baselines": calibration_baselines,
        "selected_decoder_outputs": selected_outputs,
    }
    (RESULTS / "calibration.json").write_text(
        json.dumps(calibration_artifact, indent=2) + "\n")
    thresholds = {
        "design_sha256": hashlib.sha256(
            (RESULTS / "design.md").read_bytes()).hexdigest(),
        "population_brain_indices_sha256":
            population_document["ordered_brain_indices_sha256"],
        "calibration_seeds": CALIBRATION_SEEDS,
        "heldout_seeds": HELDOUT_SEEDS,
        "integrated_seeds": INTEGRATED_SEEDS,
        "baseline_windows": BASELINE_WINDOWS,
        "neural_steps_per_decision": NEURAL_STEPS,
        "neural_window_seconds": WINDOW_SECONDS,
        "sd_floor_hz": floor_hz,
        "hysteresis": .5,
        "refractory_decisions": 5,
        "a_hold_frames": HOLD_FRAMES,
        "a_release_frames": RELEASE_FRAMES,
        "selected": selected,
        "grid_scores": scores,
        "frozen_before_heldout": True,
    }
    (RESULTS / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2) + "\n")
    print("frozen selection", json.dumps(selected, sort_keys=True), flush=True)

    trials = []
    with PokemonEmulator() as game:
        for seed in HELDOUT_SEEDS:
            for condition in (
                "bedroom_live", "bedroom_no_vision",
                "bedroom_frozen", "bedroom_shuffled",
            ):
                trials.extend(heldout_trial(
                    game, brain, projection, populations, seed, condition,
                    selected, floor_hz))
            print("heldout", seed, flush=True)

    secondary = []
    sequences = secondary_sequences()
    for seed in HELDOUT_SEEDS:
        for condition, frames in sequences.items():
            secondary.extend(secondary_trial(
                brain, projection, populations, seed, condition, frames,
                selected, floor_hz))
        print("secondary heldout", seed, flush=True)

    replay_records, heldout_baselines = extract_replay_data(trials)
    actual_outputs = [{
        "seed": row["seed"], "condition": row["condition"],
        "decision": row["decision"], "action": row["action"],
        "state": row["controller_state"],
    } for row in trials]
    ablation = replay_population(
        replay_records, heldout_baselines, selected, floor_hz, ablated=True)
    cell_permutation = replay_population(
        replay_records, heldout_baselines, selected, floor_hz,
        cell_permutation=True)
    temporal_shuffle = replay_population(
        replay_records, heldout_baselines, selected, floor_hz,
        temporal_shuffle=True)
    halves = {
        "historical_ranks_1_10": replay_population(
            replay_records, heldout_baselines, selected, floor_hz,
            subset=np.arange(10)),
        "historical_ranks_11_20": replay_population(
            replay_records, heldout_baselines, selected, floor_hz,
            subset=np.arange(10, 20)),
    }
    population_rows = population_document["neurons"]
    type_indices = defaultdict(list)
    for index, row in enumerate(population_rows):
        type_indices[row["cell_type"]].append(index)
    leave_group_out = {}
    diagnostic_types = sorted(
        {cell_type for cell_type, positions in type_indices.items()
         if len(positions) > 1} | {"DNp01"})
    for cell_type in diagnostic_types:
        keep = np.asarray([
            index for index, row in enumerate(population_rows)
            if row["cell_type"] != cell_type
        ], int)
        leave_group_out[cell_type] = replay_population(
            replay_records, heldout_baselines, selected, floor_hz, subset=keep)
    dnp01_reference = replay_dnp01_reference(
        replay_records, heldout_baselines)

    controls = {
        "population_ablation": ablation,
        "cell_permutation": cell_permutation,
        "temporal_shuffle": temporal_shuffle,
        "subpopulation_split": halves,
        "leave_cell_type_out": leave_group_out,
        "dnp01_experiment12_reference": dnp01_reference,
    }
    (RESULTS / "heldout-trials.json").write_text(json.dumps({
        "metadata": {
            "seeds": HELDOUT_SEEDS,
            "conditions": [
                "bedroom_live", "bedroom_no_vision",
                "bedroom_frozen", "bedroom_shuffled",
            ],
            "secondary_conditions": sorted(sequences),
            "horizon": NATURAL_HORIZON,
            "population": population_document["name"],
        },
        "records": trials,
        "secondary_records": secondary,
    }, indent=2) + "\n")
    (RESULTS / "controls.json").write_text(
        json.dumps(controls, indent=2) + "\n")

    signal_live_no = compare_outputs(
        actual_outputs, "bedroom_no_vision", "signal")
    event_live_no = compare_outputs(
        actual_outputs, "bedroom_no_vision", "event")
    quality = event_quality(actual_outputs)
    rates_by_condition = {
        condition: event_rate([
            row for row in actual_outputs if row["condition"] == condition
        ])
        for condition in (
            "bedroom_live", "bedroom_no_vision",
            "bedroom_frozen", "bedroom_shuffled",
        )
    }
    dnp_drop = leave_group_out["DNp01"]
    dnp_drop_test = compare_outputs(
        dnp_drop, "bedroom_no_vision", "event")
    pass_result = (
        signal_live_no["effect"] > 0 and signal_live_no["p_value"] < .05
        and event_live_no["effect"] > 0 and event_live_no["p_value"] < .05
        and rates_by_condition["bedroom_live"] <= .10
        and rates_by_condition["bedroom_no_vision"] <= .05
        and quality["longest_event_run"] <= 1
        and not any(row["action"] == "A" for row in ablation)
        and dnp_drop_test["effect"] > 0 and dnp_drop_test["p_value"] < .05
    )
    integration = integration_run(
        brain, projection, populations, selected, floor_hz, pass_result)
    (RESULTS / "integrated-controller.json").write_text(
        json.dumps(integration, indent=2) + "\n")
    print(json.dumps({
        "selected": selected,
        "floor_hz": floor_hz,
        "signal_live_vs_no": signal_live_no,
        "event_live_vs_no": event_live_no,
        "dnp01_drop_live_vs_no": dnp_drop_test,
        "quality": quality,
        "pass": pass_result,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
