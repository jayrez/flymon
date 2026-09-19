"""Experiment 7 motor-DN calibration; no Game Boy actions are emitted."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
from flybrain import FlyBrain

from flymon.controller import ControllerConfig
from flymon.motor import (E7_CONFIG, OnlineE3Encoder, run_neural_window,
                          synthetic_frame, verify_motor_neurons)
from flymon.spatiotemporal import SpatialProjection
from run_visual_experiment import DATA, ROOT

RESULTS = ROOT / "results/experiment-07-closed-loop"
CAPTURES = ROOT / "captures/experiment-07"
SEEDS = list(range(301, 321))
DECISIONS = 20


def image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


def derive_config(rows: list[dict]) -> tuple[ControllerConfig, dict]:
    no = [r for r in rows if r["condition"] == "no_vision"]
    q = lambda vals: float(np.quantile(vals, .95, method="higher"))
    steer_noise = q([abs(r["rates_hz"]["DNa02_L"] - r["rates_hz"]["DNa02_R"]) for r in no])
    fwd_noise = q([r["rates_hz"]["DNg100"] for r in no])
    back_noise = q([r["rates_hz"]["MDN"] for r in no])
    left = np.mean([r["rates_hz"]["DNa02_L"] - r["rates_hz"]["DNa02_R"]
                    for r in rows if r["condition"] == "left_stimulus"])
    right = np.mean([r["rates_hz"]["DNa02_L"] - r["rates_hz"]["DNa02_R"]
                     for r in rows if r["condition"] == "right_stimulus"])
    side = "L" if left - right >= 0 else "R"
    cfg = ControllerConfig(steering_deadband_hz=max(1., steer_noise + .5),
                           forward_threshold_hz=max(1., fwd_noise + .5),
                           backward_threshold_hz=max(1., back_noise + .5),
                           minimum_activity_hz=1., left_evidence_side=side)
    return cfg, {"no_vision_95pct_abs_steering_difference_hz": steer_noise,
                 "no_vision_95pct_DNg100_hz": fwd_noise, "no_vision_95pct_MDN_hz": back_noise,
                 "left_stimulus_mean_L_minus_R_hz": float(left),
                 "right_stimulus_mean_L_minus_R_hz": float(right),
                 "polarity_rule": "side with larger left-minus-right stimulus contrast maps to LEFT"}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); CAPTURES.mkdir(parents=True, exist_ok=True)
    brain = FlyBrain(data=DATA, device="cuda")
    motor_meta, populations = verify_motor_neurons(brain, DATA)
    frozen_e4 = json.loads((ROOT / "results/experiment-04-temporal-dn/classification.json").read_text())["informative_dn_top20"]
    ids = np.load(DATA / "brain.npz")["ids"]
    id_to_index = {int(body): i for i, body in enumerate(ids)}
    populations["E4_informative_DN"] = np.array([id_to_index[int(row["flywire_id"])] for row in frozen_e4], np.int32)
    motor_meta["exploratory_E4_informative_DN_indices"] = populations["E4_informative_DN"].tolist()
    projection = SpatialProjection(brain, E7_CONFIG)
    aggregate_cache = {}
    frames = {"left_stimulus": synthetic_frame("left"), "right_stimulus": synthetic_frame("right"),
              "uniform": synthetic_frame("uniform"),
              "bedroom": image(ROOT / "captures/vision/bedroom-original.png"),
              "menu": image(ROOT / "captures/vision/new_game_menu-original.png"),
              "dialogue": image(ROOT / "captures/vision/oak_dialogue-original.png")}
    for name, frame in frames.items(): Image.fromarray(frame, "RGBA").save(CAPTURES / f"calibration-{name}.png")
    rows = []
    for seed in SEEDS:
        for condition in ("no_vision", *frames):
            brain.reset(seed=seed); encoder = OnlineE3Encoder(projection)
            for decision in range(DECISIONS):
                vector, stats = (None, {"encoder_sha256": None, "encoder_mean": 0., "encoder_nonzero": 0}) \
                    if condition == "no_vision" else encoder.encode(frames[condition])
                neural, vp = run_neural_window(brain, projection, vector, populations,
                                                E7_CONFIG.neural_steps_per_frame, aggregate_cache)
                rows.append({"seed": seed, "condition": condition, "decision": decision,
                             "rates_hz": neural["controller_rates_hz"],
                             "all_group_rates_hz": neural["group_rates_hz"],
                             "per_cell_counts": neural["per_cell_counts"],
                             "visual_projection_spikes": vp, "encoder": stats})
        print("calibration seed", seed, flush=True)
    config, derivation = derive_config(rows)
    summary = {}
    for condition in ("no_vision", *frames):
        group = [r for r in rows if r["condition"] == condition]
        summary[condition] = {key: {"mean_hz": float(np.mean([r["rates_hz"][key] for r in group])),
                                    "sd_hz": float(np.std([r["rates_hz"][key] for r in group])),
                                    "baseline_delta_hz": float(np.mean([r["rates_hz"][key] for r in group]) -
                                                               np.mean([r["rates_hz"][key] for r in rows if r["condition"] == "no_vision"]))}
                              for key in ["DNa02_L", "DNa02_R", "DNg100", "MDN", "DNp01"]}
    motor_meta["source"] = "local redfly-benchmark/data/brain.npz verified before use"
    (RESULTS / "motor-neurons.json").write_text(json.dumps(motor_meta, indent=2) + "\n")
    (RESULTS / "calibration.json").write_text(json.dumps({"seeds": SEEDS, "decisions": DECISIONS,
        "neural_steps_per_decision": E7_CONFIG.neural_steps_per_frame,
        "simulated_seconds_per_decision": E7_CONFIG.neural_steps_per_frame * brain.dt,
        "threshold_derivation": derivation, "controller_config": asdict(config),
        "summary": summary, "records": rows}, indent=2) + "\n")
    print("controller", asdict(config), flush=True)


if __name__ == "__main__": main()
