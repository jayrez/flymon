"""Experiment 7: genuine framebuffer -> MaleCNS -> DN -> PyBoy loop."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from flybrain import FlyBrain

from flymon.controller import ControllerConfig, FixedDNController
from flymon.emulator import PokemonEmulator
from flymon.motor import (E7_CONFIG, OnlineE3Encoder, run_neural_window,
                          synthetic_frame, verify_motor_neurons, sha256_array)
from flymon.spatiotemporal import SpatialProjection
from run_visual_experiment import DATA, ROOT

RESULTS = ROOT / "results/experiment-07-closed-loop"
CAPTURES = ROOT / "captures/experiment-07"
STATE = ROOT / "states/bedroom.state"
SEEDS = list(range(321, 341))
HORIZON = 40
HOLD_FRAMES = 8
RELEASE_FRAMES = 4
CONDITIONS = ("live_vision", "no_vision", "frozen_frame", "shuffled_vision", "controller_null")


def shuffled(frame: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(70000 + seed)
    order = rng.permutation(frame.shape[0] * frame.shape[1])
    out = frame.reshape(-1, 4)[order].reshape(frame.shape).copy()
    out[..., 3] = 255
    return out


def causal_source(condition: str, current: np.ndarray, initial: np.ndarray, seed: int):
    if condition == "no_vision": return None
    if condition == "frozen_frame": return initial
    if condition == "shuffled_vision": return shuffled(current, seed)
    return current


def timeline_png(records: list[dict], path: Path) -> None:
    colors = {None: "#999999", "LEFT": "#377eb8", "RIGHT": "#e41a1c", "UP": "#4daf4a", "DOWN": "#984ea3"}
    im = Image.new("RGB", (900, 260), "white"); d = ImageDraw.Draw(im)
    d.text((12, 8), "Experiment 7 selected trial: action and DN rate timeline", fill="black")
    max_rate = max(1., max(max(r["candidate_dn_rates_hz"].values()) for r in records))
    keys = ["DNa02_L", "DNa02_R", "DNg100", "MDN"]
    line_colors = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3"]
    for i, r in enumerate(records):
        x = 20 + i * 21
        d.rectangle((x, 35, x + 16, 58), fill=colors[r["applied_action"]])
    for key, color in zip(keys, line_colors):
        pts = []
        for i, r in enumerate(records):
            x = 28 + i * 21; y = 240 - 160 * r["candidate_dn_rates_hz"][key] / max_rate
            pts.append((x, y))
        if len(pts) > 1: d.line(pts, fill=color, width=2)
    im.save(path)


def run_trial(game, brain, projection, populations, config, seed, condition,
              horizon=HORIZON, stimulus: np.ndarray | None = None,
              intervention: str | None = None, repeat=0):
    game.load_state(STATE); game.tick(1)
    initial = game.framebuffer(); brain.reset(seed=seed)
    encoder = OnlineE3Encoder(projection); controller = FixedDNController(config); cache = {}
    records = []
    for decision in range(horizon):
        current = game.framebuffer()
        source = stimulus if stimulus is not None else causal_source(condition, current, initial, seed)
        if source is None:
            vector = None
            enc = {"grid_sha256": None, "grid_mean": None, "grid_std": None,
                   "change_mean": None, "encoder_sha256": None, "encoder_mean": 0., "encoder_nonzero": 0}
        else: vector, enc = encoder.encode(source)
        neural, vp = run_neural_window(brain, projection, vector, populations,
                                        E7_CONFIG.neural_steps_per_frame, cache)
        rates = dict(neural["controller_rates_hz"])
        if intervention == "zero_DNa02": rates["DNa02_L"] = rates["DNa02_R"] = 0.
        elif intervention == "zero_DNg100": rates["DNg100"] = 0.
        elif intervention == "zero_MDN": rates["MDN"] = 0.
        selected, controller_state = controller.decode(rates)
        applied = None if condition == "controller_null" else selected
        if applied:
            game.press(applied.lower()); game.tick(HOLD_FRAMES); game.release(applied.lower())
        else: game.tick(HOLD_FRAMES)
        game.tick(RELEASE_FRAMES)
        post = game.framebuffer()
        records.append({"trial": f"{condition}-seed-{seed}-repeat-{repeat}", "seed": seed,
            "condition": condition, "intervention": intervention, "decision_index": decision,
            "pyboy_frame_sha256": sha256_array(current), "source_frame_sha256": None if source is None else sha256_array(source),
            "visual_encoder": enc, "LC10a_activity": {k: v for k, v in vp.items() if k.startswith("LC10a")},
            "LPLC2_activity": {k: v for k, v in vp.items() if k.startswith("LPLC2")},
            "candidate_dn_counts": neural["per_cell_counts"], "candidate_dn_rates_hz": neural["controller_rates_hz"],
            "controller_input_rates_hz": rates, "smoothed_controller_state": controller_state,
            "selected_action": selected, "applied_action": applied,
            "pyboy_frames_action_held": HOLD_FRAMES if applied else 0,
            "pyboy_release_frames": RELEASE_FRAMES, "post_action_framebuffer_sha256": sha256_array(post)})
        if condition == "live_vision" and seed == SEEDS[0] and repeat == 0 and decision % 4 == 0:
            Image.fromarray(post, "RGBA").save(CAPTURES / f"live-seed-{seed}-frame-{decision:02d}.png")
    return records


def main() -> None:
    calibration = json.loads((RESULTS / "calibration.json").read_text())
    config = ControllerConfig(**calibration["controller_config"])
    brain = FlyBrain(data=DATA, device="cuda")
    _, populations = verify_motor_neurons(brain, DATA); projection = SpatialProjection(brain, E7_CONFIG)
    trials, assays, interventions, repeats = [], [], [], []
    with PokemonEmulator() as game:
        for seed in SEEDS:
            for condition in CONDITIONS:
                trials.extend(run_trial(game, brain, projection, populations, config, seed, condition))
            print("closed-loop seed", seed, flush=True)
        for seed in range(301, 321):
            for name in ("left", "right"):
                assays.extend(run_trial(game, brain, projection, populations, config, seed,
                                        f"{name}_stimulus", stimulus=synthetic_frame(name)))
        for seed in SEEDS[:10]:
            for intervention in ("zero_DNa02", "zero_DNg100", "zero_MDN"):
                interventions.extend(run_trial(game, brain, projection, populations, config, seed,
                                               "live_vision", intervention=intervention))
        for seed in SEEDS[:5]:
            repeats.extend(run_trial(game, brain, projection, populations, config, seed,
                                     "live_vision", repeat=1))
    metadata = {"model": "flybrain 0.1.0 MaleCNS v1.0", "seeds": SEEDS, "calibration_seeds": list(range(301,321)),
        "conditions": CONDITIONS, "decision_horizon": HORIZON, "neural_steps_per_decision": E7_CONFIG.neural_steps_per_frame,
        "simulated_neural_seconds_per_decision": E7_CONFIG.neural_steps_per_frame * brain.dt,
        "pyboy_frames_per_action": HOLD_FRAMES, "release_frames": RELEASE_FRAMES,
        "controller_config": asdict(config), "bedroom_state_sha256": hashlib.sha256(STATE.read_bytes()).hexdigest()}
    (RESULTS / "trials.json").write_text(json.dumps({"metadata": metadata, "records": trials}, indent=2) + "\n")
    (RESULTS / "controls.json").write_text(json.dumps({"metadata": metadata, "left_right_assay": assays,
        "interventions": interventions, "same_seed_repeats": repeats}, indent=2) + "\n")
    selected = [r for r in trials if r["condition"] == "live_vision" and r["seed"] == SEEDS[0]]
    timeline_png(selected, CAPTURES / "action-dn-controller-timeline.png")
    print("saved Experiment 7 causal records", flush=True)


if __name__ == "__main__": main()
