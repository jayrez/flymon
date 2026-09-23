"""Experiment 24 / Generation 0: frozen E23 T4 readout in the closed Pokémon Red loop.

Closed loop (one decision = 10 FlyBrain steps = 0.2 s simulated, 12 game frames):

  PyBoy framebuffer (every game frame)
    -> E5 retina column luminance (linear light)
    -> FrozenT4Readout (E23 native model, streamed one game frame at a time)
    -> mean T4 response over the frames since the previous decision
    -> T4Injection: voltage into the matching MaleCNS T4a-d neurons (frozen gain)
    -> FlyBrain LIF MaleCNS v1.0, 10 steps
    -> frozen Experiment-14 controller (E8 DNa02 LEFT/RIGHT, E9 DNg100 UP, E13 P20 A)
    -> Game Boy button: 8 frames held + 4 released

No optimisation of any kind. Controller input is restricted to DN / population spike rates
(`ControllerInputGuard`); RAM telemetry is read only by the evaluator after the action.

Usage:
  run_generation_zero.py episode --condition gen0 --seeds 2401 2402 ... [--horizon N]
  run_generation_zero.py action-distribution      (after gen0; freezes the C0 action law)
  run_generation_zero.py parallel --instances N --horizon H   (spawns N parallel-worker processes)
  run_generation_zero.py replay-check --seed 2401 --horizon H
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import resource
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "experiment-24-generation-zero"
CAPTURES = ROOT / "captures" / "experiment-24"
STATE = ROOT / "states" / "bedroom.state"
SEEDS = tuple(range(2401, 2411))
HORIZON = 1500
CONDITIONS = ("gen0", "c0_random", "c1_no_visual", "c2_shuffled_geometry")
INJECTION_GAIN = 2.049028738843780          # 0.8 / p99 of positive E23 calibration responses
INJECTION_CAP, INJECTION_LEVELS = 0.8, 16
STATUS_EVERY = 25
SCHEMA = "experiment-24.decision.v1"


# ---------------------------------------------------------------- setup
def sensory_setup(geometry="native", seed=None):
    import run_fixed_background_experiment as E22
    from flymon.frozen_t4 import FrozenT4Readout
    ctx, const = E22.contexts()
    proj, label = None, "native"
    if geometry == "shuffled":
        proj = E22.permuted_projections(ctx, 1, 24_000_000 + int(seed))[0]
        label = f"column-shuffle-seed-{24_000_000 + int(seed)}"
    return FrozenT4Readout(ctx, const, proj=proj, geometry_label=label)


def brain_setup():
    from flybrain import FlyBrain
    from flymon.motor import E7_CONFIG, MOTOR_TYPES, resolve_dn_ids, resolve_motor_populations
    from flymon.spatiotemporal import SpatialProjection
    from run_population_event_experiment import POPULATION_KEY, load_population
    from run_visual_experiment import DATA
    doc = load_population()
    if doc["name"] != "E4_E6_frozen_P20" or doc["count"] != 20:
        raise RuntimeError("Experiment 13 frozen population identity changed")
    brain = FlyBrain(data=DATA, device="cuda")
    ids = [row["flywire_malecns_id"] for row in doc["neurons"]]
    _resolved, frozen_indices = resolve_dn_ids(brain, DATA, ids)
    _, motor = resolve_motor_populations(brain, DATA, MOTOR_TYPES)
    populations = dict(motor); populations[POPULATION_KEY] = frozen_indices
    projection = SpatialProjection(brain, E7_CONFIG)       # used only for E14's baseline windows
    return brain, populations, projection


def controller_setup(brain, populations, projection, seed):
    """Frozen E14 controller with its per-trial no-vision baseline (unchanged code path)."""
    from flymon.exploration import FrozenExplorationController, FrozenInterfaceController
    from flymon.interaction import PopulationEventController
    from flymon.locomotion import estimate_population_baseline
    from flymon.steering import estimate_baseline
    from run_population_event_experiment import baseline_trial, population_config
    from run_progression_experiment import load_frozen_configuration
    direction_config, event_spec, event_floor = load_frozen_configuration()
    vector_baseline, _, rate_windows, cell_windows, _, cache = baseline_trial(
        brain, projection, populations, seed)
    direction = FrozenExplorationController(
        estimate_baseline(rate_windows), estimate_population_baseline(rate_windows, cell_windows),
        direction_config, enable_down=False)
    event = PopulationEventController(vector_baseline, population_config(event_spec, event_floor), ablated=False)
    config = dict(direction=asdict(direction_config), event=dict(selected=event_spec, sd_floor_hz=event_floor),
                  source="Experiment 14 frozen controller (E8/E9/E13 thresholds); no parameter changed")
    return FrozenInterfaceController(direction, None, event), vector_baseline, event_floor, cache, config


def neural_window(brain, inject, populations, steps, cache):
    """E14 `run_neural_window` motor/population counting with an explicit inject list."""
    slots = cache.setdefault("motor_slots", {})
    if not slots:
        for typ, idx in populations.items():
            slot = np.full(brain.n, -1, np.int32); slot[idx] = np.arange(len(idx)); slots[typ] = slot
    per_cell = {typ: np.zeros(len(idx), np.int32) for typ, idx in populations.items()}
    for _ in range(steps):
        fired = brain.step(inject=inject)
        for typ, slot in slots.items():
            hit = slot[fired]; hit = hit[hit >= 0]
            per_cell[typ] += np.bincount(hit, minlength=len(per_cell[typ])).astype(np.int32)
    seconds = steps * float(brain.dt)
    rates = {}
    for typ, counts in per_cell.items():
        for side in ("L", "R"):
            mask = brain.side[populations[typ]] == side
            rates[f"{typ}_{side}"] = float(counts[mask].sum() / seconds)
        rates[typ] = float(counts.sum() / seconds)
    return {"per_cell_counts": {k: v.tolist() for k, v in per_cell.items()}, "controller_rates_hz": rates}


# ---------------------------------------------------------------- one episode
def injection_enabled(condition):
    """T4 drives FlyBrain only in gen0 and the shuffled-geometry control."""
    return condition in ("gen0", "c2_shuffled_geometry")


def config_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


def run_episode(condition, seed, horizon=HORIZON, game=None, brain_bits=None, t4=None,
                write_logs=True, video=False, action_law=None, tag=None):
    from flymon.emulator import PokemonEmulator
    from flymon.frozen_t4 import T4Injection
    from flymon import gameplay_eval as ev
    from run_population_event_experiment import HOLD_FRAMES, NEURAL_STEPS, POPULATION_KEY, RELEASE_FRAMES, \
        WINDOW_SECONDS, active_record
    if condition not in CONDITIONS:
        raise ValueError(condition)
    own_game = game is None
    game = game or PokemonEmulator()
    run_id = f"e24-{condition}-seed{seed}" + (f"-{tag}" if tag else "")
    timers = Counter()
    t_start = time.perf_counter()

    uses_brain = condition != "c0_random"
    if uses_brain:
        brain, populations, projection = brain_bits
        controller, vector_baseline, event_floor, cache, ctl_config = controller_setup(
            brain, populations, projection, seed)
        allowed = None
    else:
        rng = np.random.default_rng(seed)
        labels = list(action_law)
        probs = np.array([action_law[k] for k in labels], float); probs /= probs.sum()
        ctl_config = dict(source="C0 random: iid actions from the pooled Generation-0 action law",
                          law=action_law, rng="numpy default_rng(seed)")
    injection = T4Injection(t4.brain_index, INJECTION_GAIN, INJECTION_CAP, INJECTION_LEVELS) if t4 else None
    inject_enabled = injection_enabled(condition)

    game.load_state(STATE)
    game.tick(1)
    initial = game.framebuffer()
    tel0 = ev.read_telemetry(game)
    milestones = ev.MilestoneTracker(); milestones.spawn = (tel0["map"], tel0["x"], tel0["y"])
    loops = ev.LoopDetector()
    frame_buffer = []
    if t4 is not None:
        t4.reset()
        t = time.perf_counter(); frame_buffer.append(t4.update(initial)); timers["t4"] += time.perf_counter() - t

    meta = dict(run_id=run_id, experiment=24, generation=0, condition=condition, seed=seed, horizon=horizon,
                start_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                save_state=dict(path="states/bedroom.state", sha256=hashlib.sha256(STATE.read_bytes()).hexdigest()),
                spawn=dict(map=tel0["map"], x=tel0["x"], y=tel0["y"]),
                sensory=(t4.config_json() | dict(sha256=t4.sha256())) if t4 else None,
                injection=injection.config_json() if injection else None,
                injection_enabled=bool(inject_enabled),
                controller=ctl_config, controller_sha256=config_hash(ctl_config),
                timing=dict(neural_steps=NEURAL_STEPS, window_seconds=WINDOW_SECONDS, hold_frames=HOLD_FRAMES,
                            release_frames=RELEASE_FRAMES, game_fps=60, t4_updates_per_decision=HOLD_FRAMES + RELEASE_FRAMES,
                            frame_skipping=False, asynchronous_buffering=False),
                provenance="Gameplay uses the frozen E23 T4 visual readout. T5 direction selectivity remains "
                           "unresolved and is not represented as a validated motion pathway in this baseline.")
    rows, status_path = [], CAPTURES / "live" / f"{run_id}.json"
    log_path = RESULTS / "episodes" / condition / f"seed-{seed}.jsonl.gz"
    recorder = None
    if video:
        from flymon.progression import FFmpegVideoRecorder
        (CAPTURES / "videos").mkdir(parents=True, exist_ok=True)
        recorder = FFmpegVideoRecorder(CAPTURES / "videos" / f"{run_id}.mp4", fps=5.0).__enter__()
    if write_logs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gzip.open(log_path, "wt") if write_logs else None
    if writer:
        writer.write(json.dumps(dict(schema=SCHEMA, meta=meta)) + "\n")
    guard = None
    try:
        for decision in range(horizon):
            # ---- sensory summary over the frames since the previous decision ----
            t4_mean = np.mean(frame_buffer, axis=0) if frame_buffer else None
            frame_buffer = []
            inject = injection.pairs(t4_mean) if (inject_enabled and t4_mean is not None) else ()
            # ---- neural + controller ----
            dn = {}
            if uses_brain:
                t = time.perf_counter()
                neural = neural_window(brain, inject, populations, NEURAL_STEPS, cache)
                timers["brain"] += time.perf_counter() - t
                t = time.perf_counter()
                population = active_record(neural, vector_baseline, event_floor)
                rates = neural["controller_rates_hz"] | {"event_population_rates_hz": population["rates_hz"]}
                if guard is None:
                    guard = ev.ControllerInputGuard(rates.keys())
                action, state = controller.decode(guard.check(rates))
                timers["controller"] += time.perf_counter() - t
                z = np.asarray(population["z"], float)
                d = state["direction"]
                dn = dict(DNa02_L=rates["DNa02_L"], DNa02_R=rates["DNa02_R"], DNg100=rates["DNg100"],
                          steer=float(d["steering_signal_hz"]), fwd=float(d["forward_delta_hz"]),
                          p20_rms_z=float(np.sqrt(np.mean(z ** 2))), reason=("A-event" if action == "A" else d["reason"]))
            else:
                action = labels[int(rng.choice(len(labels), p=probs))]
                action = None if action == "NONE" else action
            if action == "DOWN":
                raise RuntimeError("DOWN is disabled")
            # ---- act, streaming every game frame through the frozen T4 model ----
            def tick_one():
                t = time.perf_counter(); game.tick(1); timers["emulator"] += time.perf_counter() - t
                if t4 is not None:
                    t = time.perf_counter(); frame_buffer.append(t4.update(game.framebuffer()))
                    timers["t4"] += time.perf_counter() - t
            if action is not None:
                game.press(action.lower())
            for _ in range(HOLD_FRAMES):
                tick_one()
            if action is not None:
                game.release(action.lower())
            for _ in range(RELEASE_FRAMES):
                tick_one()
            # ---- evaluator (after the action; never fed back) ----
            t = time.perf_counter()
            tel = ev.read_telemetry(game)
            new_m = milestones.update(decision, action, tel)
            flags = loops.update(action, tel)
            timers["evaluator"] += time.perf_counter() - t
            if recorder:
                recorder.write(game.framebuffer()[..., :3])
            row = dict(d=decision, action=action, map=tel["map"], x=tel["x"], y=tel["y"], window=tel["window"],
                       battle=tel["battle"], new_milestones=new_m, loops=[k for k, v in flags.items() if v])
            if t4_mean is not None:
                row["t4"] = dict(mean=float(t4_mean.mean()), max=float(t4_mean.max()),
                                 active_frac=float((t4_mean > 0).mean()),
                                 injected_cells=int(sum(len(i) for i, _ in inject)),
                                 sub=[float(t4_mean[t4.subtype == s].mean()) for s in ("T4a", "T4b", "T4c", "T4d")])
            if dn:
                row["dn"] = dn
            rows.append(row)
            if writer:
                writer.write(json.dumps(row, separators=(",", ":")) + "\n")
            if (decision + 1) % STATUS_EVERY == 0 or decision + 1 == horizon:
                el = decision + 1
                atomic_json(status_path, dict(
                    run_id=run_id, generation=0, condition=condition, seed=seed, decision=el, horizon=horizon,
                    elapsed_game_s=el * (HOLD_FRAMES + RELEASE_FRAMES) / 60.0, last_action=action,
                    actions_per_minute=sum(r["action"] is not None for r in rows) / (el * 12 / 60 / 60),
                    current_milestone=milestones.best_name(),
                    unique_tiles=len({(r["map"], r["x"], r["y"]) for r in rows}),
                    unique_maps=len({r["map"] for r in rows}),
                    t4_aggregate=row.get("t4", {}).get("mean"), controller_output=dn or None,
                    loop_state=row["loops"], wall_s=time.perf_counter() - t_start))
    finally:
        if writer:
            writer.close()
        if recorder:
            recorder.__exit__(None, None, None)
        if own_game:
            game.close()
    flags = loops.finalise()
    game_minutes = horizon * (HOLD_FRAMES + RELEASE_FRAMES) / 60.0 / 60.0
    metrics = ev.episode_metrics(rows, flags, milestones, game_minutes)
    wall = time.perf_counter() - t_start
    perf = dict(wall_s=wall, decisions_per_s=horizon / wall, game_fps=horizon * 12 / wall,
                speed_vs_realtime=(horizon * 12 / 60.0) / wall,
                stage_s={k: float(v) for k, v in timers.items()},
                stage_ms_per_decision={k: 1e3 * float(v) / horizon for k, v in timers.items()},
                max_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    summary = dict(meta=meta, metrics=metrics, performance=perf,
                   t4_mean_over_episode=float(np.mean([r["t4"]["mean"] for r in rows])) if t4 else None,
                   log=str(log_path.relative_to(ROOT)) if write_logs else None,
                   log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest() if write_logs else None)
    if write_logs:
        atomic_json(RESULTS / "runs" / condition / f"seed-{seed}.json", summary)
    return summary, rows


# ---------------------------------------------------------------- CLI stages
def stage_episode(condition, seeds, horizon, video, smoke=False):
    from flymon.emulator import PokemonEmulator
    brain_bits = brain_setup() if condition != "c0_random" else None
    law = None
    if condition == "c0_random":
        law = json.loads((RESULTS / "c0-action-law.json").read_text())["law"]
    with PokemonEmulator() as game:
        for seed in seeds:
            t4 = None
            if condition in ("gen0", "c1_no_visual"):
                t4 = sensory_setup("native")
            elif condition == "c2_shuffled_geometry":
                t4 = sensory_setup("shuffled", seed)
            s, _ = run_episode(condition, seed, horizon, game=game, brain_bits=brain_bits, t4=t4,
                               video=video and condition == "gen0", action_law=law,
                               write_logs=not smoke, tag="smoke" if smoke else None)
            if smoke:
                atomic_json(RESULTS / "smoke" / f"{condition}-seed{seed}.json", s)
            m = s["metrics"]
            print(f"[{condition} {seed}] actions {m['total_actions']} tiles {m['unique_tiles']} maps "
                  f"{m['unique_maps']} best {m['best_milestone']} loop {m['loop_fraction']:.2f} "
                  f"idle {m['idle_fraction']:.2f} wall {s['performance']['wall_s']:.0f}s", flush=True)


def stage_action_law():
    counts = Counter()
    for p in sorted((RESULTS / "runs" / "gen0").glob("seed-*.json")):
        counts.update(json.loads(p.read_text())["metrics"]["action_counts"])
    total = sum(counts.values())
    law = {k: v / total for k, v in sorted(counts.items())}
    atomic_json(RESULTS / "c0-action-law.json", dict(
        note="pooled Generation-0 action frequencies; frozen before any C0 episode", counts=dict(counts), law=law))
    print(law)


def stage_replay_check(seed, horizon):
    from flymon.emulator import PokemonEmulator
    bits = brain_setup(); t4 = sensory_setup("native")
    out = []
    with PokemonEmulator() as game:
        for rep in range(2):
            s, rows = run_episode("gen0", seed, horizon, game=game, brain_bits=bits, t4=t4, write_logs=False)
            out.append([(r["action"], r["map"], r["x"], r["y"]) for r in rows])
    same = out[0] == out[1]
    first_diff = next((i for i, (a, b) in enumerate(zip(*out)) if a != b), None)
    res = dict(seed=seed, horizon=horizon, identical=same, first_divergence=first_diff,
               actions=[Counter(a for a, *_ in o) for o in out])
    atomic_json(RESULTS / "replay-check.json", res)
    print(res)


def stage_parallel_worker(instances, horizon, rank):
    from flymon.emulator import PokemonEmulator
    bits = brain_setup(); t4 = sensory_setup("native")
    with PokemonEmulator() as game:
        s, _ = run_episode("gen0", SEEDS[rank % len(SEEDS)], horizon, game=game, brain_bits=bits, t4=t4,
                           write_logs=False, tag=f"par{instances}-{rank}")
    atomic_json(RESULTS / "parallel" / f"n{instances}-rank{rank}.json", s["performance"])


def stage_parallel(instances, horizon):
    """Launch N isolated worker processes concurrently; sample GPU memory while they run."""
    import subprocess, sys
    gpu = []
    procs = [subprocess.Popen([sys.executable, __file__, "parallel-worker", "--instances", str(instances),
                               "--horizon", str(horizon), "--rank", str(r)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for r in range(instances)]
    t0 = time.perf_counter()
    while any(p.poll() is None for p in procs):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.split(",")
            gpu.append((float(out[0]), float(out[1])))
        except Exception:
            pass
        time.sleep(2)
    codes = [p.returncode for p in procs]
    path = RESULTS / "parallel" / "gpu-memory.json"
    doc = json.loads(path.read_text()) if path.exists() else {}
    doc[str(instances)] = dict(return_codes=codes, wall_s=time.perf_counter() - t0,
                               peak_gpu_mem_mib=max((g[0] for g in gpu), default=None),
                               mean_gpu_util_pct=float(np.mean([g[1] for g in gpu])) if gpu else None)
    atomic_json(path, doc)
    print(instances, doc[str(instances)])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("episode"); e.add_argument("--condition", required=True, choices=CONDITIONS)
    e.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS)); e.add_argument("--horizon", type=int, default=HORIZON)
    e.add_argument("--video", action="store_true"); e.add_argument("--smoke", action="store_true")
    sub.add_parser("action-distribution")
    r = sub.add_parser("replay-check"); r.add_argument("--seed", type=int, default=SEEDS[0]); r.add_argument("--horizon", type=int, default=200)
    q = sub.add_parser("parallel"); q.add_argument("--instances", type=int); q.add_argument("--horizon", type=int, default=150)
    w = sub.add_parser("parallel-worker"); w.add_argument("--instances", type=int); w.add_argument("--horizon", type=int)
    w.add_argument("--rank", type=int)
    a = ap.parse_args()
    if a.cmd == "episode":
        stage_episode(a.condition, a.seeds, a.horizon, a.video, a.smoke)
    elif a.cmd == "action-distribution":
        stage_action_law()
    elif a.cmd == "replay-check":
        stage_replay_check(a.seed, a.horizon)
    elif a.cmd == "parallel":
        stage_parallel(a.instances, a.horizon)
    elif a.cmd == "parallel-worker":
        stage_parallel_worker(a.instances, a.horizon, a.rank)


if __name__ == "__main__":
    main()
