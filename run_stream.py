"""E30 live stream runtime: supervisor + watchdog + telemetry server + simulation worker.

    POKEMON_ROM=... python run_stream.py --mode broadcast            # OBS browser source: http://127.0.0.1:8765/overlay
    POKEMON_ROM=... python run_stream.py --mode research             # dashboard:          http://127.0.0.1:8765/dashboard
    POKEMON_ROM=... python run_stream.py --hours 6 --soak-out DIR    # unattended soak test

Processes
  simulation worker  PyBoy + frozen T4 pathway + frozen E25 controller + evaluator (RAM read after actions).
                     Sends telemetry out through a queue; receives only episode assignments (id, seed) and an
                     abort flag. It never receives telemetry, milestones or watchdog state.
  supervisor         consumes telemetry, persists stream state (SQLite), serves the overlay (GET only) and runs
                     the watchdog. The watchdog may restart the worker, end an episode or rotate the seed; it never
                     presses buttons, advances dialogue, moves the player or edits emulator memory.
The controller is frozen (no training during live runtime). There is no mid-episode resume: a crashed
episode is recorded as interrupted and the next episode starts from the approved start state.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "e30-stream-readiness"
CONFIG_PATH = OUT / "runtime-config.json"


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def default_config():
    from flymon.live_runtime import CHAMPION_PATH, CHAMPION_SHA256
    rom = os.environ.get("POKEMON_ROM")
    return dict(
        schema=1,
        controller=dict(genome_sha256=CHAMPION_SHA256, file=str(CHAMPION_PATH.relative_to(ROOT)),
                        file_sha256=sha256_file(CHAMPION_PATH), architecture="E25 T4-only linear softmax, 253 params",
                        frozen=True, learning_during_runtime="none"),
        vision="normal",
        episode=dict(max_decisions=3000, max_wall_s=900, timing="8 frames held + 4 released per decision (E24/E25)",
                     realtime=True, fps=60.0, start_wait="uniform 0-59 frames from rng([seed, 2500]) (E25)",
                     policy_rng="rng([seed, 2501]) (E25)"),
        seeds=dict(base=40001, rule="seed = base + all-time episode index (persisted in SQLite); disjoint from E25 "
                                    "training 2501-2508 and held-out 2601-2620"),
        start_state=dict(path="states/bedroom.state", sha256=sha256_file(ROOT / "states" / "bedroom.state"),
                         policy="every episode (and every recovery) restarts from this approved state; no mid-episode resume"),
        rom=dict(env="POKEMON_ROM", sha256=sha256_file(rom) if rom and Path(rom).is_file() else None, committed=False),
        watchdog=dict(heartbeat_timeout_s=60, startup_grace_s=240, frozen_frame_decisions=600,
                      stuck_state_decisions=900, restart_backoff_s=2.0,
                      actions_allowed=["restart crashed/frozen worker", "end episode early", "rotate seed", "log + mark"],
                      actions_forbidden=["press any button", "advance dialogue", "move/teleport player", "edit RAM"]),
        telemetry=dict(host="127.0.0.1", port=8765, frame_every_decisions=1, frame_scale=4, jpeg_quality=85),
        soak_plan=dict(duration_h=6.0, mode="real time (60 fps, 5 decisions/s), one continuous supervisor run",
                       fault_injection=[[3600, "kill (SIGKILL worker: simulated crash)"], [10800, "stop (SIGSTOP worker: simulated hang)"]],
                       pass_rule="S3: supervisor runs the full duration unattended; every crash/hang recovered automatically "
                                 "(no human action); no unrecovered stall; >= 1 completed episode per hour; memory growth of the "
                                 "worker < 200 MB between the first and last hour",
                       metrics=["uptime", "crashes", "restarts", "watchdog interventions", "episodes completed",
                                "episode duration", "progress distribution", "longest uninterrupted runtime", "worker/supervisor RSS",
                                "GPU memory / utilisation", "worker CPU", "decisions/s", "speed ratio"]),
        runtime_dir="runtime/stream", restart_conditions=["worker process exit", "heartbeat older than heartbeat_timeout_s",
                                                          "frozen frame for frozen_frame_decisions (episode ends)",
                                                          "identical (map,x,y,window) for stuck_state_decisions (episode ends)",
                                                          "episode wall clock > max_wall_s (episode ends)"])


def load_config():
    return json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else default_config()


def scientific_provenance(cfg=None):
    cfg = cfg or load_config()
    return dict(
        title="Fly Brain Plays Pokémon",
        one_liner="Pokémon frames are converted into activity in a frozen fly-derived visual pathway; an externally "
                  "evolved controller reads that activity and presses buttons.",
        vision="Frozen connectome-derived T4 motion-vision model (E23 native candidate: MaleCNS v1.0 Mi1/Mi4/Tm -> T4 "
               "weights, E5 retina sampling); sensory SHA-256 87829806e398f66b...",
        brain_simulation="In the live loop: the frozen T4 optic-lobe model only. The whole-brain MaleCNS/FlyBrain spiking "
                         "simulation is NOT run by this controller (the evolved T4-only architecture does not use it).",
        controller="Externally evolved linear softmax decoder (253 parameters) reading 14 pooled T4 features, their "
                   "changes and the controller's own previous action",
        training="Evolutionary search outside the fly model (E25: 38 generations, population 24-32), finished before the stream",
        within_run_learning="None. The controller and the visual model are frozen; nothing is trained while live.",
        ram="Evaluation / overlay / watchdog only. Never a controller input (tested).",
        biological_learning_research="E26-E29 tried dopamine-gated mushroom-body learning; stopped after held-out "
                                     "Kenyon-cell visual coding failed (E29: NO BROADER CIRCUIT).",
        allowed_claims=["A simulated fly visual system is controlling Pokémon through an evolved neural readout.",
                        "Pokémon frames are converted into activity in a fly-derived visual pathway; an evolved controller "
                        "reads that activity and presses buttons."],
        disallowed_claims=["The fly learned Pokémon.", "The connectome evolved / discovered the controller.",
                           "The brain trained itself.", "Dopamine is teaching the live agent.", "E25 is biological learning."],
        known_limitations=["E25 champion is not conventionally better than uniform-random button presses on aggregate fitness "
                           "(E25 p = 0.06)", "never reached Route 1 or a battle in evaluation",
                           "loops on Oak's Lab starter / exit dialogues"],
        controller_sha256=cfg["controller"]["genome_sha256"], start_state_sha256=cfg["start_state"]["sha256"])


# ================================================================ simulation worker
def worker_main(cfg, out_q, assign_q, abort_evt, stop_evt):
    import numpy as np
    from flymon import evolution as evo
    from flymon.emulator import PokemonEmulator
    from flymon.live_runtime import Episode, FrozenController, SensoryContext, frame_sha1, load_champion
    from flymon.stream_eval import StreamEvaluator
    from flymon.telemetry import SCHEMA_VERSION, encode_jpeg
    sensory = SensoryContext(); game = PokemonEmulator()
    genome = load_champion(expected=cfg["controller"]["genome_sha256"])
    ctl = FrozenController(genome, sensory, cfg["vision"])
    tcfg = cfg["telemetry"]; ecfg = cfg["episode"]
    t_up = time.time()
    while not stop_evt.is_set():
        try:
            eid, seed = assign_q.get(timeout=1)
        except queue.Empty:
            continue
        abort_evt.clear()
        ep = Episode(game, ctl, seed, ecfg["max_decisions"], evaluator_factory=StreamEvaluator,
                     realtime=ecfg["realtime"], fps=ecfg["fps"])
        ep.start()
        out_q.put(dict(type="episode_start", episode=eid, seed=seed, controller_sha256=ctl.sha256,
                       sensory_sha256=ctl.sensory_sha256(), start_wait_frames=ep.start_wait, ts=time.time()))
        recent = collections.deque(maxlen=12); t0 = time.time(); reason = "budget"
        rate_t, rate_n, dps = time.time(), 0, 0.0
        while not ep.done:
            if stop_evt.is_set():
                reason = "shutdown"; break
            if abort_evt.is_set():
                reason = "watchdog_abort"; break
            dec, tel, snap = ep.step()
            a = dec.action or "NONE"; recent.append(a); rate_n += 1
            now = time.time()
            if now - rate_t >= 2.0:
                dps = rate_n / (now - rate_t); rate_t, rate_n = now, 0
            fh = frame_sha1(ep.last_frame)
            msg = dict(type="decision", schema=SCHEMA_VERSION, ts=now, episode=eid, seed=seed, decision=ep.decision,
                       episode_runtime_s=now - t0, uptime_s=now - t_up, decisions_per_second=dps,
                       speed_ratio=dps * 12 / 60.0, action=a, recent_actions=list(recent),
                       controller=dict(top_action=max(dec.probabilities, key=dec.probabilities.get), probabilities=dec.probabilities,
                                       features=dec.features, sha256=ctl.sha256, frozen=True, kind="E25 T4 linear softmax"),
                       t4=dec.t4_summary, visual_pathway=dict(status="ok", mode=cfg["vision"], sensory_sha256=ctl.sensory_sha256()),
                       game={k: snap[k] for k in ("map", "map_name", "x", "y", "window", "battle", "window_run")},
                       milestone=dict(current=snap["furthest_milestone"], furthest_episode=snap["furthest_milestone"]),
                       unique_tiles=snap["unique_tiles"], watchdog={}, frame_sha1=fh)
            try:
                out_q.put_nowait(msg)
                if ep.decision % tcfg["frame_every_decisions"] == 0:
                    out_q.put_nowait(("frame", encode_jpeg(ep.last_frame, tcfg["frame_scale"], tcfg["jpeg_quality"])))
            except queue.Full:
                pass
        res = ep.evaluator.result()
        out_q.put(dict(type="episode_end", episode=eid, seed=seed, decisions=ep.decision, fitness=res["fitness"],
                       furthest_milestone=res["furthest_stream_milestone"], stream_milestones=res["stream_milestones"],
                       e25_milestones=res["milestones"], unique_tiles=res["unique_tiles"], end_reason=reason,
                       dialogue={k: res["dialogue"][k] for k in ("n_long", "counts", "decisions", "genuine_stall_decisions",
                                                                 "any_genuine_stall")},
                       action_counts=res["action_counts"], wall_s=time.time() - t0, ts=time.time()))
    game.close()


# ================================================================ supervisor
def _proc_rss_mb(pid):
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return None


def _proc_cpu_ticks(pid):
    try:
        with open(f"/proc/{pid}/stat") as fh:
            f = fh.read().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])
    except (OSError, IndexError):
        return None


def _gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        u, m, t = (float(x) for x in out.split(","))
        return dict(util_pct=u, mem_used_mb=m, mem_total_mb=t)
    except Exception:
        return None


class Supervisor:
    def __init__(self, cfg, runtime_dir, hours=None, faults=(), port=None, mp_context="spawn", worker_target=worker_main,
                 timeseries_path=None, sample_every_s=30.0, log=print):
        from flymon.stream_state import StreamStore
        from flymon.telemetry import TelemetryHub, TelemetryServer
        self.cfg, self.hours, self.faults = cfg, hours, sorted(faults)
        self.ctx = mp.get_context(mp_context); self.worker_target = worker_target
        self.store = StreamStore(Path(runtime_dir) / "stream-state.sqlite")
        self.hub = TelemetryHub(scientific_provenance(cfg))
        port = cfg["telemetry"]["port"] if port is None else port
        self.server = TelemetryServer(self.hub, cfg["telemetry"]["host"], port)
        self.timeseries_path = timeseries_path; self.sample_every_s = sample_every_s; self.log = log
        self.worker = None; self.stop_evt = self.ctx.Event(); self.restarts = 0; self.interventions = []
        self.current = None; self.last_watchdog = None; self.pending = None

    # ---------------------------------------------------------------- worker lifecycle
    def spawn(self):
        self.out_q = self.ctx.Queue(maxsize=512); self.assign_q = self.ctx.Queue(maxsize=1)
        self.abort_evt = self.ctx.Event()
        self.worker = self.ctx.Process(target=self.worker_target, args=(self.cfg, self.out_q, self.assign_q, self.abort_evt,
                                                                         self.stop_evt), daemon=True)
        self.worker.start(); self.worker_started = time.time(); self.last_beat = None
        self.cpu_prev = (time.time(), _proc_cpu_ticks(self.worker.pid))
        if getattr(self, "pending", None) is not None:                     # re-offer the untaken assignment
            self.assign_q.put((self.pending["id"], self.pending["seed"]))
        else:
            self.assign_next()

    def assign_next(self):
        seed = self.store.next_seed(self.cfg["seeds"]["base"])
        eid = self.store.start_episode(seed, self.cfg["controller"]["genome_sha256"])
        self.pending = dict(id=eid, seed=seed)
        self.assign_q.put((eid, seed))

    def watchdog(self, kind, detail, restart=False, abort=False):
        self.store.event(kind, detail)
        self.interventions.append(dict(ts=time.time(), kind=kind, detail=detail))
        self.last_watchdog = dict(kind=kind, ts=time.time())
        self.log(f"[watchdog] {kind} {detail}")
        self.hub.publish(dict(type="watchdog", kind=kind, detail=detail, ts=time.time()))
        if abort:
            self.abort_evt.set()
        if restart:
            if self.worker.is_alive():
                self.worker.kill()
            self.worker.join(10)
            if self.current is not None:                                    # interrupted, not resumed
                self.store.end_episode(self.current["id"], self.current.get("decision", 0), None, self.current.get("furthest"),
                                       kind, True, time.time() - self.current["t0"])
                self.current = None
            time.sleep(self.cfg["watchdog"]["restart_backoff_s"])
            self.store.event("worker_restart", dict(after=kind)); self.restarts += 1
            self.spawn()

    # ---------------------------------------------------------------- message handling
    def handle(self, m):
        now = time.time()
        if isinstance(m, tuple) and m[0] == "frame":
            self.hub.publish_frame(m[1]); return
        t = m["type"]
        if t == "episode_start":
            self.current = dict(id=m["episode"], seed=m["seed"], t0=now, same_frame=0, same_state=0, prev_frame=None,
                                prev_state=None, decision=0, furthest="bedroom", aborted=None)
            self.pending = None
            self.last_beat = now
            self.assign_next()
        elif t == "decision":
            c = self.current; self.last_beat = now
            if c is None:
                return
            c["decision"] = m["decision"]
            c["same_frame"] = c["same_frame"] + 1 if m["frame_sha1"] == c["prev_frame"] else 0
            st = (m["game"]["map"], m["game"]["x"], m["game"]["y"], m["game"]["window"])
            c["same_state"] = c["same_state"] + 1 if st == c["prev_state"] else 0
            c["prev_frame"], c["prev_state"] = m["frame_sha1"], st
            f = m["milestone"]["furthest_episode"]
            if f != c["furthest"]:
                c["furthest"] = f
            if self.store.milestone(f, c["id"], c["seed"]):
                self.hub.publish(dict(type="milestone_first", name=f, episode=c["id"], seed=c["seed"], ts=now))
            w = self.cfg["watchdog"]
            if c["aborted"] is None:
                if c["same_frame"] >= w["frozen_frame_decisions"]:
                    c["aborted"] = "watchdog_abort_frozen_frame"
                elif c["same_state"] >= w["stuck_state_decisions"]:
                    c["aborted"] = "watchdog_abort_stuck_state"
                elif now - c["t0"] > self.cfg["episode"]["max_wall_s"]:
                    c["aborted"] = "watchdog_abort_wallclock"
                if c["aborted"]:
                    self.watchdog(c["aborted"], dict(episode=c["id"], seed=c["seed"], decision=c["decision"]), abort=True)
            m["uptime_s"] = now - self.t_start                              # supervisor (stream) uptime
            m["milestone"]["furthest_all_time"] = self.store.furthest_all_time()
            m["watchdog"] = dict(healthy=True, restarts=self.restarts, interventions=len(self.interventions),
                                 last=self.last_watchdog)
            self.hub.publish(m)
        elif t == "episode_end":
            c = self.current or dict(id=m["episode"], t0=now)
            reason = c.get("aborted") or m["end_reason"]
            self.store.end_episode(m["episode"], m["decisions"], m["fitness"], m["furthest_milestone"], reason,
                                   bool(c.get("aborted")), m["wall_s"], summary=m)
            self.current = None
            self.hub.publish(dict(m, end_reason=reason))
            self.hub.publish(dict(type="stats", **self.store.stats(), ts=now))
            self.log(f"[episode {m['episode']}] seed {m['seed']} decisions {m['decisions']} fitness {m['fitness']:.0f} "
                     f"furthest {m['furthest_milestone']} end {reason}")

    # ---------------------------------------------------------------- resource sampling
    def sample(self, n_dec_prev):
        now = time.time(); pid = self.worker.pid if self.worker else None
        ticks = _proc_cpu_ticks(pid) if pid else None
        cpu = None
        if ticks is not None and self.cpu_prev[1] is not None:
            cpu = 100.0 * (ticks - self.cpu_prev[1]) / os.sysconf("SC_CLK_TCK") / max(now - self.cpu_prev[0], 1e-9)
        self.cpu_prev = (now, ticks)
        latest = self.hub.latest
        row = dict(ts=now, worker_rss_mb=_proc_rss_mb(pid) if pid else None, supervisor_rss_mb=_proc_rss_mb(os.getpid()),
                   worker_cpu_pct=cpu, gpu=_gpu(), decisions_per_second=latest.get("decisions_per_second"),
                   speed_ratio=latest.get("speed_ratio"), episode=latest.get("episode"), restarts=self.restarts,
                   interventions=len(self.interventions))
        if self.timeseries_path:
            with open(self.timeseries_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    # ---------------------------------------------------------------- main loop
    def run(self):
        self.server.start()
        self.store.event("supervisor_start", dict(hours=self.hours, faults=self.faults))
        self.t_start = t_start = time.time()
        self.spawn()
        next_sample = t_start; faults = list(self.faults)
        try:
            while True:
                now = time.time()
                if self.hours is not None and now - t_start >= self.hours * 3600:
                    break
                try:
                    self.handle(self.out_q.get(timeout=0.5))
                    while True:
                        self.handle(self.out_q.get_nowait())
                except queue.Empty:
                    pass
                except (EOFError, OSError):
                    pass
                now = time.time()
                if faults and now - t_start >= faults[0][0]:
                    at, kind = faults.pop(0)
                    self.store.event("fault_injected", dict(kind=kind, at_s=at, pid=self.worker.pid))
                    self.log(f"[fault] injecting {kind} into worker {self.worker.pid}")
                    os.kill(self.worker.pid, signal.SIGKILL if kind == "kill" else signal.SIGSTOP)
                w = self.cfg["watchdog"]
                if not self.worker.is_alive():
                    self.store.event("worker_crash", dict(exitcode=self.worker.exitcode))
                    self.watchdog("watchdog_restart_crash", dict(exitcode=self.worker.exitcode), restart=True)
                elif self.last_beat is None and now - self.worker_started > w["startup_grace_s"]:
                    self.watchdog("watchdog_restart_startup_timeout", dict(pid=self.worker.pid), restart=True)
                elif self.last_beat is not None and now - self.last_beat > w["heartbeat_timeout_s"]:
                    self.watchdog("watchdog_restart_hang", dict(pid=self.worker.pid, silent_s=now - self.last_beat), restart=True)
                if now >= next_sample:
                    self.sample(0); next_sample = now + self.sample_every_s
        finally:
            self.stop_evt.set()
            if self.worker is not None:
                self.worker.join(20)
                if self.worker.is_alive():
                    self.worker.kill()
            try:
                while True:
                    m = self.out_q.get_nowait()
                    if isinstance(m, dict) and m.get("type") == "episode_end":
                        self.handle(m)
            except (queue.Empty, EOFError, OSError):
                pass
            if self.current is not None:
                self.store.end_episode(self.current["id"], self.current.get("decision", 0), None,
                                       self.current.get("furthest"), "shutdown", False, time.time() - self.current["t0"])
            if getattr(self, "pending", None) is not None:                  # assigned but never started
                self.store.db.execute("DELETE FROM episodes WHERE id=?", (self.pending["id"],))
            self.store.event("supervisor_stop", dict(uptime_s=time.time() - t_start))
            self.server.stop()
        return time.time() - t_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["broadcast", "research"], default="broadcast")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--runtime-dir", default=None)
    ap.add_argument("--soak-out", default=None, help="directory for resource time series (soak test)")
    ap.add_argument("--faults", default="", help="comma list of seconds:kill|stop (soak fault injection)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--write-config", action="store_true")
    a = ap.parse_args()
    if a.write_config:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(default_config(), indent=1) + "\n")
        (OUT / "scientific-provenance.json").write_text(json.dumps(scientific_provenance(), indent=1) + "\n")
        from flymon.telemetry import schema_document
        (OUT / "telemetry-schema.json").write_text(json.dumps(schema_document(), indent=1) + "\n")
        print("wrote", CONFIG_PATH); return
    cfg = load_config()
    faults = [(float(x.split(":")[0]), x.split(":")[1]) for x in a.faults.split(",") if x]
    ts = None
    if a.soak_out:
        Path(a.soak_out).mkdir(parents=True, exist_ok=True); ts = Path(a.soak_out) / "timeseries.jsonl"
    sup = Supervisor(cfg, a.runtime_dir or (ROOT / cfg["runtime_dir"]), hours=a.hours, faults=faults, port=a.port,
                     timeseries_path=ts)
    page = "overlay" if a.mode == "broadcast" else "dashboard"
    print(f"[stream] {a.mode} mode: http://{cfg['telemetry']['host']}:{sup.server.port}/{page}", flush=True)
    up = sup.run()
    print(f"[stream] stopped after {up:.0f}s", json.dumps(sup.store.stats(), default=float), flush=True)


if __name__ == "__main__":
    main()
