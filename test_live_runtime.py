"""E30 tests: E25 champion recovery, controller input boundary, runtime lifecycle, watchdog, telemetry,
persistence and controls. ROM-dependent tests skip without POKEMON_ROM."""
import gzip
import hashlib
import inspect
import json
import os
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from flymon import evolution as evo
from flymon import live_runtime as lr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "e30-stream-readiness"
HAS_ROM = bool(os.environ.get("POKEMON_ROM")) and Path(os.environ.get("POKEMON_ROM", "")).is_file()
_SENSORY = None


def sensory():
    global _SENSORY
    if _SENSORY is None:
        _SENSORY = lr.SensoryContext()
    return _SENSORY


# ================================================================ recovery
class Recovery(unittest.TestCase):
    def test_champion_hash_params_and_orders(self):
        g = lr.load_champion()
        self.assertEqual(g.sha256(), lr.CHAMPION_SHA256)
        self.assertEqual(g.arch, "t4"); self.assertEqual(g.flat().size, 253)
        self.assertEqual(g.W.shape, (7, 35))
        prov = json.loads((OUT / "e25-champion-provenance.json").read_text())
        self.assertEqual(prov["genome_sha256"], lr.CHAMPION_SHA256); self.assertEqual(prov["parameter_count"], 253)
        self.assertEqual(prov["action_order"], ["NONE", "UP", "DOWN", "LEFT", "RIGHT", "A", "B"])
        self.assertEqual(tuple(prov["action_order"]), evo.ACTIONS)
        self.assertEqual(prov["feature_order"], evo.feature_names("t4"))
        self.assertEqual(prov["feature_order"][:2], ["T4a_L", "T4a_R"]); self.assertEqual(prov["feature_order"][-1], "prev_B")
        self.assertEqual(hashlib.sha256(lr.CHAMPION_PATH.read_bytes()).hexdigest(), prov["champion_genome_file_sha256"])

    def test_champion_matches_committed_e25_artifact(self):
        fin = json.loads((ROOT / "results/experiment-25-evolution/finalists.json").read_text())["finalists"]
        f5 = next(f for f in fin if f["sha256"] == lr.CHAMPION_SHA256)
        self.assertEqual(f5["rank"], 5)
        self.assertEqual(evo.Genome.from_json(f5["genome"]).sha256(), lr.load_champion().sha256())

    def test_serialisation_roundtrip_and_tamper(self):
        d = json.loads(lr.CHAMPION_PATH.read_text())
        self.assertEqual(evo.Genome.from_json(d).sha256(), lr.CHAMPION_SHA256)
        d["W"][3][7] += 1e-9
        with self.assertRaises(ValueError):
            evo.Genome.from_json(d)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(evo.Genome.random("t4", np.random.default_rng(0)).to_json(), f)
        with self.assertRaises(ValueError):
            lr.load_champion(f.name)                                   # wrong genome for the frozen hash
        os.unlink(f.name)

    def test_frozen_t4_hash(self):
        s = sensory()
        self.assertTrue(s.native.sha256().startswith(lr.T4_NATIVE_SHA256_PREFIX))
        self.assertNotEqual(s.readout("shuffled", 2601).sha256(), s.native.sha256())
        self.assertIsNone(s.readout("none", 2601))


# ================================================================ input boundary
class FakeMemory(dict):
    def __getitem__(self, a):
        return dict.get(self, a, 0)


class FakeGame:
    """ROM-free emulator stand-in: frames depend only on ticks and buttons; RAM is arbitrary."""

    def __init__(self, ram_seed):
        rng = np.random.default_rng(ram_seed)
        from flymon import gameplay_eval as ev
        self._pyboy = type("P", (), {})(); self._pyboy.memory = FakeMemory({a: int(rng.integers(0, 256)) for a in ev.ADDR.values()})
        self.rng = rng; self.t = 0; self.held = set()

    def load_state(self, path):
        self.t = 0

    def tick(self, n=1):
        self.t += n
        for a in list(self._pyboy.memory):                            # RAM keeps changing, differently per instance
            self._pyboy.memory[a] = int(self.rng.integers(0, 256))
        return True

    def framebuffer(self):
        f = np.zeros((144, 160, 4), np.uint8); f[..., 3] = 255
        k = (self.t // 3 + 17 * len(self.held)) % 160
        f[:, k:k + 20, :3] = 200; f[(self.t * 5) % 144, :, :3] = 90
        return f

    def press(self, b):
        self.held.add(b)

    def release(self, b):
        self.held.discard(b)


class InputBoundary(unittest.TestCase):
    def test_controller_rejects_non_frame_inputs(self):
        ctl = lr.FrozenController(lr.load_champion(), sensory(), "normal"); ctl.reset(1)
        for bad in ({"map": 38}, [{"x": 1}], [np.zeros((144, 160, 3), np.uint8)], [np.zeros(10)]):
            with self.assertRaises(TypeError):
                ctl.observe_frames(bad)
        self.assertEqual(list(inspect.signature(ctl.decide).parameters), [])
        self.assertEqual(list(inspect.signature(lr.RandomController("random_uniform").decide).parameters), [])

    def test_ram_never_reaches_controller(self):
        """Same frames + different RAM streams -> identical features and actions."""
        runs = []
        for ram_seed in (1, 2):
            ctl = lr.FrozenController(lr.load_champion(), sensory(), "normal")
            from flymon.stream_eval import StreamEvaluator
            ep = lr.Episode(FakeGame(ram_seed), ctl, 2601, 25, state_path="unused", evaluator_factory=StreamEvaluator).start()
            decs = []
            while not ep.done:
                d, tel, snap = ep.step(); decs.append((d.action, tuple(d.features.values())))
            runs.append(decs)
        self.assertEqual(runs[0], runs[1])

    def test_controller_state_holds_no_game_or_evaluator(self):
        ctl = lr.FrozenController(lr.load_champion(), sensory(), "normal"); ctl.reset(3)
        ctl.observe_frames([FakeGame(0).framebuffer()]); ctl.decide()
        allowed = (evo.Genome, lr.SensoryContext, evo.FeatureBuilder, evo.Policy, np.ndarray, list, str, type(None))
        for k, v in vars(ctl).items():
            if k == "readout":
                continue
            self.assertIsInstance(v, allowed, k)
        names = " ".join(evo.feature_names("t4")).lower()
        for f in ("map", "coord", "ram", "milestone", "window", "battle", "watchdog", "tile", "hp", "party", "menu"):
            self.assertNotIn(f, names)

    def test_controller_sources_never_touch_ram_or_evaluator(self):
        for cls in (lr.FrozenController, lr.RandomController):
            src = inspect.getsource(cls)
            for bad in ("read_telemetry", "memory", "gameplay_eval", "stream_eval", "milestone", "watchdog", "_pyboy"):
                self.assertNotIn(bad, src, (cls.__name__, bad))
        ep = inspect.getsource(lr.Episode.step)
        self.assertLess(ep.index("self.controller.decide()"), ep.index("read_telemetry"))
        self.assertLess(ep.index("self.controller.observe_frames(frames)"), ep.index("read_telemetry"))
        import run_stream
        w = inspect.getsource(run_stream.worker_main)
        self.assertNotIn("ctl.observe_frames(", w)                     # frames only flow through Episode
        self.assertIn("eid, seed = assign_q.get", w)                    # the worker receives only (id, seed)


# ================================================================ runtime / controls
class Controls(unittest.TestCase):
    def test_no_vision_zeroes_t4_features(self):
        ctl = lr.FrozenController(lr.load_champion(), sensory(), "none"); ctl.reset(5)
        ctl.observe_frames([FakeGame(0).framebuffer()] * 3)
        d = ctl.decide()
        self.assertTrue(all(v == 0 for k, v in d.features.items() if not k.startswith("prev_")))

    def test_random_controllers(self):
        for kind in ("random_uniform", "random_matched"):
            a = lr.RandomController(kind); a.reset(7); b = lr.RandomController(kind); b.reset(7)
            sa = [a.decide().action for _ in range(50)]; sb = [b.decide().action for _ in range(50)]
            self.assertEqual(sa, sb)
            self.assertTrue(set(sa) <= set(evo.ACTIONS[1:]) | {None})

    def test_frozen_during_production(self):
        ctl = lr.FrozenController(lr.load_champion(), sensory(), "normal"); ctl.reset(1)
        ctl.observe_frames([FakeGame(0).framebuffer()]); ctl.decide()
        ctl.genome.W[0, 0] += 1.0
        with self.assertRaises(RuntimeError):
            ctl.decide()

    def test_episode_lifecycle_fake(self):
        ctl = lr.RandomController("random_uniform")
        from flymon.stream_eval import StreamEvaluator
        ep = lr.Episode(FakeGame(0), ctl, 11, 7, state_path="unused", evaluator_factory=StreamEvaluator).start()
        n = 0
        while not ep.done:
            ep.step(); n += 1
        self.assertEqual(n, 7); self.assertEqual(len(ep.evaluator.rows), 7)
        r = ep.evaluator.result(); self.assertEqual(r["decisions"], 7); self.assertIn("dialogue", r)


@unittest.skipUnless(HAS_ROM, "set POKEMON_ROM")
class RomRuntime(unittest.TestCase):
    def test_same_seed_replay_matches_e25_log_and_reset(self):
        from flymon.emulator import PokemonEmulator
        from flymon.stream_eval import StreamEvaluator
        hist = json.loads(gzip.open(ROOT / "results/experiment-25-evolution/champion-heldout-logs.json.gz").read())["2605"]["log"]
        with PokemonEmulator() as game:
            outs = []
            for _ in range(2):                                          # second run tests emulator reset
                ctl = lr.FrozenController(lr.load_champion(), sensory(), "normal")
                ep = lr.Episode(game, ctl, 2605, 150, evaluator_factory=StreamEvaluator).start()
                while not ep.done:
                    ep.step()
                outs.append([list(r[:7]) for r in ep.evaluator.rows])
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], [[h[0], h[1] or "NONE"] + h[2:] for h in hist[:150]])


# ================================================================ dialogue classifier
class Dialogue(unittest.TestCase):
    def rows(self, spec):
        out, d = [], 0
        for n, m, x, y, win, hashes in spec:
            for i in range(n):
                out.append((d, "A", m, x, y, int(win), 0, 100 if win else 144, hashes(i) if win else "")); d += 1
        return out

    def test_classes(self):
        from flymon.stream_eval import classify_dialogue
        adv = lambda i: str(i // 3)                                      # text advancing
        froz = lambda i: "same"
        r = classify_dialogue(self.rows([(5, 0, 10, 1, False, adv), (40, 0, 10, 1, True, adv), (5, 40, 5, 3, False, adv),
                                         (40, 40, 5, 3, True, adv), (3, 37, 3, 2, False, adv), (40, 37, 3, 2, True, adv),
                                         (2, 37, 4, 4, False, adv), (40, 37, 4, 4, True, froz)]))
        self.assertEqual([e["cls"] for e in r["events"]], ["scripted", "scripted", "navigable", "stall"])
        self.assertEqual(r["events"][3]["reason"], "frozen_window")

    def test_repeat_loop_and_overworld(self):
        from flymon.stream_eval import classify_dialogue
        adv = lambda i: str(i // 3)
        spec = []
        for _ in range(3):
            spec += [(2, 40, 6, 4, False, adv), (35, 40, 6, 4, True, adv)]
        spec += [(160, 37, 2, 2, False, adv)]
        r = classify_dialogue(self.rows([(40, 40, 5, 3, True, adv)] + spec))
        self.assertEqual([e["cls"] for e in r["events"]], ["scripted", "navigable", "navigable", "stall"])
        self.assertEqual(r["events"][-1]["reason"], "repeat_loop")
        self.assertEqual(len(r["overworld_stall_runs"]), 1); self.assertTrue(r["any_genuine_stall"])


# ================================================================ telemetry / persistence / watchdog
def good_message(**kw):
    m = dict(type="decision", schema=1, ts=time.time(), episode=1, seed=40001, decision=3, episode_runtime_s=0.6, uptime_s=1.0,
             decisions_per_second=5.0, speed_ratio=1.0, action="UP", recent_actions=["UP"],
             controller=dict(top_action="UP", probabilities={a: 1 / 7 for a in evo.ACTIONS}, features={}, sha256="x", frozen=True,
                             kind="k"), t4={"T4a": 0.1}, visual_pathway=dict(status="ok"),
             game=dict(map=38, map_name="RedsHouse2F", x=3, y=6, window=False, battle=False, window_run=0),
             milestone=dict(current="bedroom", furthest_episode="bedroom"), unique_tiles=1, watchdog={}, frame_sha1="a")
    m.update(kw); return m


class Telemetry(unittest.TestCase):
    def test_validate(self):
        from flymon.telemetry import validate, schema_document
        validate(good_message())
        bad = good_message(); del bad["action"]
        with self.assertRaises(ValueError):
            validate(bad)
        with self.assertRaises(ValueError):
            validate(good_message(decision="3"))
        doc = schema_document(); self.assertIn("decision_message", doc); json.dumps(doc)

    def test_http_read_only_and_sse(self):
        from flymon.telemetry import TelemetryHub, TelemetryServer
        hub = TelemetryHub({"title": "t"}); srv = TelemetryServer(hub, port=0).start()
        base = f"http://127.0.0.1:{srv.port}"
        try:
            hub.publish(good_message(decision=9))
            st = json.loads(urllib.request.urlopen(base + "/state").read())
            self.assertEqual(st["latest"]["decision"], 9)
            self.assertEqual(json.loads(urllib.request.urlopen(base + "/provenance").read())["title"], "t")
            self.assertIn(b"EventSource", urllib.request.urlopen(base + "/static/app.js").read())
            for page in ("/overlay", "/dashboard"):
                self.assertIn(b"app.js", urllib.request.urlopen(base + page).read())
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(urllib.request.Request(base + "/state", data=b"{}", method="POST"))
            self.assertEqual(e.exception.code, 405)
            r = urllib.request.urlopen(base + "/events", timeout=5)
            line = r.readline()
            while not line.startswith(b"data:"):
                line = r.readline()
            self.assertEqual(json.loads(line[5:])["decision"], 9)
            r.close()
        finally:
            srv.stop()


class Persistence(unittest.TestCase):
    def test_store_roundtrip(self):
        from flymon.stream_state import StreamStore
        with tempfile.TemporaryDirectory() as d:
            s = StreamStore(Path(d) / "s.sqlite")
            self.assertEqual(s.next_seed(40001), 40001); self.assertEqual(s.next_seed(40001), 40002)
            e = s.start_episode(40001, "abc")
            self.assertTrue(s.milestone("house_exited", e, 40001)); self.assertFalse(s.milestone("house_exited", e, 40001))
            s.milestone("bedroom_exited", e, 40001)
            s.end_episode(e, 100, 1234.5, "house_exited", "budget", False, 20.0)
            s.event("watchdog_restart_crash", {"x": 1}); s.close()
            s2 = StreamStore(Path(d) / "s.sqlite")
            st = s2.stats()
            self.assertEqual(st["episodes_all_time"], 1); self.assertEqual(st["furthest_milestone_all_time"], "house_exited")
            self.assertEqual(st["best_fitness"], 1234.5); self.assertEqual(st["watchdog_interventions"], 1)
            self.assertEqual(s2.next_seed(40001), 40003)
            with self.assertRaises(ValueError):
                s2.milestone("not_a_milestone", e, 1)
            s2.close()


def _fake_worker(cfg, out_q, assign_q, abort_evt, stop_evt):
    """Scripted worker for watchdog tests. Behaviour chosen by cfg['fake']."""
    import queue as q_
    flag = Path(cfg["flag"])
    first = not flag.exists(); flag.write_text("x")
    while not stop_evt.is_set():
        try:
            eid, seed = assign_q.get(timeout=0.5)
        except q_.Empty:
            continue
        assert isinstance(eid, int) and isinstance(seed, int)
        abort_evt.clear()
        out_q.put(dict(type="episode_start", episode=eid, seed=seed, ts=time.time()))
        reason, n = "budget", 0
        for n in range(1, cfg["decisions"] + 1):
            if abort_evt.is_set():
                reason = "watchdog_abort"; break
            if first and cfg["fake"] == "crash" and n == 3:
                os._exit(3)
            if first and cfg["fake"] == "hang" and n == 3:
                time.sleep(3600)
            fh = "frozen" if cfg["fake"] == "frozen" else str(n)
            out_q.put(good_message(episode=eid, seed=seed, decision=n, frame_sha1=fh, game=dict(map=38, map_name="R", x=n, y=1,
                                                                                               window=False, battle=False, window_run=0)))
            time.sleep(0.02)
        out_q.put(dict(type="episode_end", episode=eid, seed=seed, decisions=n, fitness=1.0, furthest_milestone="bedroom",
                       end_reason=reason, wall_s=0.1, ts=time.time()))


class Watchdog(unittest.TestCase):
    def run_sup(self, fake, decisions=20, seconds=6.0):
        import run_stream
        with tempfile.TemporaryDirectory() as d:
            cfg = run_stream.default_config()
            cfg["watchdog"].update(heartbeat_timeout_s=1.5, startup_grace_s=5, frozen_frame_decisions=5, restart_backoff_s=0.1)
            cfg.update(fake=fake, decisions=decisions, flag=str(Path(d) / "flag"))
            sup = run_stream.Supervisor(cfg, d, hours=seconds / 3600, port=0, mp_context="fork", worker_target=_fake_worker,
                                        log=lambda *a: None, sample_every_s=1000)
            sup.run()
            kinds = [r[0] for r in sup.store.db.execute("SELECT kind FROM events")]
            eps = list(sup.store.db.execute("SELECT end_reason, watchdog FROM episodes WHERE ended IS NOT NULL"))
            return sup, kinds, eps

    def test_crash_recovery(self):
        sup, kinds, eps = self.run_sup("crash")
        self.assertIn("worker_crash", kinds); self.assertIn("watchdog_restart_crash", kinds); self.assertIn("worker_restart", kinds)
        self.assertIn(("watchdog_restart_crash", 1), eps)
        self.assertIn(("budget", 0), eps)                              # episodes complete after recovery

    def test_hang_recovery(self):
        sup, kinds, eps = self.run_sup("hang", seconds=7.0)
        self.assertIn("watchdog_restart_hang", kinds); self.assertIn(("budget", 0), eps)

    def test_frozen_frame_abort_marked(self):
        sup, kinds, eps = self.run_sup("frozen", decisions=40, seconds=4.0)
        self.assertIn("watchdog_abort_frozen_frame", kinds)
        self.assertIn(("watchdog_abort_frozen_frame", 1), eps)

    def test_watchdog_has_no_gameplay_actions(self):
        import run_stream
        src = inspect.getsource(run_stream.Supervisor)
        for bad in (".press(", ".release(", "memory", "load_state", "tick("):
            self.assertNotIn(bad, src)


class Artifacts(unittest.TestCase):
    def test_runtime_config_consistent(self):
        cfg = json.loads((OUT / "runtime-config.json").read_text())
        self.assertEqual(cfg["controller"]["genome_sha256"], lr.CHAMPION_SHA256)
        self.assertTrue(cfg["controller"]["frozen"])
        self.assertEqual(cfg["start_state"]["sha256"], hashlib.sha256((ROOT / "states/bedroom.state").read_bytes()).hexdigest())
        base = cfg["seeds"]["base"]
        self.assertFalse(set(range(base, base + 100000)) & (set(range(2501, 2509)) | set(range(2601, 2621))))

    def test_reproduction_exact(self):
        p = OUT / "reproduction-results.json"
        if not p.exists():
            self.skipTest("reproduction not run")
        r = json.loads(p.read_text())
        self.assertTrue(r["gate_S1"]["pass"]); self.assertTrue(r["gate_S1"]["action_logs_identical_all_seeds"])
        for c in r["conditions"].values():
            self.assertEqual(c["identical_fitness_seeds"], 20)

    def test_provenance_claims(self):
        p = json.loads((OUT / "scientific-provenance.json").read_text())
        self.assertIn("None", p["within_run_learning"]); self.assertIn("never", p["ram"].lower())
        self.assertIn("NOT run", p["brain_simulation"])
        self.assertIn("The fly learned Pokémon.", p["disallowed_claims"])


if __name__ == "__main__":
    unittest.main()
