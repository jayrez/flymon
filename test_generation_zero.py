"""Experiment 24 tests: frozen E23 sensory model, controller/evaluator separation, episodes."""
import gzip
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from flymon import adapted_reference as ar
from flymon import column_motion as cm
from flymon import frozen_t4 as ft
from flymon import gameplay_eval as ev

ROOT = Path(__file__).resolve().parent
EXP23 = ROOT / "results" / "experiment-23-spatial-inhibitory-veto"
EXP24 = ROOT / "results" / "experiment-24-generation-zero"
HAS_ROM = bool(os.environ.get("POKEMON_ROM")) and Path(os.environ.get("POKEMON_ROM", "")).is_file()
CHEM = {"Mi1": "exc", "Mi4": "inh", "Mi9": "inh", "Tm1": "exc", "Tm2": "exc", "Tm9": "exc"}
CONST = {"sigma": 0.2204, "M": 0.1424, "chemistry": CHEM}


def toy_ctx(ncol=6, seed=0):
    rng = np.random.default_rng(seed)
    proj, audit, bi = {}, {}, 100
    for sub in ft.T4_SUBTYPES:
        mats = {}
        for c in cm.ALL_PARTNERS:
            sign = -1.0 if c in ("Mi4", "Mi9") else 1.0
            m = (rng.random((2, ncol)) < 0.5) * rng.random((2, ncol)) * 0.3 * sign
            mats[c] = sparse.csr_matrix(m)
        proj[sub] = mats
        audit[sub] = {"records": [SimpleNamespace(brain_index=bi), SimpleNamespace(brain_index=bi + 1)]}
        bi += 2
    return dict(proj=proj, audit=audit, records=None, colindex=None, uv=None)


class FrozenSensoryModel(unittest.TestCase):
    def test_frozen_config_matches_e23_artifact(self):
        h = json.loads((EXP23 / "heldout-results.json").read_text())
        cand = dict(h["candidate"]); cand["drop"] = tuple(cand["drop"])
        self.assertEqual(cand, dict(ft.FROZEN_E23_CONFIG))

    def test_frozen_config_matches_e23_code(self):
        import run_spatial_veto_experiment as R
        c = R.candidate().to_json(); c["drop"] = tuple(c["drop"])
        self.assertEqual(c, dict(ft.FROZEN_E23_CONFIG))

    def test_config_cannot_be_mutated(self):
        with self.assertRaises(TypeError):
            ft.FROZEN_E23_CONFIG["G"] = 99.0
        t4 = ft.FrozenT4Readout(toy_ctx(), CONST)
        with self.assertRaises(TypeError):
            t4.config["beta"] = 5.0
        with self.assertRaises(ValueError):
            t4._pos["Mi1"].data[0] = 1.0
        with self.assertRaises(ValueError):
            t4.brain_index[0] = 0

    def test_streaming_equals_batch_e23(self):
        ctx = toy_ctx()
        t4 = ft.FrozenT4Readout(ctx, CONST)
        cfg = ar.AdaptedConfig.from_json(dict(ft.FROZEN_E23_CONFIG))
        rng = np.random.default_rng(3)
        lum = (0.5 + rng.normal(0, 0.2, (30, 6))).clip(0, 1)    # float64, as column_luminance returns
        Ys = np.stack([t4.update_from_luminance(l) for l in lum], 1)
        sig = ar.cell_signals(lum, cfg.tau_fast, cfg.tau_slow, "fixed", t4.L0, cfg.adapt)
        off = 0
        for sub in ft.T4_SUBTYPES:
            Yb, _ = ar.simulate(ctx["proj"][sub], sig, cfg, CONST["sigma"], CONST["M"], CHEM)
            np.testing.assert_allclose(Ys[off:off + 2], Yb, rtol=0, atol=1e-12)
            off += 2

    def test_mi9_excluded_and_hash_changes_with_geometry(self):
        ctx = toy_ctx()
        a = ft.FrozenT4Readout(ctx, CONST)
        self.assertNotIn("Mi9", a.classes)
        self.assertIn("Mi4", a.classes)
        b = ft.FrozenT4Readout(ctx, CONST, proj=toy_ctx(seed=9)["proj"], geometry_label="shuffle")
        self.assertNotEqual(a.sha256(), b.sha256())
        self.assertEqual(a.sha256(), ft.FrozenT4Readout(ctx, CONST).sha256())

    def test_reset_is_adapted_rest(self):
        t4 = ft.FrozenT4Readout(toy_ctx(), CONST)
        y = t4.update_from_luminance(np.full(6, t4.L0))
        self.assertEqual(float(np.abs(y).max()), 0.0)

    def test_injection_quantised_and_capped(self):
        inj = ft.T4Injection(np.arange(5), gain=2.0, cap=0.8, levels=16)
        v = inj.voltages([0.0, 0.01, 0.2, 0.39, 5.0])
        self.assertEqual(v[0], 0.0); self.assertEqual(v[-1], 0.8)
        self.assertTrue(np.allclose(v / 0.05, np.rint(v / 0.05)))
        pairs = inj.pairs([0.0, 0.01, 0.2, 0.39, 5.0])
        self.assertTrue(all(val > 0 for _, val in pairs))
        self.assertNotIn(0, np.concatenate([i for i, _ in pairs]))

    def test_injection_rule_by_condition(self):
        import run_generation_zero as G
        self.assertTrue(G.injection_enabled("gen0"))
        self.assertTrue(G.injection_enabled("c2_shuffled_geometry"))
        self.assertFalse(G.injection_enabled("c1_no_visual"))
        self.assertFalse(G.injection_enabled("c0_random"))


class Evaluator(unittest.TestCase):
    def tel(self, m=38, x=3, y=6, window=False, battle=0):
        return dict(map=m, x=x, y=y, window=window, battle=battle, wy=0 if window else 144)

    def test_guard_rejects_telemetry(self):
        g = ev.ControllerInputGuard(["DNa02_L", "DNa02_R", "event_population_rates_hz"])
        g.check({"DNa02_L": 1.0, "DNa02_R": 2.0})
        for bad in ({"map": 38}, {"DNa02_L": 1.0, "x": 3}, {"telemetry": {}}, {"new_key": 1}):
            with self.assertRaises(RuntimeError):
                g.check(bad)

    def test_milestones(self):
        m = ev.MilestoneTracker(); m.spawn = (38, 3, 6)
        m.update(0, None, self.tel())
        self.assertEqual(m.best_name(), None)
        m.update(1, "LEFT", self.tel(x=2))
        self.assertIn("M1", m.reached)
        m.update(2, "UP", self.tel(m=37))
        m.update(3, "UP", self.tel(m=0, x=5))
        self.assertEqual(m.reached["M2"], 2); self.assertEqual(m.reached["M3"], 3)
        m.update(4, "A", self.tel(m=0, window=True))
        self.assertIn("M6", m.reached)
        m.update(5, "UP", self.tel(m=12, battle=1))
        self.assertEqual(m.best_name(), "M8")

    def test_loop_detector(self):
        d = ev.LoopDetector()
        d.update("LEFT", self.tel(x=2)); f = d.update("LEFT", self.tel(x=2))
        self.assertTrue(f["wall_bump"])
        d = ev.LoopDetector()
        for i, a in enumerate(["LEFT", "RIGHT"] * 4):
            f = d.update(a, self.tel(x=3 + (i % 2)))
        self.assertTrue(f["lr_oscillation"])
        d = ev.LoopDetector()
        for _ in range(25):
            f = d.update(None, self.tel(window=True))
        self.assertTrue(f["menu_loop"])
        d = ev.LoopDetector()
        for a in ["A", None, "A", None, "A"]:
            f = d.update(a, self.tel())
        self.assertTrue(f["a_spam"])
        d = ev.LoopDetector()
        for a in [None] * 21 + ["UP"] + [None] * 5:
            d.update(a, self.tel())
        self.assertEqual(sum(x["inactivity"] for x in d.finalise()), 21)

    def test_metrics_serialise(self):
        rows = [dict(action=a, map=38, x=3 + i, y=6) for i, a in enumerate(["LEFT", None, "RIGHT"])]
        d = ev.LoopDetector(); m = ev.MilestoneTracker(); m.spawn = (38, 3, 6)
        for i, r in enumerate(rows):
            d.update(r["action"], dict(r, window=False, battle=0)); m.update(i, r["action"], dict(r, window=False, battle=0))
        out = ev.episode_metrics(rows, d.finalise(), m, 1.0)
        json.loads(json.dumps(out))
        self.assertEqual(out["unique_tiles"], 3); self.assertEqual(out["total_actions"], 2)


@unittest.skipUnless(HAS_ROM, "POKEMON_ROM not available")
class Emulator(unittest.TestCase):
    def test_reset_returns_to_save_state(self):
        from flymon.emulator import PokemonEmulator
        with PokemonEmulator() as g:
            g.load_state(ROOT / "states/bedroom.state"); g.tick(1); a = ev.read_telemetry(g)
            for b in ("down", "down", "left", "left"):
                g.tap(b, 8, 4)
            self.assertNotEqual((a["x"], a["y"]), (ev.read_telemetry(g)["x"], ev.read_telemetry(g)["y"]))
            g.load_state(ROOT / "states/bedroom.state"); g.tick(1)
            self.assertEqual(ev.read_telemetry(g), a)
            self.assertEqual((a["map"], a["x"], a["y"]), (38, 3, 6))

    def test_instances_isolated(self):
        from flymon.emulator import PokemonEmulator
        with PokemonEmulator() as g1, PokemonEmulator() as g2:
            for g in (g1, g2):
                g.load_state(ROOT / "states/bedroom.state"); g.tick(1)
            for b in ("down", "down", "left", "left"):
                g1.tap(b, 8, 4)
            self.assertEqual((ev.read_telemetry(g2)["x"], ev.read_telemetry(g2)["y"]), (3, 6))
            self.assertNotEqual(ev.read_telemetry(g1), ev.read_telemetry(g2))

    def test_random_episode_budget_log_and_determinism(self):
        import run_generation_zero as G
        law = {"NONE": .5, "LEFT": .2, "RIGHT": .2, "UP": .05, "A": .05}
        with tempfile.TemporaryDirectory() as tmp:
            old = G.RESULTS, G.CAPTURES
            G.RESULTS, G.CAPTURES = Path(tmp) / "r", Path(tmp) / "c"
            try:
                G.RESULTS.mkdir(parents=True)
                orig_root = G.ROOT
                G.ROOT = Path(tmp)
                s1, rows1 = G.run_episode("c0_random", 7, 15, action_law=law)
                s2, rows2 = G.run_episode("c0_random", 7, 15, action_law=law, write_logs=False)
            finally:
                G.RESULTS, G.CAPTURES = old; G.ROOT = orig_root
            self.assertEqual(len(rows1), 15)
            self.assertEqual([r["action"] for r in rows1], [r["action"] for r in rows2])
            self.assertEqual([(r["x"], r["y"]) for r in rows1], [(r["x"], r["y"]) for r in rows2])
            log = Path(tmp) / "r" / "episodes" / "c0_random" / "seed-7.jsonl.gz"
            lines = gzip.open(log, "rt").read().splitlines()
            self.assertEqual(json.loads(lines[0])["meta"]["seed"], 7)
            self.assertEqual([json.loads(x)["action"] for x in lines[1:]], [r["action"] for r in rows1])
            for k in ("run_id", "seed", "save_state", "controller_sha256", "timing", "start_timestamp"):
                self.assertIn(k, s1["meta"])
            json.dumps(s1)


class Repository(unittest.TestCase):
    def test_no_rom_or_new_save_states_committed(self):
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
        self.assertFalse([f for f in files if f.lower().endswith((".gb", ".gbc", ".sav"))])
        self.assertEqual([f for f in files if f.endswith(".state")], ["states/bedroom.state"])
        self.assertFalse([f for f in files if f.startswith("captures/experiment-24/videos")])

    def test_no_oracle_in_gameplay_modules(self):
        for f in ("flymon/frozen_t4.py", "flymon/gameplay_eval.py", "run_generation_zero.py"):
            src = (ROOT / f).read_text()
            self.assertNotIn("motion_nonlinearity", src)
            self.assertNotIn("OPPONENT", src)


@unittest.skipUnless((EXP24 / "aggregate-metrics.json").exists(), "benchmark not analysed")
class SavedArtifacts(unittest.TestCase):
    def test_run_summaries_parse_and_hash(self):
        import hashlib
        for p in sorted((EXP24 / "runs").glob("*/seed-*.json")):
            s = json.loads(p.read_text())
            if s.get("log"):
                self.assertEqual(hashlib.sha256((ROOT / s["log"]).read_bytes()).hexdigest(), s["log_sha256"])
            if s["meta"]["condition"] in ("gen0", "c1_no_visual"):
                self.assertEqual(s["meta"]["sensory"]["model"]["G"], 16.0)


if __name__ == "__main__":
    unittest.main()
