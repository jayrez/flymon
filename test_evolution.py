"""Experiment 25 tests: genome, features, fitness, selection, checkpoints, accelerations."""
import gzip
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flymon import evolution as evo
from flymon import frozen_t4 as ft
from flymon import gameplay_eval as ev

ROOT = Path(__file__).resolve().parent
EXP25 = ROOT / "results" / "experiment-25-evolution"
HAS_ROM = bool(os.environ.get("POKEMON_ROM")) and Path(os.environ.get("POKEMON_ROM", "")).is_file()
GPU = os.environ.get("FLYMON_GPU_TESTS") == "1"


def toy_pooling(n=48, seed=0):
    rng = np.random.default_rng(seed)
    sub = np.array([evo.T4_SUBTYPES[i % 4] for i in range(n)])
    side = np.array(["L" if (i // 4) % 2 == 0 else "R" for i in range(n)])
    rf = np.stack([np.tile([0.25, 0.75], n // 2), np.repeat(np.linspace(0.1, 0.9, 6), n // 6)], 1)
    return evo.T4Pooling(sub, side, rf + rng.normal(0, 0.001, rf.shape))


def tel(m=38, x=3, y=6, window=False, battle=0):
    return dict(map=m, x=x, y=y, window=window, battle=battle, wy=0 if window else 144)


class ActionsAndGenome(unittest.TestCase):
    def test_action_space(self):
        self.assertEqual(set(evo.ACTIONS), {"NONE", "UP", "DOWN", "LEFT", "RIGHT", "A", "B"})
        self.assertIn("DOWN", evo.ACTIONS); self.assertIn("B", evo.ACTIONS)
        for a in ("START", "SELECT"):
            self.assertNotIn(a, evo.ACTIONS); self.assertIn(a, evo.DISABLED_ACTIONS)

    def test_feature_shapes(self):
        self.assertEqual((evo.n_features("dn"), evo.n_features("t4"), evo.n_features("t4dn")), (25, 35, 53))
        pool = toy_pooling()
        dn = np.arange(9, dtype=float)
        for arch, n in (("dn", 25), ("t4", 35), ("t4dn", 53)):
            fb = evo.FeatureBuilder(arch, pool)
            x = fb.build(t4_response=np.ones(48) * 0.01, dn=dn if "dn" in arch else None)
            self.assertEqual(x.shape, (n,))

    def test_feature_names_are_neural_only(self):
        forbidden = ("map", "coord", "ram", "milestone", "fitness", "window", "battle", "tile", "loop", "dialog")
        for arch in evo.ARCHITECTURES:
            for name in evo.feature_names(arch):
                self.assertFalse(any(f in name.lower() for f in forbidden), name)

    def test_genome_contains_no_sensory_parameters(self):
        g = evo.Genome.random("t4dn", np.random.default_rng(0))
        self.assertEqual(set(g.to_json()), {"arch", "W", "b", "log_temp", "sha256"})
        self.assertEqual(g.flat().size, 7 * 53 + 7 + 1)
        before = dict(ft.FROZEN_E23_CONFIG)
        g.mutate(np.random.default_rng(1))
        self.assertEqual(dict(ft.FROZEN_E23_CONFIG), before)
        with self.assertRaises(TypeError):
            ft.FROZEN_E23_CONFIG["tau_slow"] = 4.0

    def test_serialisation_and_hash(self):
        g = evo.Genome.random("dn", np.random.default_rng(3))
        d = json.loads(json.dumps(g.to_json()))
        self.assertEqual(evo.Genome.from_json(d).sha256(), g.sha256())
        d["W"][0][0] += 1.0
        with self.assertRaises(ValueError):
            evo.Genome.from_json(d)

    def test_mutation_deterministic(self):
        g = evo.Genome.random("t4", np.random.default_rng(0))
        a = g.mutate(np.random.default_rng(9)); b = g.mutate(np.random.default_rng(9))
        self.assertEqual(a.sha256(), b.sha256())
        self.assertNotEqual(a.sha256(), g.sha256())

    def test_policy_deterministic_and_valid(self):
        g = evo.Genome.random("t4", np.random.default_rng(0))
        x = np.ones(evo.n_features("t4"))
        acts = [evo.Policy(g, np.random.default_rng(5)).act(x) for _ in range(2)]
        self.assertEqual(acts[0], acts[1])
        p = evo.Policy(g, np.random.default_rng(5)).probabilities(x)
        self.assertAlmostEqual(float(p.sum()), 1.0)


class Features(unittest.TestCase):
    def test_no_vision_zeroes_only_visual_features(self):
        fb = evo.FeatureBuilder("t4dn", toy_pooling())
        x = fb.build(t4_response=None, dn=np.arange(1, 10, dtype=float))
        names = fb.names
        t4_idx = [i for i, n in enumerate(names) if n in evo.t4_feature_names()]
        dn_idx = [i for i, n in enumerate(names) if n in evo.dn_feature_names() and not n.startswith("d_")]
        self.assertTrue(np.all(x[t4_idx] == 0))
        self.assertTrue(np.all(x[dn_idx] != 0))

    def test_deltas_and_prev_action(self):
        fb = evo.FeatureBuilder("t4", toy_pooling())
        fb.build(t4_response=np.full(48, 0.01)); fb.observe_action("DOWN")
        x = fb.build(t4_response=np.full(48, 0.03))
        n = len(evo.t4_feature_names()) // 2
        self.assertTrue(np.all(x[n:2 * n] > 0))
        self.assertEqual(x[-7 + evo.ACTIONS.index("DOWN")], 1.0)

    def test_dn_zscore(self):
        fb = evo.FeatureBuilder("dn")
        rates = {k: 10.0 for k in evo.DN_KEYS}
        v = fb.dn_vector(rates, {k: 5.0 for k in evo.DN_KEYS}, {k: 0.1 for k in evo.DN_KEYS}, 2.0)
        self.assertTrue(np.allclose(v[:8], 5.0))      # SD floor 1 Hz applied
        self.assertEqual(v[8], 2.0)


class Fitness(unittest.TestCase):
    def run_tracker(self, seq):
        t = evo.FitnessTracker((38, 3, 6))
        for d, (a, te) in enumerate(seq):
            t.update(d, a, te)
        return t.result()

    def test_milestone_dominance(self):
        inside = [("LEFT", tel(x=1 + (i % 8), y=1 + (i // 8) % 8)) for i in range(400)]
        inside += [("UP", tel(m=37, x=i % 8, y=i // 8 % 8)) for i in range(200)]
        outside = [("DOWN", tel(m=37, x=3, y=7)), ("DOWN", tel(m=0, x=5, y=6))] + [(None, tel(m=0, x=5, y=6))] * 50
        self.assertGreater(self.run_tracker(outside)["fitness"], self.run_tracker(inside)["fitness"])
        bedroom = [("LEFT", tel(x=i % 8, y=i // 8 % 8)) for i in range(300)]
        exit_ = [("UP", tel(m=37, x=7, y=1))] + [(None, tel(m=37, x=7, y=1))] * 30
        self.assertGreater(self.run_tracker(exit_)["fitness"], self.run_tracker(bedroom)["fitness"])

    def test_exploration_cap(self):
        r = self.run_tracker([("LEFT", tel(x=i, y=i // 3)) for i in range(120)])
        self.assertEqual(r["terms"]["exploration"], evo.TILE_POINTS * evo.TILE_CAP_PER_MAP)

    def test_dialogue_lock_not_rewarded(self):
        lock = [("A", tel(window=True))] + [(None, tel(window=True))] * 40
        r = self.run_tracker(lock)
        self.assertEqual(r["completed_window_cycles"], 0); self.assertEqual(r["dialogue_locks"], 1)
        self.assertEqual(r["terms"]["interaction"], 0)
        good = [("A", tel(window=True)), (None, tel(window=True)), ("B", tel())] * 2
        r2 = self.run_tracker(good)
        self.assertEqual(r2["completed_window_cycles"], 2)
        self.assertGreater(r2["fitness"], r["fitness"])

    def test_loop_penalty(self):
        osc = [(a, tel(x=3 + (i % 2))) for i, a in enumerate(["LEFT", "RIGHT"] * 30)]
        r = self.run_tracker(osc)
        self.assertGreater(r["loop_fraction_nonidle"], 0.5)
        self.assertGreater(r["terms"]["penalty"], 100)

    def test_map_pingpong_not_rewarded(self):
        pp = [("UP", tel(m=38 if i % 2 else 37, x=7, y=1)) for i in range(50)]
        r = self.run_tracker(pp)
        self.assertEqual(r["unique_maps"], 2)
        self.assertLessEqual(r["terms"]["exploration"], 2 * evo.TILE_POINTS + evo.MAP_POINTS)


class Selection(unittest.TestCase):
    def test_rank_deterministic_and_elites_preserved(self):
        rng = np.random.default_rng(0)
        pop = [evo.Genome.random("t4", rng) for _ in range(16)]
        fit = [float(i % 5) for i in range(16)]
        r1, _ = evo.rank(pop, fit); r2, _ = evo.rank(list(reversed(pop)), list(reversed(fit)))
        self.assertEqual([g.sha256() for g in r1], [g.sha256() for g in r2])
        nxt, ne = evo.next_population(r1, np.random.default_rng(1), 16)
        self.assertEqual(ne, 2)
        self.assertEqual([g.sha256() for g in nxt[:ne]], [g.sha256() for g in r1[:ne]])
        self.assertEqual(len(nxt), 16)

    def test_checkpoint_resume_deterministic(self):
        import run_evolution as R
        def fake_evaluate(_p, tasks):
            out = []
            for t in tasks:
                g = evo.Genome.from_json(t["genome"])
                out.append(dict(genome_sha256=g.sha256(), fitness=float(g.b.sum() + 0.01 * t["seed"]),
                                milestones={m: None for m, _ in ev.MILESTONES}, action_counts={a: 1 for a in evo.ACTIONS},
                                decisions=7, completed_window_cycles=0, unique_tiles=1, unique_maps=1,
                                idle_fraction=0.0, loop_fraction_nonidle=0.0, dialogue_locks=0, successful_moves=0,
                                revisit_rate=0.0, task=dict(seed=t["seed"]), wall_s=0.0, terms={}, sensory_sha256=None))
            return out
        old = R.evaluate, R.RESULTS, R.CHECKPOINTS
        try:
            R.evaluate = fake_evaluate
            with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                R.RESULTS = Path(a); R.CHECKPOINTS = Path(a) / "ck"
                R.run_lineage("x", "t4", 8, 4, 7, None)
                straight = json.loads(gzip.open(sorted((Path(a) / "ck" / "x").glob("*.gz"))[-1]).read())
                R.RESULTS = Path(b); R.CHECKPOINTS = Path(b) / "ck"
                R.run_lineage("x", "t4", 8, 2, 7, None)
                R.run_lineage("x", "t4", 8, 4, 7, None)          # resumes from generation 2
                resumed = json.loads(gzip.open(sorted((Path(b) / "ck" / "x").glob("*.gz"))[-1]).read())
            self.assertEqual([g["sha256"] for g in straight["next_population"]],
                             [g["sha256"] for g in resumed["next_population"]])
            self.assertEqual(straight["fitness"], resumed["fitness"])
        finally:
            R.evaluate, R.RESULTS, R.CHECKPOINTS = old


class Accelerations(unittest.TestCase):
    def test_fast_column_sampler_matches(self):
        from flymon import column_motion as cm
        from flymon.fast_io import FastColumnSampler
        from flymon.retina import load_mapping
        records, _ = load_mapping(ROOT / "results/experiment-05-retinotopic/retina-mapping.json")
        colindex = cm.column_index(records)
        uv = np.asarray([(r.u, r.v) for r in records], np.float32)
        fs = FastColumnSampler(records, colindex, uv)
        rng = np.random.default_rng(0)
        for _ in range(3):
            f = rng.integers(0, 256, (144, 160, 4), dtype=np.uint8); f[..., 3] = 255
            ref = cm.column_luminance(f[None], records, colindex, uv)[0]
            np.testing.assert_allclose(fs.luminance(f), ref, atol=1e-6)

    @unittest.skipUnless(GPU, "set FLYMON_GPU_TESTS=1")
    def test_vector_injection_bit_identical(self):
        import run_generation_zero as G
        from flymon.fast_io import VectorInjector, device_pairs
        from flymon.frozen_t4 import T4Injection
        brain, _, _ = G.brain_setup()
        idx = np.arange(1000, 7000)
        inj = T4Injection(idx, 2.0)
        rng = np.random.default_rng(1)
        seqs = [inj.pairs(rng.random(len(idx)) * 0.4) for _ in range(5)]
        brain.reset(seed=3); a = [brain.step(inject=device_pairs(p, brain)).copy() for p in seqs for _ in range(10)]
        vi = VectorInjector(brain); brain.reset(seed=3)
        b = [vi.step(vi.prepare(p)).copy() for p in seqs for _ in range(10)]
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))

    def test_shuffled_geometry_changes_only_geometry(self):
        from scipy import sparse
        from test_generation_zero import CONST, toy_ctx
        import run_fixed_background_experiment as E22
        ctx = toy_ctx(); ctx["colindex"] = {i: i for i in range(6)}
        a = ft.FrozenT4Readout(ctx, CONST)
        perm = E22.permuted_projections(ctx, 1, 25_000_000 + 2601)[0]
        b = ft.FrozenT4Readout(ctx, CONST, proj=perm, geometry_label="shuffle")
        self.assertEqual(a.config, b.config); self.assertEqual(a.sigma, b.sigma)
        self.assertNotEqual(a.sha256(), b.sha256())
        for c in a.classes:
            self.assertAlmostEqual(a._pos[c].data.sum(), b._pos[c].data.sum()); self.assertAlmostEqual(a._neg[c].data.sum(), b._neg[c].data.sum())


class Repository(unittest.TestCase):
    def test_no_rom_or_state_bytes(self):
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
        self.assertFalse([f for f in files if f.lower().endswith((".gb", ".gbc", ".sav", ".mp4"))])
        self.assertEqual([f for f in files if f.endswith(".state")], ["states/bedroom.state"])

    def test_controller_code_never_reads_telemetry(self):
        src = (ROOT / "flymon" / "evolution.py").read_text()
        body = src[src.index("class FeatureBuilder"):src.index("# ------------------------------------------------------------------ genome")]
        self.assertNotIn("read_telemetry", body); self.assertNotIn("memory", body)
        runner = (ROOT / "run_evolution.py").read_text()
        feat = runner[runner.index("x = fb.build"):runner.index("x = fb.build") + 80]
        self.assertNotIn("tel", feat)


@unittest.skipUnless((EXP25 / "sensory-hash.json").exists(), "no E25 artifacts")
class Artifacts(unittest.TestCase):
    def test_sensory_hash_matches_e24(self):
        h = json.loads((EXP25 / "sensory-hash.json").read_text())
        self.assertTrue(h["matches_e24"])
        for p in EXP25.glob("*-generations.jsonl"):
            for line in p.read_text().splitlines():
                for s in json.loads(line)["sensory_sha256"]:
                    self.assertTrue(s == h["native_sha256"])


if __name__ == "__main__":
    unittest.main()
