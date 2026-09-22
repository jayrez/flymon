"""Experiment 16 tests: stimulus construction, resolvers, seed split, no-leakage,
and no forbidden supervision. CPU only; the population resolvers read MaleCNS
metadata (no GPU / no FlyBrain simulation)."""
import json
import re
import unittest
from pathlib import Path

import numpy as np

from flymon import ethology
import run_ethological_experiment as R

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "redfly-benchmark" / "data"
EXP16 = ROOT / "results" / "experiment-16-ethological-visual-motor"


def luminance_seq(name):
    return ethology.mean_luminance(ethology._rgba(ethology.base_sequence(name)))


class StimulusTests(unittest.TestCase):
    def test_catalogue_reproducible(self):
        a = ethology.catalogue(); b = ethology.catalogue()
        self.assertEqual(list(a), list(b))
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_left_right_reverses_direction(self):
        left = ethology.base_sequence("on_left")
        right = ethology.base_sequence("on_right")
        # left sweep is the right sweep reversed in time (mirror path)
        np.testing.assert_array_equal(left, right[::-1])

    def test_on_off_edge_polarity(self):
        on = ethology.base_sequence("on_left")   # bright bar on dark bg
        off = ethology.base_sequence("off_left")  # dark bar on bright bg
        self.assertGreater(on.max(), 0.9); self.assertLess(on.mean(), 0.5)
        self.assertLess(off.min(), 0.1); self.assertGreater(off.mean(), 0.5)
        # each frame has exactly one bar (constant bright area)
        areas = [(f > 0.5).sum() for f in on]
        self.assertEqual(len(set(areas)), 1)

    def test_loom_radius_monotonic_increase(self):
        seq = ethology.base_sequence("loom_dark")   # dark disc on bright bg grows
        dark_area = [(f < 0.5).sum() for f in seq]
        self.assertTrue(all(b >= a for a, b in zip(dark_area, dark_area[1:])))
        self.assertGreater(dark_area[-1], dark_area[0])

    def test_recede_radius_monotonic_decrease(self):
        seq = ethology.base_sequence("recede_dark")
        dark_area = [(f < 0.5).sum() for f in seq]
        self.assertTrue(all(b <= a for a, b in zip(dark_area, dark_area[1:])))
        self.assertLess(dark_area[-1], dark_area[0])

    def test_recede_is_time_reversed_loom(self):
        np.testing.assert_array_equal(ethology.base_sequence("recede_dark"),
                                      ethology.base_sequence("loom_dark")[::-1])

    def test_direction_pairs_luminance_matched(self):
        tol = 1e-6
        self.assertAlmostEqual(luminance_seq("on_left"), luminance_seq("on_right"), delta=tol)
        self.assertAlmostEqual(luminance_seq("off_left"), luminance_seq("off_right"), delta=tol)
        self.assertAlmostEqual(luminance_seq("on_up"), luminance_seq("on_down"), delta=tol)
        self.assertAlmostEqual(luminance_seq("off_up"), luminance_seq("off_down"), delta=tol)

    def test_frozen_repeats_single_frame(self):
        seq = ethology.base_sequence("off_left")
        fz = ethology.temporal_variant(seq, "frozen_first")
        for f in fz:
            np.testing.assert_array_equal(f, seq[0])

    def test_shuffle_preserves_frame_multiset(self):
        seq = ethology.base_sequence("loom_dark")
        sh = ethology.temporal_variant(seq, "shuffle")
        order = ethology.shuffle_order(len(seq))
        self.assertFalse(np.array_equal(order, np.arange(len(seq))))  # real shuffle
        a = sorted(f.tobytes() for f in seq); b = sorted(f.tobytes() for f in sh)
        self.assertEqual(a, b)

    def test_reverse_exactly_reverses(self):
        seq = ethology.base_sequence("flow_expand")
        np.testing.assert_array_equal(ethology.temporal_variant(seq, "reverse"), seq[::-1])


class ResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (DATA / "brain.npz").exists():
            raise unittest.SkipTest("brain.npz not available")
        meta = np.load(DATA / "brain.npz")
        cls.brain = type("B", (), {"cell_type": meta["cell_type"], "side": meta["side"],
                                   "n": len(meta["ids"])})()

    def test_group_resolver_deterministic(self):
        g1 = R.build_groups(self.brain); g2 = R.build_groups(self.brain)
        self.assertEqual(list(g1), list(g2))
        for k in g1:
            np.testing.assert_array_equal(g1[k], g2[k])

    def test_subtypes_and_dns_resolve(self):
        g = R.build_groups(self.brain)
        for sub in ("T4a_L", "T4d_R", "T5a_L", "T5d_R"):
            self.assertIn(sub, g); self.assertGreater(len(g[sub]), 0)
        for dn in ("DNp01_L", "DNp01_R", "DNg13_L", "DNa02_R"):
            self.assertIn(dn, g); self.assertEqual(len(g[dn]), 1)
        self.assertIn("LPLC2_R", g); self.assertIn("VS", g)

    def test_groups_disjoint(self):
        g = R.build_groups(self.brain)
        names, gid = R.group_id_array(self.brain.n, g)  # raises on overlap
        self.assertEqual(len(names), len(g))
        # every group's members map back to its own id
        for k, name in enumerate(names):
            self.assertTrue(np.all(gid[g[name]] == k))


class SplitAndLeakageTests(unittest.TestCase):
    def test_seed_split_disjoint(self):
        self.assertFalse(set(R.CAL_SEEDS) & set(R.HELDOUT_SEEDS))
        self.assertEqual(len(R.HELDOUT_SEEDS), 20)

    def test_preferred_direction_uses_calibration_only(self):
        # If stats exist, the calibration and held-out seed sets must be disjoint and
        # the primary must be evaluated on the held-out seeds.
        path = EXP16 / "statistics.json"
        if not path.exists():
            self.skipTest("statistics not yet generated")
        s = json.loads(path.read_text())
        self.assertFalse(set(s["seeds_calibration"]) & set(s["seeds_heldout"]))
        self.assertEqual(s["primary"]["n"], len(s["seeds_heldout"]))

    def test_paired_deterministic_moments(self):
        a = np.array([1.0, 2.0, 3.0, 4.0]); b = np.array([0.5, 2.5, 2.0, 5.0])
        from analyze_ethological_experiment import paired
        p1 = paired(a, b); p2 = paired(a, b)
        self.assertAlmostEqual(p1["mean"], p2["mean"])         # deterministic
        self.assertAlmostEqual(p1["median"], p2["median"])
        self.assertAlmostEqual(p1["mean"], float((a - b).mean()))


class NoForbiddenSupervisionTests(unittest.TestCase):
    def test_sources_free_of_game_supervision(self):
        # Concrete access constructs, not words in disclaimers: no emulator/RAM reads,
        # no game-memory labels, no reward/RL/policy machinery, no button presses.
        forbidden = [r"PokemonEmulator", r"import\s+pyboy", r"from\s+pyboy",
                     r"load_state", r"read_memory", r"get_memory", r"game_ram",
                     r"player_x", r"player_y", r"room_id", r"map_id",
                     r"reinforc", r"q_learning", r"behavior_cloning", r"policy_gradient",
                     r"desired_action", r"press_button", r"\.tap\(", r"reward\s*="]
        for fname in ("flymon/ethology.py", "run_ethological_experiment.py",
                      "analyze_ethological_experiment.py"):
            text = (ROOT / fname).read_text()
            for pat in forbidden:
                self.assertIsNone(re.search(pat, text, re.IGNORECASE), f"{pat} in {fname}")
        # And the experiment never imports the emulator or controller modules.
        for fname in ("run_ethological_experiment.py", "analyze_ethological_experiment.py"):
            text = (ROOT / fname).read_text()
            self.assertNotIn("flymon.emulator", text)
            self.assertNotIn("flymon.controller", text)


if __name__ == "__main__":
    unittest.main()
