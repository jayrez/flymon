"""Experiment 17 tests: opt-in dynamics adapter, stimulus reuse, seed/selection
hygiene, and absence of game supervision. CPU only and fast by default; the
GPU equivalence test runs when FLYMON_GPU_TESTS=1."""
import json
import os
import re
import unittest
from pathlib import Path

import numpy as np

from flymon import ethology
from flymon.optic_dynamics import (OpticDynamicsConfig, build_repaired, candidates,
                                   expand_types)
import run_optic_dynamics_experiment as R

ROOT = Path(__file__).resolve().parent
EXP17 = ROOT / "results" / "experiment-17-optic-lobe-dynamics"


class FakeFlyBrain:
    """Minimal stand-in exposing only what the adapter's setup touches."""
    dt = 0.020; tau = 0.100; gain = 3.0; tonic = 0.14
    noise_hz = 1.2; noise_amp = 0.22; eye_gain = 0.62

    def __init__(self, cell_type=None, batch=1, seed=0):
        self.xp = np
        self.device = "cpu"
        self.batch = batch
        self.cell_type = np.asarray(cell_type)
        self.n = len(self.cell_type)
        self.decay = np.float32(np.exp(-self.dt / self.tau))
        self.refractory_steps = 0
        self.reset(seed)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.v = np.zeros((self.n, self.batch), np.float32)
        self.fired = np.empty(0, np.int64)
        self.steps = 0


TYPES = np.array(["R1-6", "L1", "L2", "L3", "Mi1", "Mi9", "Mi4", "CT1", "T4a", "T5a"])


def fake(cfg):
    return build_repaired(FakeFlyBrain, cfg, cell_type=TYPES)


class ConfigTests(unittest.TestCase):
    def test_reference_is_reference(self):
        self.assertTrue(OpticDynamicsConfig().is_reference())

    def test_any_change_is_not_reference(self):
        self.assertFalse(OpticDynamicsConfig(tonic_add={"L1": 0.1}).is_reference())
        self.assertFalse(OpticDynamicsConfig(tau_scale={"Mi9": 2.0}).is_reference())
        self.assertFalse(OpticDynamicsConfig(delay_steps={"Mi9": 1}).is_reference())
        self.assertFalse(OpticDynamicsConfig(eye_gain_scale=2.0).is_reference())

    def test_candidate_grid_is_small_and_named(self):
        c = candidates(0.09)
        self.assertLessEqual(len(c), 6)
        self.assertEqual(c[0].name, "A_reference")
        self.assertTrue(c[0].is_reference())
        self.assertEqual(len({x.name for x in c}), len(c))

    def test_expand_types_exact_and_prefix(self):
        e = expand_types(TYPES, {"L1": 1.0})
        self.assertEqual(list(e[1.0]), [1])
        e2 = expand_types(TYPES, {"T4*": 2.0})
        self.assertEqual(list(e2[2.0]), [8])


class AdapterTargetingTests(unittest.TestCase):
    def test_tonic_targets_only_named_types(self):
        b = fake(OpticDynamicsConfig(name="t", tonic_add={"L1": 0.09, "L2": 0.09}))
        tonic = np.asarray(b._tonic_vec).ravel()
        base = float(b.tonic)
        self.assertAlmostEqual(tonic[1], base + 0.09, places=6)   # L1
        self.assertAlmostEqual(tonic[2], base + 0.09, places=6)   # L2
        for i in (0, 3, 4, 5, 6, 7, 8, 9):
            self.assertAlmostEqual(tonic[i], base, places=6)

    def test_tau_scale_targets_only_named_types(self):
        b = fake(OpticDynamicsConfig(name="s", tau_scale={"Mi9": 3.0}))
        decay = np.asarray(b._decay_vec).ravel()
        ref = float(np.exp(-FakeFlyBrain.dt / FakeFlyBrain.tau))
        self.assertGreater(decay[5], ref)             # Mi9 slower -> decays less
        for i in (0, 1, 2, 3, 4, 6, 7, 8, 9):
            self.assertAlmostEqual(decay[i], ref, places=6)

    def test_delay_targets_only_named_types(self):
        b = fake(OpticDynamicsConfig(name="d", delay_steps={"Mi9": 2, "CT1": 2}))
        self.assertEqual(b._max_delay, 2)
        mask = np.asarray(b._delay_mask[2])
        self.assertTrue(mask[5] and mask[7])          # Mi9, CT1
        self.assertFalse(mask[[0, 1, 2, 3, 4, 6, 8, 9]].any())

    def test_reference_config_builds_uniform_vectors(self):
        b = fake(OpticDynamicsConfig())
        self.assertTrue(b._uniform_decay)
        self.assertTrue(b._uniform_tonic)
        self.assertIsNone(b._delay_mask)

    def test_reset_clears_temporal_state(self):
        b = fake(OpticDynamicsConfig(name="d", delay_steps={"Mi9": 2}))
        b._spike_history = [np.zeros((b.n, 1), np.float32)] * 3
        b.reset(seed=1)
        self.assertEqual(len(b._spike_history), 0)

    def test_setup_is_deterministic(self):
        cfg = OpticDynamicsConfig(name="x", tonic_add={"L1": 0.09}, tau_scale={"Mi9": 3.0},
                                  delay_steps={"CT1": 1})
        a, b = fake(cfg), fake(cfg)
        np.testing.assert_array_equal(np.asarray(a._tonic_vec), np.asarray(b._tonic_vec))
        np.testing.assert_array_equal(np.asarray(a._decay_vec), np.asarray(b._decay_vec))
        self.assertEqual(a._max_delay, b._max_delay)


class StimulusReuseTests(unittest.TestCase):
    def test_experiment16_catalogue_reused_unchanged(self):
        cat = ethology.catalogue()
        for name in R.CAL_STIMULI + R.GAIN_STIMULI + R.TRACE_STIMULI:
            self.assertIn(name, cat)

    def test_frozen_stimulus_has_no_frame_to_frame_motion(self):
        seq = ethology.temporal_variant(ethology.base_sequence("off_left"), "frozen_first")
        diffs = [float(np.abs(seq[i + 1] - seq[i]).sum()) for i in range(len(seq) - 1)]
        self.assertEqual(max(diffs), 0.0)

    def test_shuffle_preserves_frame_multiset(self):
        seq = ethology.base_sequence("off_left")
        sh = ethology.temporal_variant(seq, "shuffle")
        self.assertEqual(sorted(f.tobytes() for f in seq), sorted(f.tobytes() for f in sh))

    def test_timing_constants_match_experiment16(self):
        self.assertEqual(R.STEPS_PER_FRAME, 20)
        self.assertEqual(ethology.FRAMES, 10)


class SeedAndSelectionTests(unittest.TestCase):
    def test_calibration_and_heldout_seeds_disjoint(self):
        self.assertFalse(set(R.CAL_SEEDS) & set(R.HELDOUT_SEEDS))
        self.assertEqual(len(R.CAL_SEEDS), 20)
        self.assertEqual(len(R.HELDOUT_SEEDS), 20)

    def test_e17_seeds_unused_by_earlier_experiments(self):
        used = set(range(101, 113)) | set(range(201, 221)) | set(range(301, 341)) \
            | set(range(401, 441)) | set(range(501, 541)) | set(range(861, 931)) \
            | set(range(1101, 1141))
        self.assertFalse(used & set(R.CAL_SEEDS + R.HELDOUT_SEEDS))

    def test_gain_grid_is_preregistered(self):
        self.assertEqual(tuple(R.GAIN_GRID), (0.5, 1.0, 2.0, 4.0, 8.0))

    def test_selected_config_recorded_from_calibration_only(self):
        p = EXP17 / "selected-config.json"
        if not p.exists():
            self.skipTest("calibration not yet run")
        sel = json.loads(p.read_text())
        self.assertIn("calibration", sel["selection_rule"])
        self.assertIn("preferred_directions", sel)
        # preferred directions must name actual stimulus conditions, not held-out data
        for sub, pref in sel["preferred_directions"].items():
            self.assertIn(pref, ethology.catalogue())

    def test_heldout_cannot_alter_selected_config(self):
        """selected-config.json must predate / not depend on held-out responses."""
        sp = EXP17 / "selected-config.json"; hp = EXP17 / "heldout-responses.npz"
        if not (sp.exists() and hp.exists()):
            self.skipTest("stages not both run")
        self.assertLess(sp.stat().st_mtime, hp.stat().st_mtime)
        src = (ROOT / "run_optic_dynamics_experiment.py").read_text()
        calib = src.split('if args.stage == "calibrate"')[1].split('if args.stage == "traces"')[0]
        self.assertNotIn("HELDOUT_SEEDS", calib)
        self.assertIn("CAL_SEEDS", calib)


class TemporalTraceTests(unittest.TestCase):
    def test_traces_aligned_to_full_stimulus(self):
        p = EXP17 / "temporal-traces.npz"
        if not p.exists():
            self.skipTest("traces not yet run")
        z = np.load(p, allow_pickle=False)
        keys = [k for k in z.files if "|" in k]
        self.assertTrue(keys)
        expected = ethology.FRAMES * R.STEPS_PER_FRAME
        for k in keys:
            self.assertEqual(z[k].shape[0], expected)
            self.assertEqual(z[k].shape[1], len(z["group_names"]))


class NoGameSupervisionTests(unittest.TestCase):
    def test_no_ram_actions_reward_or_rl(self):
        forbidden = [r"PokemonEmulator", r"import\s+pyboy", r"from\s+pyboy", r"load_state",
                     r"read_memory", r"get_memory", r"game_ram", r"player_x", r"player_y",
                     r"room_id", r"map_id", r"reinforc", r"q_learning", r"behavior_cloning",
                     r"policy_gradient", r"desired_action", r"press_button", r"\.tap\(",
                     r"reward\s*="]
        for f in ("flymon/optic_dynamics.py", "run_optic_dynamics_experiment.py",
                  "analyze_optic_dynamics_experiment.py", "run_positive_control.py"):
            text = (ROOT / f).read_text()
            for pat in forbidden:
                self.assertIsNone(re.search(pat, text, re.IGNORECASE), f"{pat} in {f}")

    def test_no_controller_modules_touched(self):
        for f in ("run_optic_dynamics_experiment.py", "analyze_optic_dynamics_experiment.py",
                  "flymon/optic_dynamics.py"):
            text = (ROOT / f).read_text()
            for mod in ("flymon.controller", "flymon.emulator", "flymon.motor",
                        "flymon.steering", "flymon.locomotion", "flymon.progression"):
                self.assertNotIn(mod, text)


@unittest.skipUnless(os.environ.get("FLYMON_GPU_TESTS") == "1", "GPU equivalence test opt-in")
class GpuReferenceEquivalenceTests(unittest.TestCase):
    def test_reference_config_matches_stock_flybrain(self):
        from flybrain import FlyBrain
        from flymon.retina import load_mapping
        recs, _ = load_mapping(ROOT / "results/experiment-05-retinotopic/retina-mapping.json")
        cat = ethology.catalogue()
        stock = FlyBrain(data=R.DATA, device="cuda", batch=1)
        rep = build_repaired(FlyBrain, OpticDynamicsConfig(), data=R.DATA, device="cuda", batch=1)
        drives = R.encode(cat["off_left"], recs, len(stock.visual))

        def run(b):
            b.reset(seed=1201)
            return [np.asarray(b.step(eye_drive=drives[i // 20])).copy() for i in range(40)]
        for x, y in zip(run(stock), run(rep)):
            np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    unittest.main()
