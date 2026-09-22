"""Experiment 18 tests: column geometry, model primitives, ablations, leakage.

CPU only and fast: geometry logic is tested on tiny synthetic connectomes, and the
real-metadata tests skip if the MaleCNS files are absent. flyvis is never required.
"""
import json
import re
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from flymon import column_motion as cm
from flymon import ethology
from flymon import optic_columns as oc
import run_column_motion_experiment as R

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "redfly-benchmark" / "data"
EXP18 = ROOT / "results" / "experiment-18-column-motion"


def tiny_connectome():
    """T5a at index 0; fast Tm1 at columns x=1,2,3; slow Tm9 at columns x=-1,-2,-3."""
    n = 13
    ct = np.array(["T5a"] + ["Tm1"] * 6 + ["Tm9"] * 6)
    side = np.array(["L"] * n)
    cart = np.full((n, 2), np.nan)
    for k in range(6):
        cart[1 + k] = (1 + k % 3, 0.0)      # fast arm to the +x side
        cart[7 + k] = (-(1 + k % 3), 0.0)   # slow arm to the -x side
    rows = [0] * 12
    cols = list(range(1, 13))
    vals = [1.0] * 6 + [-1.0] * 6           # fast excitatory, slow inhibitory
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    ids = np.arange(n)
    return W, ct, side, cart, ids


class GeometryTests(unittest.TestCase):
    def test_offset_points_from_slow_to_fast(self):
        W, ct, side, cart, ids = tiny_connectome()
        audit = oc.audit_receptive_fields(W, ct, side, cart, ids, subtypes=("T5a",))
        recs = audit["T5a"]["records"]
        self.assertEqual(len(recs), 1)
        v = np.array(recs[0].offset)
        self.assertGreater(v[0], 0)                 # fast centroid is at +x
        self.assertAlmostEqual(v[1], 0.0, places=6)
        self.assertAlmostEqual(recs[0].magnitude, 4.0, places=6)

    def test_uses_presynaptic_row_not_postsynaptic(self):
        """W is W[post, pre]; the audit must read row i = inputs onto neuron i."""
        W, ct, side, cart, ids = tiny_connectome()
        flipped = W.T.tocsr()                       # wrong convention
        a = oc.audit_receptive_fields(W, ct, side, cart, ids, subtypes=("T5a",))
        b = oc.audit_receptive_fields(flipped, ct, side, cart, ids, subtypes=("T5a",))
        self.assertEqual(len(a["T5a"]["records"]), 1)
        self.assertEqual(len(b["T5a"]["records"]), 0)   # no inputs under the flipped graph

    def test_min_partner_inclusion_rule(self):
        W, ct, side, cart, ids = tiny_connectome()
        strict = oc.audit_receptive_fields(W, ct, side, cart, ids, subtypes=("T5a",), min_partners=7)
        self.assertEqual(len(strict["T5a"]["records"]), 0)

    def test_screen_direction_mirrors_right_eye(self):
        v = np.array([1.0, 0.0])
        self.assertGreater(oc.screen_direction(v, "L")[0], 0)
        self.assertLess(oc.screen_direction(v, "R")[0], 0)
        self.assertEqual(oc.cardinal(oc.screen_direction(v, "L")), "right")
        self.assertEqual(oc.cardinal(oc.screen_direction(v, "R")), "left")

    def test_circular_stats(self):
        m, r = oc.circular_stats(np.zeros(10))
        self.assertAlmostEqual(m, 0.0, places=6); self.assertAlmostEqual(r, 1.0, places=6)
        _, r2 = oc.circular_stats(np.array([0.0, np.pi]))
        self.assertAlmostEqual(r2, 0.0, places=6)

    def test_geometry_shuffle_changes_assignment_not_marginals(self):
        W, ct, side, cart, ids = tiny_connectome()
        rng = np.random.default_rng(0)
        perm = oc.shuffle_columns(ct, ("Tm1", "Tm9"), rng)
        shuffled = oc._permuted(cart, ct.astype(str), perm)
        fin = ~np.isnan(cart[:, 0])
        np.testing.assert_allclose(np.sort(shuffled[fin, 0]), np.sort(cart[fin, 0]))

    def test_geometry_uses_no_stimulus_or_label(self):
        """The geometry module must not import or call any stimulus/response code."""
        import ast
        src = (ROOT / "flymon/optic_columns.py").read_text()
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for bad in ("flymon.ethology", "flymon.column_motion", "flymon.retina", "flybrain"):
            self.assertNotIn(bad, imported)
        for bad_call in ("catalogue(", "column_luminance(", "cell_signals(", "responses("):
            self.assertNotIn(bad_call, src)


class StimulusTests(unittest.TestCase):
    def test_parametric_bar_constant_area_and_speed(self):
        seq = cm.parametric_bar("right", "OFF", frames=20, speed=2.0, bar_px=18)
        areas = [(f < 0.5).sum() for f in seq]
        self.assertEqual(len(set(areas)), 1)
        centre = [float(np.mean(np.flatnonzero(f[0] < 0.5))) for f in seq]
        steps = np.diff(centre)
        self.assertTrue(np.allclose(steps, 2.0, atol=0.6))
        self.assertTrue(all(s > 0 for s in steps))          # rightward

    def test_left_is_reverse_of_right(self):
        a = cm.parametric_bar("left", "OFF", frames=20, speed=2.0)
        b = cm.parametric_bar("right", "OFF", frames=20, speed=2.0)
        np.testing.assert_array_equal(a, b[::-1])

    def test_contrast_scales_without_moving_mean(self):
        for c in (0.25, 1.0):
            seq = cm.parametric_bar("right", "ON", frames=10, speed=1.0, contrast=c)
            self.assertLessEqual(seq.max() - seq.min(), c + 1e-6)

    def test_reverse_variant_reverses_frames(self):
        seq = cm.parametric_bar("right", "OFF", frames=12, speed=1.0)
        np.testing.assert_array_equal(cm.temporal_variant(seq, "reverse"), seq[::-1])

    def test_e16_catalogue_unchanged(self):
        """Experiment 18 must not perturb the frozen Experiment-16 stimuli."""
        manifest = ROOT / "results/experiment-16-ethological-visual-motor/stimulus-manifest.json"
        if not manifest.exists():
            self.skipTest("E16 manifest absent")
        from flymon.dataset import digest
        m = json.loads(manifest.read_text())["stimuli"]
        cat = ethology.catalogue()
        for name, meta in m.items():
            self.assertEqual(digest(cat[name].tobytes()), meta["sequence_sha256"], name)


class ModelTests(unittest.TestCase):
    def test_low_pass_smooths_and_identity_at_zero_tau(self):
        x = np.zeros((20, 1)); x[10:] = 1.0
        np.testing.assert_array_equal(cm.low_pass(x, 0), x)
        y = cm.low_pass(x, 5.0)
        self.assertLess(y[10, 0], 1.0)
        self.assertGreater(y[-1, 0], y[10, 0])       # relaxes toward the step

    def test_slow_filter_lags_fast_filter(self):
        x = np.zeros((40, 1)); x[5:] = 1.0
        fast = cm.low_pass(x, 1.0); slow = cm.low_pass(x, 8.0)
        self.assertGreater(fast[10, 0], slow[10, 0])

    def test_rectification_removes_negative_inputs(self):
        lum = np.random.default_rng(0).normal(size=(10, 5)) + 0.5
        cfg_lin = cm.ModelConfig(name="M1", rectify_inputs=False)
        cfg_rec = cm.ModelConfig(name="M2", rectify_inputs=True)
        s_lin = cm.cell_signals(lum, cfg_lin); s_rec = cm.cell_signals(lum, cfg_rec)
        self.assertLess(min(v.min() for v in s_lin.values()), 0.0)
        self.assertGreaterEqual(min(v.min() for v in s_rec.values()), 0.0)

    def test_m0_has_no_temporal_filtering(self):
        lum = np.random.default_rng(1).normal(size=(12, 4)) + 0.5
        cfg = cm.ModelConfig(name="M0", temporal=False)
        sig = cm.cell_signals(lum, cfg)
        base = cm.contrast_signal(lum)
        np.testing.assert_allclose(sig["Mi1"], cm.POLARITY["Mi1"] * base)

    def test_polarity_signs_are_literature_fixed(self):
        self.assertGreater(cm.POLARITY["Mi1"], 0)      # ON pathway
        self.assertLess(cm.POLARITY["Mi9"], 0)         # sign-inverting
        for t in ("Tm1", "Tm2", "Tm9"):
            self.assertLess(cm.POLARITY[t], 0)         # OFF pathway

    def test_arm_membership(self):
        self.assertEqual(cm.arm_of("Mi1"), "fast")
        self.assertEqual(cm.arm_of("Mi9"), "slow")
        self.assertEqual(cm.arm_of("Tm9"), "slow")
        self.assertIsNone(cm.arm_of("DNp01"))

    def test_dsi_formula(self):
        self.assertAlmostEqual(cm.dsi(1.0, 0.0), 1.0, places=6)
        self.assertAlmostEqual(cm.dsi(1.0, 1.0), 0.0, places=6)
        self.assertAlmostEqual(cm.dsi(0.0, 1.0), -1.0, places=6)
        self.assertAlmostEqual(cm.dsi(0.0, 0.0), 0.0, places=6)   # eps guards 0/0


class SplitAndLeakageTests(unittest.TestCase):
    def test_calibration_and_heldout_parameters_disjoint(self):
        self.assertFalse(set(R.CAL_SPEEDS) & set(R.HELDOUT_SPEEDS))
        self.assertFalse(set(R.CAL_SEEDS) & set(R.HELDOUT_SEEDS))

    def test_e18_seeds_unused_by_earlier_experiments(self):
        used = set(range(101, 113)) | set(range(201, 221)) | set(range(301, 341)) \
            | set(range(401, 441)) | set(range(501, 541)) | set(range(861, 931)) \
            | set(range(1101, 1141)) | set(range(1201, 1241))
        self.assertFalse(used & set(R.CAL_SEEDS + R.HELDOUT_SEEDS))

    def test_calibration_stage_never_touches_heldout(self):
        src = (ROOT / "run_column_motion_experiment.py").read_text()
        block = src.split("def stage_calibrate")[1].split("def stage_heldout")[0]
        self.assertNotIn("HELDOUT_SEEDS", block)
        self.assertNotIn('stimulus_bank("heldout")', block)
        self.assertIn('stimulus_bank("calibration")', block)

    def test_correlator_never_selected_as_biological_model(self):
        src = (ROOT / "run_column_motion_experiment.py").read_text()
        block = src.split("# preregistered selection")[1].split("print(")[0]
        self.assertIn('v["config"]["name"] in order', block)
        self.assertNotIn("M_HR", block.split("selected =")[0].split("eligible =")[-1])
        if (EXP18 / "selected-config.json").exists():
            sel = json.loads((EXP18 / "selected-config.json").read_text())
            self.assertNotEqual(sel["config"]["name"], "M_HR")

    def test_no_game_supervision(self):
        forbidden = [r"PokemonEmulator", r"import\s+pyboy", r"from\s+pyboy", r"load_state",
                     r"read_memory", r"game_ram", r"player_x", r"player_y", r"room_id",
                     r"map_id", r"reinforc", r"q_learning", r"behavior_cloning",
                     r"policy_gradient", r"desired_action", r"press_button", r"\.tap\(",
                     r"reward\s*="]
        for f in ("flymon/optic_columns.py", "flymon/column_motion.py",
                  "run_column_motion_experiment.py", "analyze_column_motion_experiment.py"):
            text = (ROOT / f).read_text()
            for pat in forbidden:
                self.assertIsNone(re.search(pat, text, re.IGNORECASE), f"{pat} in {f}")

    def test_no_controller_modules_imported(self):
        for f in ("flymon/optic_columns.py", "flymon/column_motion.py",
                  "run_column_motion_experiment.py"):
            text = (ROOT / f).read_text()
            for mod in ("flymon.controller", "flymon.emulator", "flymon.motor",
                        "flymon.steering", "flymon.locomotion", "flymon.progression"):
                self.assertNotIn(mod, text)

    def test_experiment17_adapter_untouched(self):
        """E18 must not modify the E17 opt-in adapter's reference guarantee."""
        src = (ROOT / "flymon/optic_dynamics.py").read_text()
        self.assertIn("def is_reference", src)
        self.assertIn("if cfg.is_reference():", src)


class RealMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (DATA / "brain.npz").exists() or not R.ANNOT.exists():
            raise unittest.SkipTest("MaleCNS files absent")

    def test_column_coords_resolve_for_medulla_not_for_t4t5(self):
        hexes, cart = oc.load_column_coords(DATA / "brain.npz", R.ANNOT)
        meta = np.load(DATA / "brain.npz")
        ct = meta["cell_type"].astype(str)
        for t in ("Mi1", "Mi9", "Tm1", "Tm2", "Tm9"):
            idx = np.flatnonzero(ct == t)
            self.assertGreater(np.mean(~np.isnan(cart[idx, 0])), 0.9, t)
        for t in ("T4a", "T5a"):
            idx = np.flatnonzero(ct == t)
            self.assertEqual(float(np.mean(~np.isnan(cart[idx, 0]))), 0.0, t)

    def test_geometry_summary_matches_gate(self):
        p = EXP18 / "gate-status.json"
        if not p.exists():
            self.skipTest("analysis not yet run")
        g = json.loads(p.read_text())
        self.assertIn(g["geometry_gate"], ("PASS", "PARTIAL", "FAIL"))
        self.assertIn(g["verdict"], ("PASS", "PARTIAL", "FAIL"))
        self.assertIn(g["category"], list("ABCDEF"))


if __name__ == "__main__":
    unittest.main()
