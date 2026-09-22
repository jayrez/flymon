"""Experiment 19 tests: mechanism equations, ablations, windows, splits, leakage.

CPU only and fast; heavy MaleCNS tests skip if the data files are absent.
"""
import json
import re
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from flymon import column_motion as cm
from flymon import motion_nonlinearity as mn
from flymon import optic_columns as oc
import run_motion_nonlinearity_experiment as R

ROOT = Path(__file__).resolve().parent
EXP18 = ROOT / "results" / "experiment-18-column-motion"
EXP19 = ROOT / "results" / "experiment-19-motion-nonlinearity"


def drives(n=4, T=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)), rng.normal(size=(n, T))


class MechanismEquationTests(unittest.TestCase):
    def test_n0_n1_reproduce_e18_linear_sum(self):
        F, S = drives()
        for fam in ("N0_E18", "N1_LEXT"):
            y = mn.apply_mechanism(F, S, mn.Mechanism(family=fam))
            np.testing.assert_allclose(y, np.maximum(F + S, 0.0))

    def test_coincidence_equation(self):
        F, S = drives()
        Fr, Sr = np.maximum(F, 0), np.maximum(S, 0)
        y = mn.apply_mechanism(F, S, mn.Mechanism(family="N2_COINCIDENCE", gate="identity"))
        np.testing.assert_allclose(y, Fr * Sr)
        y2 = mn.apply_mechanism(F, S, mn.Mechanism(family="N2_COINCIDENCE", gate="saturating"))
        np.testing.assert_allclose(y2, Fr * (Sr / (1.0 + Sr)))

    def test_shunting_equation_and_alpha_zero_removes_division(self):
        F, S = drives()
        Fr, Sr = np.maximum(F, 0), np.maximum(S, 0)
        y = mn.apply_mechanism(F, S, mn.Mechanism(family="N3_SHUNT", alpha=2.0))
        np.testing.assert_allclose(y, Fr / (1.0 + 2.0 * Sr))
        y0 = mn.apply_mechanism(F, S, mn.Mechanism(family="N3_SHUNT", alpha=0.0))
        np.testing.assert_allclose(y0, Fr)          # interaction removed

    def test_subunit_equation_and_gamma_zero(self):
        F, S = drives()
        Fr, Sr = np.maximum(F, 0), np.maximum(S, 0)
        m = mn.Mechanism(family="N4_SUBUNIT", beta=1.0, gamma=0.5)
        np.testing.assert_allclose(mn.apply_mechanism(F, S, m),
                                   np.maximum(Fr - 1.0 * Sr + 0.5 * Fr * Sr, 0.0))
        m0 = mn.Mechanism(family="N4_SUBUNIT", beta=1.0, gamma=0.0)
        np.testing.assert_allclose(mn.apply_mechanism(F, S, m0), np.maximum(Fr - Sr, 0.0))

    def test_opponent_is_antisymmetric_and_product_is_not(self):
        F, S = drives()
        opp = mn.Mechanism(family="OPPONENT_SIGNED", delta=1)
        a = mn.apply_mechanism(F, S, opp)
        b = mn.apply_mechanism(S, F, opp)            # swapping arms must invert the sign
        raw_a = F * mn._shift(S, 1) - S * mn._shift(F, 1)
        np.testing.assert_allclose(a, np.maximum(raw_a, 0.0))
        np.testing.assert_allclose(b, np.maximum(-raw_a, 0.0))
        p = mn.Mechanism(family="P_ONLY_SIGNED", delta=1)
        self.assertFalse(np.allclose(mn.apply_mechanism(F, S, p),
                                     mn.apply_mechanism(S, F, p)))

    def test_shift_delays_without_wrapping(self):
        x = np.arange(10, dtype=float)[None, :]
        y = mn._shift(x, 2)
        self.assertEqual(y[0, 0], 0.0)               # edge padded with first sample
        np.testing.assert_allclose(y[0, 2:], x[0, :-2])
        np.testing.assert_allclose(mn._shift(x, 0), x)

    def test_all_outputs_non_negative(self):
        F, S = drives(seed=3)
        for fam in mn.SELECTABLE + mn.CONTROL_ONLY:
            y = mn.apply_mechanism(F, S, mn.Mechanism(family=fam))
            self.assertGreaterEqual(y.min(), 0.0, fam)

    def test_signed_vs_magnitude_family_routing(self):
        for fam in ("N0_E18", "N1_LEXT", "N5_HR_CONTROL", "P_ONLY_SIGNED", "OPPONENT_SIGNED"):
            self.assertIn(fam, mn.SIGNED_FAMILIES)
        for fam in ("N2_COINCIDENCE", "N3_SHUNT", "N4_SUBUNIT", "P_ONLY", "OPPONENT_PRODUCT"):
            self.assertNotIn(fam, mn.SIGNED_FAMILIES)

    def test_controls_never_selectable(self):
        self.assertFalse(set(mn.CONTROL_ONLY) & set(mn.SELECTABLE))
        for fam in mn.CONTROL_ONLY:
            self.assertNotIn(fam, mn.SIMPLICITY)


class AblationTests(unittest.TestCase):
    def test_temporal_flat_equalises_taus(self):
        m = mn.Mechanism(family="N2_COINCIDENCE", tau_fast=2.0, tau_slow=12.0)
        f = mn.temporal_flat(m)
        self.assertEqual(f.tau_slow, f.tau_fast)
        self.assertEqual(f.tau_fast, 2.0)

    def test_interaction_ablation_removes_the_term(self):
        self.assertEqual(mn.interaction_ablation(
            mn.Mechanism(family="N3_SHUNT", alpha=2.0)).alpha, 0.0)
        self.assertEqual(mn.interaction_ablation(
            mn.Mechanism(family="N4_SUBUNIT", gamma=1.0)).gamma, 0.0)
        self.assertEqual(mn.interaction_ablation(
            mn.Mechanism(family="N2_COINCIDENCE")).gamma, 0.0)
        self.assertEqual(mn.interaction_ablation(
            mn.Mechanism(family="N5_HR_CONTROL")).family, "N1_LEXT")

    def test_interaction_ablation_equivalence(self):
        """Ablating N4's gamma must equal the pure subtractive parent."""
        F, S = drives(seed=7)
        m = mn.Mechanism(family="N4_SUBUNIT", beta=1.0, gamma=1.0)
        a = mn.apply_mechanism(F, S, mn.interaction_ablation(m))
        Fr, Sr = np.maximum(F, 0), np.maximum(S, 0)
        np.testing.assert_allclose(a, np.maximum(Fr - 1.0 * Sr, 0.0))

    def test_geometry_permutation_preserves_marginals_and_is_reproducible(self):
        m = sparse.csr_matrix(np.array([[1.0, 0, 2.0], [0, 3.0, 0]]))
        rng1 = np.random.default_rng(5); rng2 = np.random.default_rng(5)
        p1 = rng1.permutation(3); p2 = rng2.permutation(3)
        np.testing.assert_array_equal(p1, p2)                 # reproducible
        np.testing.assert_allclose(np.sort(m[:, p1].toarray(), axis=None),
                                   np.sort(m.toarray(), axis=None))   # weights preserved


class TemporalMetricTests(unittest.TestCase):
    def test_motion_window_is_predetermined(self):
        self.assertEqual(R.MOTION_WINDOW, (mn.STARTUP_FRAMES, R.FRAMES))
        self.assertEqual(mn.STARTUP_FRAMES, 10)

    def test_metric_traces_window(self):
        Y = np.tile(np.arange(10, dtype=float), (3, 1))
        np.testing.assert_allclose(mn.metric_traces(Y), Y.mean(axis=1))
        np.testing.assert_allclose(mn.metric_traces(Y, (2, 5)), Y[:, 2:5].mean(axis=1))

    def test_event_window_uses_given_frames_not_response_peak(self):
        Y = np.zeros((2, 20)); Y[0, 0] = 100.0; Y[1, 19] = 100.0
        # crossing frames are supplied (from stimulus+anatomy), so the huge peaks at the
        # edges must be ignored when they fall outside the window
        out = mn.event_window_means(Y, np.array([10, 10]), half=2)
        np.testing.assert_allclose(out, [0.0, 0.0])

    def test_predicted_crossing_matches_bar_trajectory(self):
        """The predicted crossing frame must track the rendered bar centre."""
        frames, speed, bar = 60, 1.0, 18
        seq = cm.parametric_bar("right", "OFF", frames=frames, speed=speed, bar_px=bar)
        centres = [float(np.mean(np.flatnonzero(f[0] < 0.5))) for f in seq]
        for probe in (20, 35, 50):
            u = centres[probe] / (cm.W - 1)
            est = mn.predicted_crossing(np.array([u]), np.array([0.5]), "right",
                                        frames, speed, bar)
            self.assertLess(abs(float(est[0]) - probe), 1.5)

    def test_predicted_crossing_reverses_for_opposite_direction(self):
        a = mn.predicted_crossing(np.array([0.2]), np.array([0.5]), "right", 60, 1.0, 18)
        b = mn.predicted_crossing(np.array([0.2]), np.array([0.5]), "left", 60, 1.0, 18)
        self.assertAlmostEqual(float(a[0]) + float(b[0]), 59.0, places=5)


class SplitAndLeakageTests(unittest.TestCase):
    def test_calibration_and_heldout_stimulus_values_disjoint(self):
        self.assertFalse(set(R.CAL_SPEEDS) & set(R.HELDOUT_SPEEDS))

    def test_extended_tau_grid_applied(self):
        fams = {}
        for m in R.build_grid():
            fams.setdefault(m.family, set()).add(m.tau_slow)
        self.assertTrue({12.0, 16.0, 24.0} <= fams["N1_LEXT"])   # beyond E18's 8
        self.assertTrue({3.0, 5.0, 8.0} <= fams["N1_LEXT"])      # includes E18's grid

    def test_grid_has_no_per_subtype_or_per_neuron_parameters(self):
        for m in R.build_grid():
            for f in ("T4a", "T5a", "subtype", "neuron"):
                self.assertNotIn(f, json.dumps(m.to_json()))

    def test_calibration_stage_never_uses_heldout_bank(self):
        src = (ROOT / "run_motion_nonlinearity_experiment.py").read_text()
        block = src.split("def stage_calibrate")[1].split("def stage_heldout")[0]
        self.assertIn('stimulus_bank("calibration")', block)
        self.assertNotIn('stimulus_bank("heldout")', block)
        self.assertNotIn("HELDOUT_SPEEDS", block)

    def test_selected_model_is_never_a_control(self):
        p = EXP19 / "selected-model.json"
        if not p.exists():
            self.skipTest("calibration not yet run")
        sel = json.loads(p.read_text())
        self.assertIn(sel["mechanism"]["family"], mn.SELECTABLE)
        self.assertNotIn(sel["mechanism"]["family"], mn.CONTROL_ONLY)

    def test_no_game_supervision(self):
        forbidden = [r"PokemonEmulator", r"import\s+pyboy", r"from\s+pyboy", r"load_state",
                     r"read_memory", r"game_ram", r"player_x", r"player_y", r"room_id",
                     r"map_id", r"reinforc", r"q_learning", r"behavior_cloning",
                     r"policy_gradient", r"desired_action", r"press_button", r"\.tap\(",
                     r"reward\s*="]
        for f in ("flymon/motion_nonlinearity.py", "run_motion_nonlinearity_experiment.py",
                  "analyze_motion_nonlinearity_experiment.py"):
            path = ROOT / f
            if not path.exists():
                continue
            text = path.read_text()
            for pat in forbidden:
                self.assertIsNone(re.search(pat, text, re.IGNORECASE), f"{pat} in {f}")

    def test_no_controller_or_vp_dn_modules(self):
        text = (ROOT / "run_motion_nonlinearity_experiment.py").read_text()
        for mod in ("flymon.controller", "flymon.emulator", "flymon.motor", "flymon.steering",
                    "flymon.locomotion", "flymon.progression", "flymon.interaction"):
            self.assertNotIn(mod, text)


class GeometryReuseTests(unittest.TestCase):
    def test_arms_frozen_from_experiment18(self):
        self.assertEqual(oc.FAST_ARM["T4"], ("Mi1",))
        self.assertEqual(oc.SLOW_ARM["T4"], ("Mi9", "Mi4"))
        self.assertEqual(oc.FAST_ARM["T5"], ("Tm1", "Tm2"))
        self.assertEqual(oc.SLOW_ARM["T5"], ("Tm9",))

    def test_eye_mirroring_preserved(self):
        v = np.array([1.0, 0.0])
        self.assertEqual(oc.cardinal(oc.screen_direction(v, "L")), "right")
        self.assertEqual(oc.cardinal(oc.screen_direction(v, "R")), "left")
        w = np.array([0.0, 1.0])
        self.assertEqual(oc.cardinal(oc.screen_direction(w, "L")),
                         oc.cardinal(oc.screen_direction(w, "R")))   # vertical not mirrored

    def test_dsi_formula(self):
        self.assertAlmostEqual(mn.dsi(1.0, 0.0), 1.0, places=6)
        self.assertAlmostEqual(mn.dsi(1.0, 1.0), 0.0, places=6)
        self.assertAlmostEqual(mn.dsi(0.0, 1.0), -1.0, places=6)

    def test_experiment18_artifacts_unchanged(self):
        """E19 must not rewrite any Experiment-18 result file."""
        sel = EXP18 / "selected-config.json"
        if not sel.exists():
            self.skipTest("E18 artifacts absent")
        d = json.loads(sel.read_text())
        self.assertEqual(d["selected"], "M1|tf2.0|ts8.0|p1.0")
        self.assertFalse(d["met_preregistered_threshold"])
        g = json.loads((EXP18 / "geometry-summary.json").read_text())
        self.assertAlmostEqual(g["T5a"]["concentration"], 0.884, places=2)


if __name__ == "__main__":
    unittest.main()
