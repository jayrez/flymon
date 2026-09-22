"""Experiment 22 tests: fixed reference, pre-adaptation, polarity, controls, leakage."""
import json
import re
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from flymon import adapted_reference as ar
from flymon import column_motion as cm
from flymon import tonic_disinhibition as td

ROOT = Path(__file__).resolve().parent
EXP22 = ROOT / "results" / "experiment-22-fixed-background-adaptation"
CHEM = {"Mi1": "exc", "Mi9": "inh", "Mi4": "inh", "Tm1": "exc", "Tm2": "exc", "Tm9": "exc"}


def toy_proj():
    return {"Mi1": sparse.csr_matrix(np.array([[0.5, 0.0, 0.0]])),
            "Mi9": sparse.csr_matrix(np.array([[0.0, -0.3, 0.0]])),
            "Mi4": sparse.csr_matrix(np.array([[0.0, 0.0, -0.2]]))}


def lum_seq(T=30, cols=3, L0=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return (L0 + rng.normal(0, 0.1, size=(T, cols))).astype(np.float32)


def sim(cfg, lum, L0=0.5):
    sig = ar.cell_signals(lum, cfg.tau_fast, cfg.tau_slow, cfg.reference,
                          L0 if cfg.reference == "fixed" else None, cfg.adapt)
    return ar.simulate(toy_proj(), sig, cfg, 0.2, 1.0, CHEM)


class SignalPath(unittest.TestCase):
    def test_fixed_reference_is_L_minus_L0(self):
        lum = lum_seq()
        np.testing.assert_allclose(ar.contrast(lum, "fixed", 0.37), lum - 0.37)
        sig = ar.cell_signals(lum, 0.0, 0.0, "fixed", 0.37, 0, temporal=False)
        np.testing.assert_allclose(sig["Mi1"], lum - 0.37, rtol=1e-6)
        np.testing.assert_allclose(sig["Mi9"], -(lum - 0.37), rtol=1e-6)

    def test_sequence_mean_control_reproduces_canonical(self):
        lum = lum_seq(seed=3)
        ours = ar.cell_signals(lum, 4.0, 12.0, "sequence_mean")
        ref = td.signals_for(lum, td.TonicConfig())
        for k in ref:
            np.testing.assert_array_equal(ours[k], ref[k])

    def test_sequence_mean_rejects_adaptation(self):
        with self.assertRaises(ValueError):
            ar.cell_signals(lum_seq(), 4.0, 12.0, "sequence_mean", adapt=10)

    def test_static_L0_gives_zero_contrast(self):
        lum = np.full((20, 4), 0.42, np.float32)
        sig = ar.cell_signals(lum, 4.0, 12.0, "fixed", 0.42, 40)
        for s in sig.values():
            self.assertEqual(float(np.abs(s).max()), 0.0)

    def test_pre_adaptation_initialises_filters_at_zero_contrast(self):
        lum = np.full((10, 2), 0.9, np.float32)           # a step at stimulus onset
        adapted = ar.cell_signals(lum, 4.0, 12.0, "fixed", 0.5, 40)
        cold = ar.cell_signals(lum, 4.0, 12.0, "fixed", 0.5, 0)
        a = ar.drop_adaptation(adapted["Mi1"], 40, axis=0)
        self.assertEqual(adapted["Mi1"].shape[0], 50)
        self.assertTrue(np.all(adapted["Mi1"][:40] == 0.0))
        self.assertLess(a[0, 0], 0.4 * 0.99)              # rising from rest, not at steady state
        np.testing.assert_allclose(cold["Mi1"][0], 0.4, rtol=1e-6)   # cold filter starts at frame 0

    def test_adaptation_frames_excluded_from_scoring(self):
        cfg = ar.AdaptedConfig(adapt=25)
        Y, aux = sim(cfg, lum_seq(T=30))
        self.assertEqual(Y.shape[1], 30)
        self.assertEqual(aux["V"].shape[1], 30)
        for c in aux["classes"].values():
            self.assertEqual(c["gE"].shape[1], 30)

    def test_step_polarity(self):
        L0 = 0.5
        for pol, sign in (("ON", +1), ("OFF", -1)):
            lum = np.concatenate([np.full((10, 2), L0), np.full((20, 2), L0 + sign * 0.25)]).astype(np.float32)
            sig = ar.cell_signals(lum, 4.0, 12.0, "fixed", L0, 40)
            post = {k: ar.drop_adaptation(v, 40, axis=0)[10:].mean() for k, v in sig.items()}
            pre = {k: np.abs(ar.drop_adaptation(v, 40, axis=0)[:10]).max() for k, v in sig.items()}
            for k in ("Mi1", "Mi4"):
                self.assertEqual(np.sign(post[k]), sign)
            for k in ("Mi9", "Tm1", "Tm2", "Tm9"):
                self.assertEqual(np.sign(post[k]), -sign)
            self.assertTrue(all(v == 0.0 for v in pre.values()))


class LuminanceScale(unittest.TestCase):
    def test_display_gray_is_not_linear_half(self):
        self.assertLess(ar.rendered_linear(0.5), 0.25)

    def test_measured_L0_close_to_declared(self):
        self.assertAlmostEqual(ar.measured_L0(0.5), 0.5, delta=0.01)

    def test_bars_symmetric_around_background(self):
        hi, lo = ar.weber_levels(0.5, 1.0)
        self.assertAlmostEqual(hi - 0.5, 0.5 - lo)
        hi, lo = ar.weber_levels(0.6, 1.0)
        self.assertLessEqual(hi, 1.0); self.assertGreaterEqual(lo, 0.0)
        self.assertAlmostEqual(hi - 0.6, 0.6 - lo)

    def test_gray_bar_uses_e19_trajectory(self):
        a = ar.gray_bar("right", "ON", 20, 2.0)
        mask = cm.parametric_bar("right", "ON", 20, 2.0) > 0.5
        bg = ar.display_for(0.5)
        np.testing.assert_array_equal(a != np.float32(bg), mask)


class ModelPhysics(unittest.TestCase):
    def test_no_negative_conductance(self):
        for cfg in (ar.AdaptedConfig(beta=5.0, r0_mi9=1.0, r0_mi4=0.5, adapt=40),
                    ar.AdaptedConfig(beta=5.0, adapt=0, membrane_cold=True)):
            _, aux = sim(cfg, lum_seq(seed=2))
            for c in aux["classes"].values():
                self.assertGreaterEqual(c["gE"].min(), 0.0)
                self.assertGreaterEqual(c["gI"].min(), 0.0)

    def test_deterministic_and_finite(self):
        cfg = ar.AdaptedConfig(adapt=40, r0_mi9=0.5)
        a, _ = sim(cfg, lum_seq(seed=5)); b, _ = sim(cfg, lum_seq(seed=5))
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(a).all())

    def test_adapted_rest_matches_tonic_rest(self):
        cfg = ar.AdaptedConfig(adapt=40, r0_mi9=1.0)
        Y, aux = sim(cfg, np.full((10, 3), 0.5, np.float32))
        np.testing.assert_allclose(aux["V"], np.broadcast_to(aux["V_rest"][:, None], aux["V"].shape), atol=1e-12)
        self.assertEqual(float(Y.max()), 0.0)

    def test_mi9_mi4_tonic_separate(self):
        cfg = ar.AdaptedConfig(r0_mi9=1.0, r0_mi4=0.0, adapt=40)
        _, aux = sim(cfg, np.full((10, 3), 0.5, np.float32))
        self.assertGreater(aux["classes"]["Mi9"]["gI0"][0], 0.0)
        self.assertEqual(aux["classes"]["Mi4"]["gI0"][0], 0.0)

    def test_drop_classes(self):
        _, aux = sim(ar.AdaptedConfig(drop=("Mi9", "Mi4")), lum_seq())
        self.assertEqual(set(aux["classes"]), {"Mi1"})

    def test_temporal_flat(self):
        c = ar.temporal_flat(ar.AdaptedConfig())
        self.assertEqual(c.tau_slow, c.tau_fast)

    def test_config_roundtrip(self):
        c = ar.AdaptedConfig(drop=("Mi9",), adapt=40, r0_mi4=0.5)
        self.assertEqual(ar.AdaptedConfig.from_json(json.loads(json.dumps(c.to_json()))), c)

    def test_no_per_subtype_parameters(self):
        fields = set(ar.AdaptedConfig().to_json())
        self.assertFalse(any(re.search(r"T[45][abcd]", f) for f in fields))


class Runner(unittest.TestCase):
    def test_geometry_shuffle_permutes_columns(self):
        import run_fixed_background_experiment as R
        ctx = dict(colindex={i: i for i in range(5)},
                   proj={"T4a": {t: sparse.csr_matrix(np.arange(5, dtype=float)[None, :] + 1)
                                 for t in cm.ALL_PARTNERS}})
        p = R.permuted_projections(ctx, 3, 1)
        for q in p:
            v = q["T4a"]["Mi1"].toarray().ravel()
            self.assertEqual(sorted(v), [1, 2, 3, 4, 5])
        self.assertTrue(any(not np.array_equal(q["T4a"]["Mi1"].toarray().ravel(), np.arange(1, 6))
                            for q in p))

    def test_calibration_and_heldout_speeds_disjoint(self):
        import run_fixed_background_experiment as R
        cal = {m["speed"] for _s, m in R.gray_bank("calibration").values() if "speed" in m}
        held = {m["speed"] for _s, m in R.gray_bank("heldout").values()
                if m.get("axis") == "speed"}
        self.assertFalse(cal & held)

    def test_grid_is_class_level_and_preregistered_size(self):
        import run_fixed_background_experiment as R
        g = R.build_grid()
        self.assertEqual(len(g), 288)
        self.assertEqual(len({c.key() for c in g}), 288)

    def test_no_oracle_import_in_model(self):
        src = (ROOT / "flymon" / "adapted_reference.py").read_text()
        self.assertNotIn("motion_nonlinearity", src)
        self.assertNotIn("OPPONENT", src)
        self.assertNotIn("heldout", src.lower())


@unittest.skipUnless((EXP22 / "heldout-results.json").exists(), "held-out not run")
class SavedArtifacts(unittest.TestCase):
    def test_r0_reproduced(self):
        sel = json.loads((EXP22 / "selected-config.json").read_text())
        self.assertTrue(sel["r0_reproduction"]["matches"])
        h = json.loads((EXP22 / "heldout-results.json").read_text())
        self.assertTrue(h["reference_conditions"]["E21_params"]["R0"]["matches_e21_saved_heldout"])

    def test_polarity_audit_passed(self):
        a = json.loads((EXP22 / "polarity-audit.json").read_text())
        self.assertTrue(all(v["passed"] for k, v in a.items() if k.startswith(("R1", "R2"))))

    def test_selected_frozen_before_heldout(self):
        sel = json.loads((EXP22 / "selected-config.json").read_text())
        h = json.loads((EXP22 / "heldout-results.json").read_text())
        self.assertEqual(sel["config"], h["config"])

    def test_output_not_rescaled_oracle(self):
        z = np.load(EXP22 / "heldout-per-neuron.npz")
        for s in ("T4a", "T4b", "T4c", "T4d"):
            a, b = z[f"e22_{s}_dsi"], z[f"oracle_{s}_dsi"]
            if np.std(a) > 0 and np.std(b) > 0:
                self.assertLess(abs(np.corrcoef(a, b)[0, 1]), 0.999)

    def test_saved_summary_matches_per_neuron(self):
        z = np.load(EXP22 / "heldout-per-neuron.npz")
        h = json.loads((EXP22 / "heldout-results.json").read_text())
        for s, v in h["heldout"]["subtypes"].items():
            self.assertAlmostEqual(v["dsi_mean"], float(z[f"e22_{s}_dsi"].mean()), places=10)


if __name__ == "__main__":
    unittest.main()
