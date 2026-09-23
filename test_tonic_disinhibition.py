"""Experiment 21 tests: tonic conductance physics, release, ablations, leakage, anti-cheating."""
import json
import re
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from flymon import tonic_disinhibition as td

ROOT = Path(__file__).resolve().parent
EXP20 = ROOT / "results" / "experiment-20-conductance-dendrite"
EXP21 = ROOT / "results" / "experiment-21-tonic-disinhibition"
CHEM = {"Mi1": "exc", "Mi9": "inh", "Mi4": "inh", "Tm1": "exc", "Tm2": "exc", "Tm9": "exc"}


def toy_proj():
    return {"Mi1": sparse.csr_matrix(np.array([[0.5, 0.0, 0.0]])),
            "Mi9": sparse.csr_matrix(np.array([[0.0, -0.3, 0.0]])),
            "Mi4": sparse.csr_matrix(np.array([[0.0, 0.0, -0.2]]))}


def signals(T=20, value=None, seed=0):
    rng = np.random.default_rng(seed)
    if value is None:
        return {t: rng.normal(0, 0.2, size=(T, 3)) for t in td.T4_CLASSES}
    return {t: np.full((T, 3), value) for t in td.T4_CLASSES}


def run(cfg, sig, sigma=0.2, M=1.0):
    return td.simulate(toy_proj(), sig, cfg, sigma, M, CHEM)


class ConductancePhysics(unittest.TestCase):
    def test_conductances_never_negative(self):
        for fam, bounded in (("D1_UNIFORM_TONIC", False), ("D2_TONIC_INHIBITION", False),
                             ("D3_BOUNDED_TONIC_INHIBITION", True)):
            cfg = td.TonicConfig(family=fam, bounded=bounded, beta=5.0, r0_inh=1.0)
            _, aux = run(cfg, signals(seed=1))
            for c in aux["classes"].values():
                self.assertGreaterEqual(c["gE"].min(), 0.0)
                self.assertGreaterEqual(c["gI"].min(), 0.0)

    def test_tonic_baseline_present(self):
        _, aux = run(td.TonicConfig(r0_inh=1.0, G=2.0), signals(value=0.0))
        self.assertGreater(aux["classes"]["Mi9"]["gI0"][0], 0.0)
        g = aux["classes"]["Mi9"]["gI"]
        np.testing.assert_allclose(g, np.broadcast_to(aux["classes"]["Mi9"]["gI0"][:, None], g.shape))

    def test_inhibition_can_fall_below_baseline_but_not_zero(self):
        # a strongly negative signal = OFF-cell Mi9 silenced by ON motion
        _, aux = run(td.TonicConfig(r0_inh=1.0, beta=1.0), signals(value=-0.1))
        g, g0 = aux["classes"]["Mi9"]["gI"], aux["classes"]["Mi9"]["gI0"]
        self.assertTrue((g < g0[:, None]).all())
        self.assertTrue((g >= 0).all())
        _, aux2 = run(td.TonicConfig(r0_inh=1.0, beta=1.0), signals(value=-10.0))
        self.assertEqual(aux2["classes"]["Mi9"]["gI"].min(), 0.0)

    def test_zero_baseline_mode_is_e20_like(self):
        """r0 = 0 reproduces the E20 max(s, 0) presynaptic mapping (per unit scale)."""
        sig = signals(seed=4)
        cfg = td.TonicConfig(family="D1_UNIFORM_TONIC", r0_exc=0.0, r0_inh=0.0, beta=1.0)
        r = td.presynaptic_rate(sig["Mi9"], 0.0, cfg, sigma=1.0, inhibitory=True)
        np.testing.assert_allclose(r, np.maximum(sig["Mi9"], 0.0))

    def test_static_input_stays_at_rest_and_output_zero(self):
        for cfg in (td.TonicConfig(r0_inh=1.0, E_inh=-0.2), td.TonicConfig(family="D1_UNIFORM_TONIC",
                                                                         r0_exc=1.0, r0_inh=1.0)):
            Y, aux = run(cfg, signals(value=0.0))
            np.testing.assert_allclose(Y, 0.0, atol=1e-12)
            np.testing.assert_allclose(aux["V"], np.broadcast_to(aux["V_rest"][:, None], aux["V"].shape), atol=1e-12)

    def test_release_raises_input_resistance(self):
        _, aux = run(td.TonicConfig(r0_inh=1.0, beta=1.0), {"Mi1": np.zeros((5, 3)),
                                                            "Mi9": np.full((5, 3), -0.1),
                                                            "Mi4": np.zeros((5, 3))})
        self.assertTrue((aux["R_in"] > aux["R_rest"][:, None]).all())

    def test_deterministic_and_finite(self):
        cfg = td.TonicConfig(family="D3_BOUNDED_TONIC_INHIBITION", bounded=True, G=16.0, beta=2.0)
        a, _ = run(cfg, signals(seed=3)); b, _ = run(cfg, signals(seed=3))
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(a).all())

    def test_bounded_transfer_range(self):
        cfg = td.TonicConfig(bounded=True, r0_inh=1.0, beta=2.0)
        s = np.linspace(-5, 5, 101)[None, :]
        r = td.presynaptic_rate(s, 1.0, cfg, sigma=0.2, inhibitory=True)
        self.assertGreaterEqual(r.min(), 0.0); self.assertLessEqual(r.max(), 2.0)
        self.assertAlmostEqual(float(td.presynaptic_rate(np.zeros((1, 1)), 1.0, cfg, 0.2, True)[0, 0]), 1.0)


class Ablations(unittest.TestCase):
    def test_clamp_holds_inhibition_at_baseline(self):
        _, aux = run(td.TonicConfig(r0_inh=1.0, clamp_inhibition=True), signals(seed=5))
        for t in ("Mi9", "Mi4"):
            c = aux["classes"][t]
            np.testing.assert_allclose(c["gI"], np.repeat(c["gI0"][:, None], c["gI"].shape[1], 1))

    def test_no_release_blocks_decrease_only(self):
        _, aux = run(td.TonicConfig(r0_inh=1.0, no_release=True), signals(seed=6))
        c = aux["classes"]["Mi9"]
        self.assertTrue((c["gI"] >= c["gI0"][:, None] - 1e-12).all())
        _, up = run(td.TonicConfig(r0_inh=1.0, no_release=True), signals(value=0.3))
        self.assertTrue((up["classes"]["Mi9"]["gI"] > up["classes"]["Mi9"]["gI0"][:, None]).all())

    def test_class_removal(self):
        for t in ("Mi1", "Mi9", "Mi4"):
            _, aux = run(td.TonicConfig(drop_classes=(t,)), signals(seed=7))
            self.assertNotIn(t, aux["classes"])

    def test_neutral_inhibition_moves_reversal_only(self):
        cfg = td.TonicConfig(r0_inh=1.0, E_inh=-0.2)
        _, a = run(cfg, signals(seed=8)); _, b = run(cfg.replace(neutral_inhibition=True), signals(seed=8))
        np.testing.assert_allclose(a["gI"], b["gI"])
        self.assertFalse(np.allclose(a["V"], b["V"]))

    def test_temporal_flat(self):
        self.assertEqual(td.temporal_flat(td.TonicConfig()).tau_slow, td.TonicConfig().tau_fast)


class RoutingAndLeakage(unittest.TestCase):
    def test_chemistry_routing(self):
        _, aux = run(td.TonicConfig(r0_exc=1.0, r0_inh=1.0, family="D1_UNIFORM_TONIC"), signals(value=0.0))
        self.assertGreater(aux["classes"]["Mi1"]["gE0"][0], 0); self.assertEqual(aux["classes"]["Mi1"]["gI0"][0], 0)
        self.assertEqual(aux["classes"]["Mi9"]["gE0"][0], 0); self.assertGreater(aux["classes"]["Mi9"]["gI0"][0], 0)

    def test_grid_has_no_subtype_direction_or_neuron_parameters(self):
        import run_tonic_disinhibition_experiment as R
        for c in R.build_grid():
            blob = json.dumps(c.to_json())
            for bad in ("T4a", "T5a", "subtype", "direction", "neuron", "left", "right"):
                self.assertNotIn(bad, blob)

    def test_calibration_never_uses_heldout(self):
        src = (ROOT / "run_tonic_disinhibition_experiment.py").read_text()
        block = src.split("def stage_calibrate")[1].split("def stage_heldout")[0]
        self.assertIn('stimulus_bank("calibration")', block)
        self.assertNotIn('stimulus_bank("heldout")', block)

    def test_model_does_not_use_oracle(self):
        src = (ROOT / "flymon/tonic_disinhibition.py").read_text()
        for bad in (r"motion_nonlinearity", r"apply_mechanism", r"OPPONENT", r"_shift\s*\(",
                    r"np\.correlate"):
            self.assertIsNone(re.search(bad, src), bad)

    def test_no_game_supervision(self):
        forbidden = [r"PokemonEmulator", r"import\s+pyboy", r"load_state", r"read_memory",
                     r"player_x", r"room_id", r"map_id", r"reinforc", r"behavior_cloning",
                     r"desired_action", r"\.tap\(", r"reward\s*="]
        for f in ("flymon/tonic_disinhibition.py", "run_tonic_disinhibition_experiment.py",
                  "analyze_tonic_disinhibition_experiment.py"):
            p = ROOT / f
            if p.exists():
                for pat in forbidden:
                    self.assertIsNone(re.search(pat, p.read_text(), re.IGNORECASE), f"{pat} in {f}")

    def test_e20_artifacts_unchanged(self):
        g = json.loads((EXP20 / "gate-status.json").read_text())
        self.assertEqual((g["verdict"], g["category"]), ("FAIL", "E"))
        self.assertAlmostEqual(g["families"]["C2_TWOCOMP_NL"]["mean_dsi"], 0.0246, places=3)


class SavedArtifacts(unittest.TestCase):
    def test_d0_reproduces_e20(self):
        p = EXP21 / "selected-config.json"
        if not p.exists():
            self.skipTest("calibration not run")
        self.assertTrue(json.loads(p.read_text())["d0_reproduction"]["matches"])

    def test_metrics_reproduce_from_saved_arrays(self):
        p, q = EXP21 / "heldout-results.json", EXP21 / "heldout-per-neuron.npz"
        if not (p.exists() and q.exists()):
            self.skipTest("held-out not run")
        out = json.loads(p.read_text()); z = np.load(q)
        for s, v in out["heldout"]["subtypes"].items():
            self.assertAlmostEqual(float(z[f"e21_{s}_dsi"].mean()), v["dsi_mean"], places=9)

    def test_output_not_rescaled_oracle(self):
        q = EXP21 / "heldout-per-neuron.npz"
        if not q.exists():
            self.skipTest("held-out not run")
        z = np.load(q)
        for k in [k for k in z.files if k.startswith("e21_") and k.endswith("_dsi")]:
            s = k.split("_")[1]
            a, b = z[k], z[f"oracle_{s}_dsi"]; n = min(len(a), len(b))
            self.assertLess(abs(np.corrcoef(a[:n], b[:n])[0, 1]), 0.999, s)


if __name__ == "__main__":
    unittest.main()
