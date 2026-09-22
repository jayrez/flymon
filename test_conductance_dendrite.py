"""Experiment 20 tests: conductance physics, sign mapping, ablations, anti-cheating."""
import json
import re
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from flymon import conductance_dendrite as cd
from flymon import column_motion as cm

ROOT = Path(__file__).resolve().parent
EXP19 = ROOT / "results" / "experiment-19-motion-nonlinearity"
EXP20 = ROOT / "results" / "experiment-20-conductance-dendrite"


def toy_projection():
    """One T5-like neuron: fast arm excitatory (Tm1), slow arm excitatory (Tm9);
    plus a T4-like inhibitory slow arm (Mi9 negative weight)."""
    ncol = 4
    proj = {
        "Tm1": sparse.csr_matrix(np.array([[1.0, 0, 0, 0]])),
        "Tm9": sparse.csr_matrix(np.array([[0, 0, 0.5, 0]])),
        "Mi1": sparse.csr_matrix(np.array([[1.0, 0, 0, 0]])),
        "Mi9": sparse.csr_matrix(np.array([[0, 0, -0.5, 0]])),
    }
    return proj, ncol


def toy_signals(T=20, ncol=4, seed=0):
    rng = np.random.default_rng(seed)
    return {t: rng.normal(size=(T, ncol)) for t in ("Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9")}


class ConductancePhysicsTests(unittest.TestCase):
    def test_conductances_are_non_negative(self):
        proj, _ = toy_projection(); sig = toy_signals()
        for fam in ("T4", "T5"):
            g = cd.arm_conductances(proj, sig, fam, cd.ConductanceConfig())
            for arm, (gE, gI) in g.items():
                self.assertGreaterEqual(gE.min(), 0.0, f"{fam}/{arm} gE")
                self.assertGreaterEqual(gI.min(), 0.0, f"{fam}/{arm} gI")

    def test_weight_sign_routes_channel_not_negative_conductance(self):
        """Negative MaleCNS weight must create inhibitory conductance, never negative gE."""
        proj, _ = toy_projection(); sig = toy_signals()
        g4 = cd.arm_conductances(proj, sig, "T4", cd.ConductanceConfig())
        # T4 fast (Mi1, +w) is purely excitatory; slow (Mi9, -w) purely inhibitory
        self.assertGreater(g4["fast"][0].sum(), 0.0)
        self.assertEqual(g4["fast"][1].sum(), 0.0)
        self.assertEqual(g4["slow"][0].sum(), 0.0)
        self.assertGreater(g4["slow"][1].sum(), 0.0)
        # T5 slow (Tm9, +w) is excitatory -- the pathway asymmetry survives
        g5 = cd.arm_conductances(proj, sig, "T5", cd.ConductanceConfig())
        self.assertGreater(g5["slow"][0].sum(), 0.0)
        self.assertEqual(g5["slow"][1].sum(), 0.0)

    def test_presynaptic_rectification_not_arm_magnitude(self):
        """Activation is ReLU at the presynaptic cell; negative signal contributes zero."""
        proj, ncol = toy_projection()
        sig = {t: np.full((5, ncol), -1.0) for t in ("Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9")}
        g = cd.arm_conductances(proj, sig, "T5", cd.ConductanceConfig())
        for arm, (gE, gI) in g.items():
            self.assertEqual(gE.sum(), 0.0)
            self.assertEqual(gI.sum(), 0.0)

    def test_zero_input_stays_at_rest(self):
        proj, ncol = toy_projection()
        sig = {t: np.zeros((10, ncol)) for t in ("Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9")}
        for model in cd.MODELS:
            cfg = cd.ConductanceConfig(model=model)
            Y, aux = cd.simulate(proj, sig, "T5", cfg)
            np.testing.assert_allclose(Y, 0.0, atol=1e-12)
            np.testing.assert_allclose(aux["V_A"], cfg.E_leak, atol=1e-12)

    def test_voltage_bounded_by_reversals(self):
        proj, _ = toy_projection(); sig = toy_signals(seed=2)
        cfg = cd.ConductanceConfig(model="C1_TWOCOMP", g_syn=4.0)
        _, aux = cd.simulate(proj, sig, "T4", cfg)
        for V in (aux["V_A"], aux["V_B"]):
            self.assertGreaterEqual(V.min(), min(cfg.E_inh, cfg.E_leak) - 1e-9)
            self.assertLessEqual(V.max(), cfg.E_exc + 1e-9)

    def test_no_nan_or_inf(self):
        proj, _ = toy_projection(); sig = toy_signals(seed=5)
        for model in cd.MODELS:
            for gs in (0.5, 4.0):
                Y, aux = cd.simulate(proj, sig, "T4",
                                     cd.ConductanceConfig(model=model, g_syn=gs))
                self.assertTrue(np.isfinite(Y).all())
                self.assertTrue(np.isfinite(aux["V_A"]).all())

    def test_deterministic(self):
        proj, _ = toy_projection(); sig = toy_signals(seed=1)
        cfg = cd.ConductanceConfig(model="C2_TWOCOMP_NL")
        a, _ = cd.simulate(proj, sig, "T5", cfg)
        b, _ = cd.simulate(proj, sig, "T5", cfg)
        np.testing.assert_array_equal(a, b)

    def test_output_is_non_negative_and_shaped(self):
        proj, _ = toy_projection(); sig = toy_signals(seed=6)
        Y, _ = cd.simulate(proj, sig, "T5", cd.ConductanceConfig(model="C1_TWOCOMP"))
        self.assertEqual(Y.shape, (1, 20))
        self.assertGreaterEqual(Y.min(), 0.0)

    def test_excitation_depolarises_inhibition_hyperpolarises(self):
        proj, ncol = toy_projection()
        exc = {t: np.zeros((6, ncol)) for t in ("Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9")}
        exc["Mi1"] = np.ones((6, ncol))                    # excitatory drive only
        _, a = cd.simulate(proj, exc, "T4", cd.ConductanceConfig(model="C0_SINGLE"))
        self.assertGreater(a["V_A"][0, -1], 0.0)
        inh = {t: np.zeros((6, ncol)) for t in exc}
        inh["Mi9"] = np.ones((6, ncol))                    # inhibitory drive only
        _, b = cd.simulate(proj, inh, "T4", cd.ConductanceConfig(model="C0_SINGLE", E_inh=-0.2))
        self.assertLess(b["V_A"][0, -1], 0.0)


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.proj, self.ncol = toy_projection()
        self.sig = toy_signals(seed=3)
        self.cfg = cd.ConductanceConfig(model="C1_TWOCOMP", g_couple=0.5, E_inh=-0.2)

    def test_coupling_removal_actually_decouples(self):
        d = cd.decouple(self.cfg)
        self.assertEqual(d.g_couple, 0.0)
        _, aux = cd.simulate(self.proj, self.sig, "T4", d)
        # with zero coupling, compartment B must not influence A: A equals a
        # single-compartment solution driven by the fast arm alone
        g = cd.arm_conductances(self.proj, self.sig, "T4", d)
        solo = cd.ConductanceConfig(**{**d.to_json(), "model": "C0_SINGLE"})
        VA_expect, _, _ = cd.integrate({"fast": g["fast"],
                                        "slow": (np.zeros_like(g["slow"][0]),
                                                 np.zeros_like(g["slow"][1]))}, solo)
        np.testing.assert_allclose(aux["V_A"], VA_expect, atol=1e-9)

    def test_sign_destruction_moves_inhibition_to_excitation(self):
        d = cd.destroy_sign(self.cfg)
        self.assertTrue(d.magnitude_only)
        g = cd.arm_conductances(self.proj, self.sig, "T4", d)
        self.assertEqual(g["slow"][1].sum(), 0.0)          # no inhibitory conductance left
        self.assertGreater(g["slow"][0].sum(), 0.0)        # it became excitatory

    def test_inhibition_neutralisation_sets_reversal_to_leak(self):
        n = cd.neutralise_inhibition(self.cfg)
        self.assertTrue(n.neutral_inhibition)
        E_e, E_i = cd._reversals(n)
        self.assertEqual(E_i, n.E_leak)
        # timing preserved: conductances identical to the unablated model
        a = cd.arm_conductances(self.proj, self.sig, "T4", self.cfg)
        b = cd.arm_conductances(self.proj, self.sig, "T4", n)
        np.testing.assert_allclose(a["slow"][1], b["slow"][1])

    def test_arm_swap_exchanges_compartment_inputs(self):
        s = cd.swap(self.cfg)
        self.assertTrue(s.swap_arms)
        _, normal = cd.simulate(self.proj, self.sig, "T4", self.cfg)
        _, swapped = cd.simulate(self.proj, self.sig, "T4", s)
        np.testing.assert_allclose(normal["V_A"], swapped["V_B"], atol=1e-9)
        np.testing.assert_allclose(normal["V_B"], swapped["V_A"], atol=1e-9)

    def test_temporal_flat_equalises_taus(self):
        f = cd.temporal_flat(self.cfg)
        self.assertEqual(f.tau_slow, f.tau_fast)


class AntiCheatingTests(unittest.TestCase):
    def test_module_contains_no_opponent_expression(self):
        src = (ROOT / "flymon/conductance_dendrite.py").read_text()
        banned = [r"_shift\s*\(", r"OPPONENT", r"opponent_product",
                  r"F\s*\*\s*S", r"Fr\s*\*\s*Sr", r"np\.correlate"]
        for pat in banned:
            self.assertIsNone(re.search(pat, src), f"{pat} appears in the conductance model")

    def test_model_does_not_import_the_oracle(self):
        src = (ROOT / "flymon/conductance_dendrite.py").read_text()
        self.assertNotIn("motion_nonlinearity", src)
        self.assertNotIn("apply_mechanism", src)

    def test_output_is_not_a_rescaled_copy_of_the_oracle(self):
        p = EXP20 / "heldout-per-neuron.npz"
        if not p.exists():
            self.skipTest("held-out not yet run")
        z = np.load(p)
        subs = sorted({k.split("_")[1] for k in z.files if k.startswith("cond_")})
        for s in subs:
            a, b = z[f"cond_{s}_dsi"], z[f"oracle_{s}_dsi"]
            n = min(len(a), len(b))
            r = abs(np.corrcoef(a[:n], b[:n])[0, 1])
            self.assertLess(r, 0.999, f"{s}: conductance output is a copy of the oracle")


class SplitAndLeakageTests(unittest.TestCase):
    def test_calibration_and_heldout_speeds_disjoint(self):
        import run_motion_nonlinearity_experiment as E19
        self.assertFalse(set(E19.CAL_SPEEDS) & set(E19.HELDOUT_SPEEDS))

    def test_calibration_stage_never_uses_heldout_bank(self):
        src = (ROOT / "run_conductance_dendrite_experiment.py").read_text()
        block = src.split("def stage_calibrate")[1].split("def stage_heldout")[0]
        self.assertIn('stimulus_bank("calibration")', block)
        self.assertNotIn('stimulus_bank("heldout")', block)

    def test_grid_has_no_subtype_or_direction_parameters(self):
        import run_conductance_dendrite_experiment as E20
        for c in E20.build_grid():
            blob = json.dumps(c.to_json())
            for bad in ("T4a", "T5a", "subtype", "direction", "neuron"):
                self.assertNotIn(bad, blob)

    def test_no_game_supervision(self):
        forbidden = [r"PokemonEmulator", r"import\s+pyboy", r"load_state", r"read_memory",
                     r"game_ram", r"player_x", r"room_id", r"map_id", r"reinforc",
                     r"q_learning", r"behavior_cloning", r"desired_action", r"\.tap\(",
                     r"reward\s*="]
        for f in ("flymon/conductance_dendrite.py", "run_conductance_dendrite_experiment.py",
                  "analyze_conductance_dendrite_experiment.py"):
            path = ROOT / f
            if not path.exists():
                continue
            text = path.read_text()
            for pat in forbidden:
                self.assertIsNone(re.search(pat, text, re.IGNORECASE), f"{pat} in {f}")

    def test_prior_experiment_artifacts_unchanged(self):
        p = EXP19 / "gate-status.json"
        if not p.exists():
            self.skipTest("E19 artifacts absent")
        g = json.loads(p.read_text())
        self.assertEqual(g["verdict"], "FAIL")
        self.assertEqual(g["category"], "F")
        self.assertAlmostEqual(g["decomposition"]["signed_opponent"]["mean_dsi"], 0.3562, places=3)


class ReproducibilityTests(unittest.TestCase):
    def test_metrics_reproduce_from_saved_artifacts(self):
        p = EXP20 / "heldout-results.json"
        q = EXP20 / "heldout-per-neuron.npz"
        if not (p.exists() and q.exists()):
            self.skipTest("held-out not yet run")
        out = json.loads(p.read_text()); z = np.load(q)
        for s, v in out["heldout"]["subtypes"].items():
            d = z[f"cond_{s}_dsi"]
            self.assertAlmostEqual(float(d.mean()), v["dsi_mean"], places=6)


if __name__ == "__main__":
    unittest.main()
