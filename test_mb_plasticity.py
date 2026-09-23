"""Experiment 26 Phases 3-5 tests: audited edge set, eligibility, three-factor rule, state."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flymon import mb_plasticity as mp

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "redfly-benchmark" / "data"
GPU = os.environ.get("FLYMON_GPU_TESTS") == "1"


def toy_edges():
    """4 KC cells (idx 10-13); MBON01 = 20 (appetitive), MBON11 = 21 (aversive)."""
    pre = np.array([10, 11, 12, 13, 10, 11, 12, 13]); post = np.array([20] * 4 + [21] * 4)
    comp = np.array(["appetitive"] * 4 + ["aversive"] * 4, dtype=object)
    w0 = np.array([1e-3, 2e-3, 5e-4, 1e-3, 1e-3, 3e-4, 2e-3, 1e-3])
    return mp.EdgeSet(pre, post, pre + 1000, post + 1000, w0, comp, "toyhash")


def learner(eta=0.5, tau=5.0, **kw):
    es = toy_edges()
    cfg = mp.PlasticityConfig(eta=eta, tau_elig_steps=tau, **kw)
    return es, cfg, mp.Learner(es, cfg, np.array([10, 11, 12, 13]), {"appetitive": np.array([30]), "aversive": np.array([31])})


class AuditedEdgeSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scipy import sparse
        m = np.load(DATA / "brain.npz")
        cls.ct = m["cell_type"].astype(str)
        cls.es = mp.audited_edge_set(m["cell_type"], m["ids"], sparse.load_npz(DATA / "weights.npz").tocsr())

    def test_exact_hash_and_count(self):
        self.assertTrue(self.es.sha256.startswith(mp.AUDITED_EDGE_HASH))
        self.assertEqual(self.es.n, 418)
        ref = json.loads((ROOT / "results/experiment-26/proposed-plastic-edges.json").read_text())
        self.assertEqual(self.es.sha256, ref["sha256"])

    def test_only_audited_types(self):
        self.assertTrue(np.all(self.ct[self.es.pre_idx] == "KCg-d"))
        post = self.ct[self.es.post_idx]
        self.assertTrue(set(post) == {"MBON01", "MBON11"})
        self.assertTrue(np.all((post == "MBON01") == (self.es.compartment == "appetitive")))
        self.assertEqual(int((self.es.compartment == "appetitive").sum()), 212)
        self.assertTrue(np.all(self.es.w0 > 0))

    def test_edge_metadata_immutable(self):
        with self.assertRaises(ValueError):
            self.es.w0[0] = 1.0


class Eligibility(unittest.TestCase):
    def test_decay_and_clip(self):
        es, cfg, L = learner(eta=0.0, tau=5.0)
        st = mp.PlasticState.zeros(es)
        L.step(np.array([1.0, 0, 0, 0]), {}, st)
        self.assertEqual(L.elig[0], 1.0)
        L.step(np.zeros(4), {}, st)
        self.assertAlmostEqual(L.elig[0], np.exp(-1 / 5))
        for _ in range(100):
            L.step(np.ones(4), {}, st)
        self.assertLessEqual(L.elig.max(), cfg.e_max)

    def test_activity_without_dan_changes_no_weight(self):
        es, cfg, L = learner()
        st = mp.PlasticState.zeros(es)
        for _ in range(20):
            L.step(np.ones(4), {"appetitive": 0.0, "aversive": 0.0}, st)
        self.assertTrue(np.all(st.delta == 0)); self.assertTrue(L.elig.max() > 0)


class ThreeFactorRule(unittest.TestCase):
    def test_dan_without_eligibility_no_update(self):
        es, cfg, L = learner()
        st = mp.PlasticState.zeros(es)
        for _ in range(10):
            L.step(np.zeros(4), {"appetitive": 1.0, "aversive": 1.0}, st)
        self.assertTrue(np.all(st.delta == 0))

    def test_paired_depresses_only_own_compartment(self):
        es, cfg, L = learner()
        st = mp.PlasticState.zeros(es)
        L.step(np.array([1.0, 1.0, 0, 0]), {"appetitive": 0.5, "aversive": 0.0}, st)
        app = es.compartment == "appetitive"
        self.assertTrue(np.all(st.delta[app][:2] < 0))
        self.assertTrue(np.all(st.delta[app][2:] == 0))          # inactive KCs untouched
        self.assertTrue(np.all(st.delta[~app] == 0))             # wrong compartment untouched

    def test_cross_compartment_control(self):
        es, cfg, L = learner(cross_compartment=True)
        st = mp.PlasticState.zeros(es)
        L.step(np.ones(4), {"appetitive": 0.5, "aversive": 0.0}, st)
        self.assertTrue(np.all(st.delta < 0))

    def test_bounds(self):
        es, cfg, L = learner(eta=100.0)
        st = mp.PlasticState.zeros(es)
        for _ in range(50):
            L.step(np.ones(4), {"appetitive": 1.0, "aversive": 1.0}, st)
        self.assertTrue(np.allclose(st.delta, -es.w0))
        self.assertTrue(np.all(st.effective(es) >= 0))
        es2, cfg2, L2 = learner(eta=100.0, sign=1.0, k_max=2.0)
        st2 = mp.PlasticState.zeros(es2)
        for _ in range(50):
            L2.step(np.ones(4), {"appetitive": 1.0, "aversive": 1.0}, st2)
        self.assertTrue(np.allclose(st2.effective(es2), 2 * es2.w0))

    def test_plasticity_off(self):
        es, cfg, L = learner(eta=0.0)
        st = mp.PlasticState.zeros(es)
        L.step(np.ones(4), {"appetitive": 1.0, "aversive": 1.0}, st)
        self.assertTrue(np.all(st.delta == 0))

    def test_gate_threshold(self):
        es, cfg, L = learner()
        D = L.dan_signal({"appetitive": 1.0, "aversive": 0.3}, {"appetitive": 1.0, "aversive": 0.1})
        self.assertEqual(D["appetitive"], 0.0); self.assertAlmostEqual(D["aversive"], 0.2)

    def test_shuffled_eligibility_changes_pattern(self):
        es, cfg, L = learner(shuffle_eligibility_seed=3)
        st = mp.PlasticState.zeros(es)
        L.step(np.array([1.0, 0, 0, 0]), {"appetitive": 0.5}, st)
        app = es.compartment == "appetitive"
        self.assertEqual(int((st.delta[app] != 0).sum()), 1)
        self.assertNotEqual(np.flatnonzero(st.delta[app])[0], 0)

    def test_deterministic(self):
        runs = []
        for _ in range(2):
            es, cfg, L = learner()
            st = mp.PlasticState.zeros(es); rng = np.random.default_rng(0)
            for _ in range(30):
                L.step((rng.random(4) < 0.3).astype(float), {"appetitive": float(rng.random()), "aversive": 0.2}, st)
            runs.append(st.delta.copy())
        np.testing.assert_array_equal(*runs)
        self.assertTrue(np.all(np.isfinite(runs[0])))


class StateSerialisation(unittest.TestCase):
    def test_zero_init_roundtrip_and_hash_check(self):
        es = toy_edges()
        st = mp.PlasticState.zeros(es)
        self.assertTrue(np.all(st.delta == 0))
        st.delta[:] = -es.w0 / 2
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            st.save(p, mp.PlasticityConfig(eta=0.1), dict(note="test"))
            back = mp.PlasticState.load(p, es)
            np.testing.assert_array_equal(back.delta, st.delta)
            bad = mp.EdgeSet(es.pre_idx, es.post_idx, es.pre_body, es.post_body, es.w0, es.compartment, "other")
            with self.assertRaises(ValueError):
                mp.PlasticState.load(p, bad)

    def test_learned_state_independent_of_neural_reset(self):
        # PlasticState holds no neural state; a brain reset cannot touch it (GPU version below)
        es = toy_edges(); st = mp.PlasticState.zeros(es); st.delta[:] = -es.w0
        before = st.delta.copy()
        _, _, L = learner(); L.reset_traces()
        np.testing.assert_array_equal(st.delta, before)


@unittest.skipUnless(GPU, "set FLYMON_GPU_TESTS=1")
class PlasticBrainGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flybrain import FlyBrain
        from scipy import sparse
        from run_visual_experiment import DATA as D
        cls.b = FlyBrain(data=D, device="cuda")
        m = np.load(D / "brain.npz")
        cls.W = sparse.load_npz(D / "weights.npz").tocsr()
        cls.es = mp.audited_edge_set(m["cell_type"], m["ids"], cls.W)
        cls.pb = mp.PlasticBrain(cls.b, cls.es)
        cls.kc = np.flatnonzero(m["cell_type"] == "KCg-d")

    def simulate(self, delta, seed=5, n=100):
        self.b.reset(seed=seed)
        return [self.pb.step(self.pb.prepare([(self.kc, 0.3)]), delta).copy() for _ in range(n)]

    def test_zero_delta_bit_identical_to_flybrain(self):
        from flymon.fast_io import device_pairs
        self.b.reset(seed=5)
        a = [self.b.step(inject=device_pairs([(self.kc, 0.3)], self.b)).copy() for _ in range(100)]
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, self.simulate(np.zeros(self.es.n)))))

    def test_frozen_matrix_unchanged_and_weight_reset_restores(self):
        import hashlib
        h0 = hashlib.sha256(self.b._W.data.get().tobytes()).hexdigest()
        base = self.simulate(np.zeros(self.es.n)); learned = self.simulate(-self.es.w0)
        self.assertEqual(hashlib.sha256(self.b._W.data.get().tobytes()).hexdigest(), h0)
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(base, learned)))
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(base, self.simulate(np.zeros(self.es.n)))))

    def test_neural_reset_preserves_learned_delta(self):
        st = mp.PlasticState(self.es.sha256, -self.es.w0.copy())
        a = self.simulate(st.delta, seed=7); self.b.reset(seed=99)
        np.testing.assert_array_equal(st.delta, -self.es.w0)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, self.simulate(st.delta, seed=7))))


if __name__ == "__main__":
    unittest.main()
