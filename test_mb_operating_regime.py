"""Experiment 27 tests: operating-point configuration, refractory semantics, identity at stock."""
import json
import os
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiment-27-mb-operating-regime"
GPU = os.environ.get("FLYMON_GPU_TESTS") == "1"


class Config(unittest.TestCase):
    def test_stock_defaults_and_roundtrip(self):
        from run_mb_operating_regime import OperatingPoint
        op = OperatingPoint()
        self.assertEqual((op.refractory_ms, op.apl_gain, op.kc_gain, op.sensory_input), (0.0, 1.0, 1.0, True))
        op2 = OperatingPoint(refractory_ms=40.0)
        self.assertEqual(OperatingPoint(**json.loads(json.dumps(asdict(op2)))), op2)

    def test_least_invasive_ordering(self):
        from run_mb_operating_regime import OperatingPoint as O
        pts = [O(refractory_ms=40, kc_gain=0.5), O(refractory_ms=40, apl_gain=2.0), O(refractory_ms=60), O(refractory_ms=20)]
        self.assertEqual(sorted(pts, key=lambda o: o.magnitude())[0], O(refractory_ms=20))
        self.assertEqual(sorted(pts, key=lambda o: o.magnitude())[-1].kc_gain, 0.5)

    def test_grids_preregistered(self):
        import run_mb_operating_regime as R
        self.assertEqual(R.REFRACTORY_GRID_MS, (0, 20, 40, 60, 80, 100))
        self.assertEqual(R.CAL_SEEDS, tuple(range(2801, 2807))); self.assertEqual(R.CONF_SEEDS, tuple(range(2811, 2821)))
        self.assertFalse(set(R.CAL_SEEDS) & set(R.CONF_SEEDS)); self.assertFalse(set(R.CAL_SEEDS) & set(R.E26_SEEDS))

    def test_harness_default_is_e26_stock(self):
        import inspect
        import run_mb_plasticity as E26
        sig = inspect.signature(E26.Harness.__init__)
        self.assertEqual(sig.parameters["refractory"].default, 0.0)
        self.assertTrue(sig.parameters["sensory_input"].default)


@unittest.skipUnless((OUT / "frozen-operating-point.json").exists(), "operating point not frozen")
class Artifacts(unittest.TestCase):
    def test_confirmation_used_frozen_config(self):
        fz = json.loads((OUT / "frozen-operating-point.json").read_text())
        conf = json.loads((OUT / "confirmation-results.json").read_text())
        self.assertEqual(conf["result"]["operating_point"], fz["operating_point"])
        self.assertEqual(conf["result"]["seeds"], list(range(2811, 2821)))

    def test_plastic_set_and_t4_unchanged_in_every_run(self):
        for p in OUT.glob("*.json"):
            d = json.loads(p.read_text())
            for r in (d.get("results") or []) + [d.get("result")] + [d.get("stock_same_seeds")]:
                if isinstance(r, dict) and "edge_hash" in r:
                    self.assertTrue(r["edge_hash"].startswith("65dbc6264c04a2fa"))
                    self.assertTrue(r["t4_sha256"].startswith("87829806"))


@unittest.skipUnless(GPU, "set FLYMON_GPU_TESTS=1")
class GPU_(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flybrain import FlyBrain
        from scipy import sparse
        from run_visual_experiment import DATA
        from flymon import mb_plasticity as mp
        from run_mb_operating_regime import OperatingPoint, OperatingPointBrain
        cls.FB, cls.DATA, cls.mp, cls.OP, cls.OPB = FlyBrain, DATA, mp, OperatingPoint, OperatingPointBrain
        m = np.load(DATA / "brain.npz"); cls.ct = m["cell_type"].astype(str)
        cls.W = sparse.load_npz(DATA / "weights.npz").tocsr()
        cls.es = mp.audited_edge_set(m["cell_type"], m["ids"], cls.W)

    def brain(self, refractory=0.0):
        return self.FB(data=self.DATA, device="cuda", refractory=refractory)

    def test_refractory_off_bit_identical_to_stock(self):
        b = self.brain(); ob = self.OPB(b, self.es, self.OP(), self.ct, self.W)
        b.reset(seed=3); a = [b.step().copy() for _ in range(150)]
        b.reset(seed=3); c = [ob.step(None, None).copy() for _ in range(150)]
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, c)))

    def test_refractory_blocks_spikes_and_no_leak_across_reset(self):
        b = self.brain(0.040)                                   # 2 steps
        ob = self.OPB(b, self.es, self.OP(refractory_ms=40), self.ct, self.W)
        b.reset(seed=4); spikes = [ob.step(None, None).copy() for _ in range(200)]
        last = {}
        for t, f in enumerate(spikes):
            for i in f:
                if i in last:
                    self.assertGreaterEqual(t - last[i], 3)
                last[i] = t
        # reset clears refractory state and reproduces the same sequence
        b.reset(seed=4); again = [ob.step(None, None).copy() for _ in range(200)]
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(spikes, again)))
        self.assertTrue(int(b.last_spike.max()) >= 0)

    def test_gain_identity_and_scope(self):
        b = self.brain(); base = self.OPB(b, self.es, self.OP(), self.ct, self.W)
        g1 = self.OPB(b, self.es, self.OP(apl_gain=1.0, kc_gain=1.0), self.ct, self.W)
        b.reset(seed=6); a = [base.step(None, None).copy() for _ in range(100)]
        b.reset(seed=6); c = [g1.step(None, None).copy() for _ in range(100)]
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, c)))
        import hashlib
        h0 = hashlib.sha256(b._W.data.get().tobytes()).hexdigest()
        kg = self.OPB(b, self.es, self.OP(kc_gain=0.5), self.ct, self.W)
        b.reset(seed=6); [kg.step(None, None) for _ in range(20)]
        self.assertEqual(hashlib.sha256(b._W.data.get().tobytes()).hexdigest(), h0)


if __name__ == "__main__":
    unittest.main()
