"""Experiment 28 tests: joint grid, fixed DAN assay, per-population criteria, selection, isolation."""
import json
import os
import subprocess
import unittest
from pathlib import Path

import numpy as np

import run_mb_joint_operating_point as J

ROOT = Path(__file__).resolve().parent
OUT = J.OUT
GPU = os.environ.get("FLYMON_GPU_TESTS") == "1"


def synth(all_pass=True, steps=1, apl=5.0, **over):
    c = dict(R1=True, R2=True, R3=True, R4_PAM=True, R4_PPL1=True, R5_MBON01=True, R5_MBON11=True, R6=True)
    c.update(over)
    c["R4"] = c["R4_PAM"] and c["R4_PPL1"]; c["R5"] = c["R5_MBON01"] and c["R5_MBON11"]
    c["all_pass"] = all(c[k] for k in ("R1", "R2", "R3", "R4", "R5", "R6")) if all_pass else False
    return dict(key=f"s{steps}a{apl}", criteria=c, refractory_steps=steps,
                operating_point=dict(refractory_ms=steps * 20.0, apl_gain=apl, kc_gain=1.0, sensory_input=True))


class Provenance(unittest.TestCase):
    def test_base_contains_e27_merge(self):
        r = subprocess.run(["git", "merge-base", "--is-ancestor", "b36e30d08039ff1269eb8f2219687c1697ee74f1", "HEAD"], cwd=ROOT)
        self.assertEqual(r.returncode, 0)
        r = subprocess.run(["git", "merge-base", "--is-ancestor", "f43b6180410ee46c3fc852ad5806af6abe0991df", "HEAD"], cwd=ROOT)
        self.assertEqual(r.returncode, 0)


class Grid(unittest.TestCase):
    def test_seeds_disjoint_and_fresh(self):
        self.assertEqual(J.CAL_SEEDS, tuple(range(2901, 2907))); self.assertEqual(J.CONF_SEEDS, tuple(range(2911, 2921)))
        self.assertFalse(set(J.CAL_SEEDS) & set(J.CONF_SEEDS))
        import run_mb_operating_regime as E27
        self.assertFalse((set(J.CAL_SEEDS) | set(J.CONF_SEEDS)) & (set(E27.CAL_SEEDS) | set(E27.CONF_SEEDS) | set(E27.E26_SEEDS)))

    def test_refractory_mapping_and_uniqueness(self):
        self.assertEqual([J.refractory_steps(r) for r in (20, 30, 40, 50, 60)], [1, 2, 2, 2, 3])
        g = J.grid()
        self.assertEqual(len(g), 25)
        self.assertEqual(len({p.key() for p in g}), 25)
        self.assertEqual(len({(J.refractory_steps(p.refractory_ms), p.apl_gain) for p in g}), 25)
        self.assertEqual(sorted({J.refractory_steps(p.refractory_ms) for p in g}), [1, 2, 3, 4, 5])
        self.assertTrue(all(p.kc_gain == 1.0 and p.sensory_input for p in g))

    def test_fixed_dan_amplitude_no_adaptive_rule(self):
        self.assertEqual(J.DAN_AMPLITUDE, 0.3)
        import inspect
        self.assertEqual(inspect.signature(J.dan_assay_fixed).parameters["amplitude"].default, 0.3)
        src = inspect.getsource(J.score) + inspect.getsource(J.evaluate_point)
        self.assertNotIn("rule_amplitude", src); self.assertNotIn("E27.dan_assay(", src)


class Scoring(unittest.TestCase):
    def inputs(self, pam=12.0, ppl=12.0, m01=(100, 85), m11=(100, 85), r3=True):
        pops = {p: dict(ceiling_normalised_duty=0.2, rate_hz=5.0) for p in ("KCg-d", "MBON01", "MBON11", "PAM01", "PPL101")}
        base = dict(populations=pops, finite=True, whole_brain_fraction_at_ceiling=0.0, whole_brain_fraction_silent=0.05)
        dan = {g: dict(delta_pp=v, positive_seeds=6, passes=bool(v >= 10)) for g, v in (("PAM01", pam), ("PPL101", ppl))}
        lev = dict(per_population={m: dict(baseline=[b] * 6, zeroed=[z] * 6) for m, (b, z) in (("MBON01", m01), ("MBON11", m11))})
        return base, dict(R3=r3), dan, lev

    def test_all_pass(self):
        c, _ = J.score(*self.inputs()); self.assertTrue(c["all_pass"])

    def test_r4_per_dan(self):
        c, _ = J.score(*self.inputs(ppl=5.0))
        self.assertTrue(c["R4_PAM"]); self.assertFalse(c["R4_PPL1"]); self.assertFalse(c["R4"]); self.assertFalse(c["all_pass"])

    def test_r5_per_mbon_not_combined(self):
        # MBON01 -20 %, MBON11 -5 %: combined would be -12.5 % but R5 must fail
        c, r5 = J.score(*self.inputs(m01=(100, 80), m11=(100, 95)))
        self.assertTrue(c["R5_MBON01"]); self.assertFalse(c["R5_MBON11"]); self.assertFalse(c["R5"])
        self.assertAlmostEqual(r5["MBON01"]["zeroed_rel"], -0.2)


class Selection(unittest.TestCase):
    def test_all_pass_beats_nonpassing(self):
        rows = [synth(all_pass=False, steps=1, apl=3.0, R5_MBON11=False), synth(steps=5, apl=8.0)]
        passing, ranked = J.select(rows)
        self.assertEqual(ranked[0]["refractory_steps"], 5); self.assertEqual(len(passing), 1)

    def test_lower_refractory_then_lower_apl(self):
        rows = [synth(steps=3, apl=3.0), synth(steps=2, apl=8.0), synth(steps=2, apl=4.0)]
        _, ranked = J.select(rows)
        self.assertEqual((ranked[0]["refractory_steps"], ranked[0]["operating_point"]["apl_gain"]), (2, 4.0))


@unittest.skipUnless((OUT / "grid-results.json").exists(), "grid not run")
class Artifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = json.loads((OUT / "grid-results.json").read_text())

    def test_grid_complete_exactly_once_calibration_only(self):
        keys = [r["key"] for r in self.g["results"]]
        self.assertEqual(sorted(keys), sorted(p.key() for p in J.grid()))
        self.assertEqual(len(keys), len(set(keys)))
        for r in self.g["results"]:
            self.assertEqual(r["seeds"], list(J.CAL_SEEDS))
        self.assertEqual(self.g["seeds"], list(J.CAL_SEEDS))
        txt = (OUT / "grid-results.json").read_text() + (OUT / "candidate-ranking.json").read_text()
        for s in J.CONF_SEEDS:
            self.assertNotIn(f'"seeds": [{s}', txt)
        for r in self.g["results"]:
            self.assertFalse(set(r["seeds"]) & set(J.CONF_SEEDS))

    def test_provenance_consistent(self):
        prov = {json.dumps({k: r["provenance"][k] for k in ("edge_hash", "t4_sha256", "base_w_sha256")}) for r in self.g["results"]}
        self.assertEqual(len(prov), 1)
        p = self.g["results"][0]["provenance"]
        self.assertTrue(p["edge_hash"].startswith("65dbc6264c04a2fa")); self.assertTrue(p["t4_sha256"].startswith("87829806"))

    def test_frozen_semantics(self):
        fz = json.loads((OUT / "frozen-operating-point.json").read_text())
        if fz["operating_point"] is None:
            self.assertEqual(fz["status"], "NO PASSING OPERATING POINT")
            self.assertTrue((OUT / "best-nonpassing-candidate.json").exists())
        else:
            self.assertIn(fz["key"], [p.key() for p in J.grid()])          # exact grid point, no interpolation
            self.assertFalse((OUT / "best-nonpassing-candidate.json").exists())
            row = next(r for r in self.g["results"] if r["key"] == fz["key"])
            self.assertTrue(row["criteria"]["all_pass"])

    def test_confirmation_uses_frozen_or_labelled_descriptive(self):
        p = OUT / "confirmation-results.json"
        if not p.exists():
            self.skipTest("confirmation not run")
        c = json.loads(p.read_text()); fz = json.loads((OUT / "frozen-operating-point.json").read_text())
        self.assertEqual(c["result"]["seeds"], list(J.CONF_SEEDS))
        if fz["operating_point"] is None:
            self.assertTrue(c["descriptive_confirmation_only"])
        else:
            self.assertEqual(c["operating_point"], fz["operating_point"]); self.assertFalse(c["descriptive_confirmation_only"])


@unittest.skipUnless(GPU, "set FLYMON_GPU_TESTS=1")
class GPU_(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flybrain import FlyBrain
        from scipy import sparse
        from run_visual_experiment import DATA
        from flymon import mb_plasticity as mp
        from run_mb_operating_regime import OperatingPoint, OperatingPointBrain
        cls.FB, cls.DATA, cls.OP, cls.OPB = FlyBrain, DATA, OperatingPoint, OperatingPointBrain
        m = np.load(DATA / "brain.npz"); cls.ct = m["cell_type"].astype(str)
        cls.W = sparse.load_npz(DATA / "weights.npz").tocsr()
        cls.es = mp.audited_edge_set(m["cell_type"], m["ids"], cls.W)
        cls.apl = np.flatnonzero(cls.ct == "APL")

    def one_step_v(self, gain, fired_idx):
        """One noiseless step from v = 0 with a chosen set of neurons marked as having just fired."""
        import cupy as cp
        b = self.FB(data=self.DATA, device="cuda"); b.noise_hz = 0.0
        ob = self.OPB(b, self.es, self.OP(apl_gain=gain), self.ct, self.W)
        b.reset(seed=1); b.fired = cp.asarray(np.asarray(fired_idx, np.int64))
        import hashlib
        h0 = hashlib.sha256(b._W.data.get().tobytes()).hexdigest()
        ob.step(None, None)
        self.assertEqual(hashlib.sha256(b._W.data.get().tobytes()).hexdigest(), h0)   # _W never mutated
        return b.v.get()[:, 0].astype(np.float64), float(b.gain)

    def test_apl_gain_scope(self):
        a = int(self.apl[0]); other = int(np.flatnonzero(self.ct == "KCg-d")[0])
        v1, gain = self.one_step_v(1.0, [a]); v5, _ = self.one_step_v(5.0, [a])
        expected = 4.0 * gain * self.W[:, a].toarray().ravel()
        mask = (v1 < 1.0) & (v5 < 1.0) & (v1 > 0) & (v5 > 0)                      # exclude neurons reset by spiking
        self.assertGreater(np.count_nonzero(expected[mask]), 100)                   # APL targets are actually probed
        np.testing.assert_allclose((v5 - v1)[mask], expected[mask], atol=1e-5)
        np.testing.assert_array_equal((v5 - v1)[mask & (expected == 0)], 0.0)       # non-APL targets untouched
        n1, _ = self.one_step_v(1.0, [other]); n5, _ = self.one_step_v(5.0, [other])   # APL silent: identical
        np.testing.assert_array_equal(n1, n5)

    def test_joint_candidate_deterministic_and_refractory(self):
        import hashlib
        b = self.FB(data=self.DATA, device="cuda", refractory=0.060)
        ob = self.OPB(b, self.es, self.OP(refractory_ms=60, apl_gain=5.0), self.ct, self.W)
        h0 = hashlib.sha256(b._W.data.get().tobytes()).hexdigest()
        b.reset(seed=2901); a = [ob.step(None, None).copy() for _ in range(200)]
        b.reset(seed=2901); c = [ob.step(None, None).copy() for _ in range(200)]
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, c)))
        self.assertGreaterEqual(int(b.last_spike.max()), 0)
        b.reset(seed=2901)                                                             # reset hygiene
        self.assertTrue(bool((b.last_spike == -10**6).all())); self.assertEqual(b.steps, 0); self.assertEqual(b.fired.size, 0)
        last = {}
        for t, f in enumerate(a):
            for i in f:
                if i in last:
                    self.assertGreaterEqual(t - last[i], 4)                     # 3 refractory steps
                last[i] = t
        self.assertEqual(hashlib.sha256(b._W.data.get().tobytes()).hexdigest(), h0)
        self.assertTrue(self.es.sha256.startswith("65dbc6264c04a2fa"))


if __name__ == "__main__":
    unittest.main()
