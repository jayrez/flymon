"""Experiment 29 tests: provenance, seeds, KC/VP audit, gate, frozen ranking, caps, isolation, GPU checks."""
import hashlib
import inspect
import json
import os
import subprocess
import unittest
from pathlib import Path

import numpy as np

import run_broader_mb_circuit as E29

ROOT = Path(__file__).resolve().parent
OUT = E29.OUT
GPU = os.environ.get("FLYMON_GPU_TESTS") == "1"
has = lambda n: (OUT / n).exists()
load = lambda n: json.loads((OUT / n).read_text())


class Provenance(unittest.TestCase):
    def test_main_ancestry_and_e28_merge(self):
        for c in ("50e3685665457ff434ff9d9bae9ac1fdd8a385f8", "a576fc6fefefd304c8a5f7abd208750e99757af6",
                  "b36e30d08039ff1269eb8f2219687c1697ee74f1"):
            self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", c, "HEAD"], cwd=ROOT).returncode, 0)


class Config(unittest.TestCase):
    def test_seed_blocks_disjoint_and_unused(self):
        import run_mb_operating_regime as E27, run_mb_joint_operating_point as E28
        blocks = [set(E29.VISUAL_SEEDS), set(E29.LEVERAGE_SEEDS), set(E29.CONF_SEEDS)]
        self.assertEqual(E29.VISUAL_SEEDS, tuple(range(3001, 3007))); self.assertEqual(E29.CONF_SEEDS, tuple(range(3021, 3031)))
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(blocks[i] & blocks[j])
        prev = set(E27.E26_SEEDS) | set(E27.CAL_SEEDS) | set(E27.CONF_SEEDS) | set(E28.CAL_SEEDS) | set(E28.CONF_SEEDS)
        self.assertFalse(set().union(*blocks) & prev)

    def test_exactly_two_regimes_probe_60_8(self):
        self.assertEqual(list(E29.REGIMES), ["60/8", "80/3"])
        p = E29.REGIMES["60/8"]; q = E29.REGIMES["80/3"]
        self.assertEqual((p.refractory_ms, p.apl_gain, p.kc_gain, p.sensory_input), (60.0, 8.0, 1.0, True))
        self.assertEqual((q.refractory_ms, q.apl_gain, q.kc_gain), (80.0, 3.0, 1.0))

    def test_gate_constants_frozen(self):
        self.assertEqual((E29.MIN_CLASS_CELLS, E29.MIN_REL, E29.MIN_CONS_FRAC, E29.LEV_MAX_REL), (10, 0.05, 5 / 6, -0.10))
        self.assertEqual((E29.min_cons(6), E29.min_cons(10)), (5, 9))
        self.assertEqual((E29.MAX_DYNAMIC, E29.MAX_CONFIRM), (10, 3))

    def test_no_learning_in_runner(self):
        src = inspect.getsource(E29)
        self.assertNotIn("Learner(", src); self.assertNotIn("learning=True", src)


class Gate(unittest.TestCase):
    def c(self, rel, cons, sign=1):
        return dict(rel_diff=sign * rel, mean_diff=sign * rel * 100, consistent_seeds=cons)

    def test_gate_exact(self):
        self.assertTrue(E29.gate(10, self.c(.05, 5), self.c(.05, 6), 6))
        self.assertFalse(E29.gate(9, self.c(.5, 6), self.c(.5, 6), 6))            # class size
        self.assertFalse(E29.gate(50, self.c(.049, 6), self.c(.5, 6), 6))         # no-visual magnitude
        self.assertFalse(E29.gate(50, self.c(.5, 4), self.c(.5, 6), 6))           # consistency
        self.assertFalse(E29.gate(50, self.c(.5, 6), self.c(.049, 6), 6))         # gray magnitude
        self.assertFalse(E29.gate(50, self.c(.5, 6), self.c(.5, 6, -1), 6))       # opposite sign
        self.assertFalse(E29.gate(50, self.c(.5, 8), self.c(.5, 10), 10))         # 8/10 < 9/10 at confirmation


class StaticAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not has("kc-population-audit.json"):
            raise unittest.SkipTest("static audit not run")
        cls.kc = load("kc-population-audit.json"); cls.vp = load("visual-projection-set.json")
        cls.conn = load("kc-mbon-connectivity.json"); cls.m = np.load(ROOT / "redfly-benchmark/data/brain.npz")

    def test_all_kc_types_and_counts(self):
        ct = self.m["cell_type"].astype(str)
        kc = ct[np.char.startswith(ct, "KC")]
        self.assertEqual(sorted(self.kc["types"]), sorted(set(kc)))
        self.assertEqual(self.kc["total_kc"], len(kc)); self.assertEqual(sum(v["n"] for v in self.kc["types"].values()), len(kc))
        ids = sorted(b for v in self.kc["types"].values() for b in v["body_ids"])
        self.assertEqual(hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(), self.kc["population_sha256"])

    def test_visual_projection_set_deterministic_and_frozen(self):
        sc = self.m["superclass"].astype(str)
        ids = sorted(int(b) for b in self.m["ids"][sc == "visual_projection"])
        self.assertEqual(ids, self.vp["body_ids"])
        self.assertEqual(hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(), self.vp["sha256"])
        self.assertTrue(self.vp["sha256"].startswith("3f31cf6de31b51a3"))

    def test_all_mbon_types_audited(self):
        ct = self.m["cell_type"].astype(str)
        self.assertEqual(sorted(self.conn["mbon_types"]), sorted(set(ct[np.char.startswith(ct, "MBON")])))
        self.assertIn("KCg-d->MBON01", self.conn["pairs"]); self.assertIn("KCg-d->MBON11", self.conn["pairs"])
        self.assertEqual(self.conn["pairs"]["KCg-d->MBON01"]["edges"] + self.conn["pairs"]["KCg-d->MBON11"]["edges"], 418)

    def test_connectivity_deterministic_from_W(self):
        from scipy import sparse
        W = sparse.load_npz(ROOT / "redfly-benchmark/data/weights.npz").tocsr()
        ct = self.m["cell_type"].astype(str)
        for key in ("KCg-d->MBON27", "KCab-p->MBON19", "KCg-d->MBON01"):
            k, mb = key.split("->")
            sub = W[np.flatnonzero(ct == mb)][:, np.flatnonzero(ct == k)]
            self.assertEqual(sub.nnz, self.conn["pairs"][key]["edges"])
            self.assertAlmostEqual(float(abs(sub).sum()), self.conn["pairs"][key]["abs_weight_sum"], places=9)

    def test_candidate_edges_exist_and_reference_hash(self):
        from scipy import sparse
        from flymon import mb_plasticity as mp
        W = sparse.load_npz(ROOT / "redfly-benchmark/data/weights.npz").tocsr()
        ct, ids = self.m["cell_type"].astype(str), self.m["ids"]
        es = E29.candidate_edges(ct, ids, W, "KCg-d", "MBON27")
        self.assertTrue(np.all(np.asarray(W[es.post_idx, es.pre_idx]).ravel() == es.w0.astype(np.float32)))
        a = E29.candidate_edges(ct, ids, W, "KCg-d", "MBON01"); b = E29.candidate_edges(ct, ids, W, "KCg-d", "MBON11")
        ref = mp.audited_edge_set(self.m["cell_type"], ids, W)
        self.assertEqual(a.n + b.n, ref.n)
        self.assertEqual(set(zip(a.pre_body, a.post_body)) | set(zip(b.pre_body, b.post_body)),
                         set(zip(ref.pre_body, ref.post_body)))
        self.assertTrue(ref.sha256.startswith("65dbc6264c04a2fa"))

    def test_dan_provenance_recorded(self):
        d = load("dan-compartment-audit.json")
        self.assertIn("never used for matching", d["rule"])
        p = d["pairs"]["KCg-d->MBON01"]
        self.assertEqual(p["primary_dan"], "PAM01")
        for r in p["dan_rows"]:
            for k in ("dan_to_mbon", "dan_to_kc", "kc_to_dan", "mbon_to_dan", "topology_match"):
                self.assertIn(k, r)


class Ranking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not has("candidate-ranking-freeze.json"):
            raise unittest.SkipTest("ranking not frozen")
        cls.fz = load("candidate-ranking-freeze.json"); cls.dl = load("dynamic-candidate-list.json")

    def test_freeze_hash(self):
        E29.check_freeze()
        self.assertFalse(self.fz["leverage_results_existed"])

    def test_ranking_frozen_before_dynamic_results(self):
        if has("dynamic-leverage-results.json"):
            self.assertLess(self.fz["frozen_at_utc"],
                            __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime(
                                (OUT / "dynamic-leverage-results.json").stat().st_mtime)))

    def test_ranking_reproducible_and_cap(self):
        rows, eligible = E29.build_ranking()
        self.assertEqual([r["key"] for r in rows], self.fz["ranked_keys"])
        self.assertEqual([r["key"] for r in eligible[:10]], self.fz["tested_keys"])
        self.assertLessEqual(len(self.dl["candidates"]), 10)
        self.assertEqual([c["key"] for c in self.dl["candidates"]], self.fz["tested_keys"])
        self.assertEqual({r["key"] for r in self.dl["references"]}, {"KCg-d->MBON01", "KCg-d->MBON11"})

    def test_dynamic_results_match_frozen_list(self):
        if not has("dynamic-leverage-results.json"):
            self.skipTest("no leverage yet")
        d = load("dynamic-leverage-results.json")
        cand = {f"{r['kc_type']}->{r['mbon_type']}" for r in d["results"] if r["role"] == "candidate"}
        self.assertEqual(cand, set(self.fz["tested_keys"]))
        self.assertEqual({r["regime"] for r in d["results"]}, {"60/8", "80/3"})
        self.assertTrue(all(r["w_unchanged"] for r in d["results"]))
        self.assertTrue(all(r["seeds"] == list(E29.LEVERAGE_SEEDS) for r in d["results"]))
        refs = {f"{r['kc_type']}->{r['mbon_type']}" for r in d["results"]}
        self.assertTrue({"KCg-d->MBON01", "KCg-d->MBON11"} <= refs)

    def test_confirmation_cap_and_isolation(self):
        if not has("confirmation-results.json"):
            self.skipTest("no confirmation yet")
        c = load("confirmation-results.json"); cc = load("candidate-circuits.json")
        self.assertLessEqual(len(c["chosen"]), 3)
        if not c["descriptive_confirmation_only"]:
            self.assertEqual(c["chosen"], cc["confirmation_set"])
        self.assertEqual(c["seeds"], list(E29.CONF_SEEDS))
        self.assertEqual(set(c["per_regime"]), {"60/8", "80/3"})
        v = load("kc-visual-response.json")
        self.assertEqual(v["seeds"], list(E29.VISUAL_SEEDS))
        s = load("operating-regime-sensitivity.json")
        self.assertEqual({r["regime"] for r in s["rows"]}, {"60/8", "80/3"})


@unittest.skipUnless(GPU, "set FLYMON_GPU_TESTS=1")
class GPU_(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = E29.Runner("60/8")

    def test_all_kc_trial_matches_e26_trial(self):
        R = self.R
        stim, _ = E29.E27.E26.make_stimuli()
        a = R.visual_trial(3001, stim["vertical"], True, decisions=6)
        R.h.brain.reset(seed=3001)
        # E26 visual_trial restricted to 6 decisions
        R.h.t4.reset(); rec = []; kc = np.zeros(len(R.h.pop["KC"]))
        R.h.brain.reset(seed=3001)
        for d in range(6):
            resp = [R.h.t4.update_from_luminance(R.h.sampler.luminance(f)) for f in stim["vertical"](d)]
            pairs = R.h.inj.pairs(np.mean(resp, axis=0))
            for _ in range(10):
                kc += R.h.step(pairs, None, rec)[1]
        np.testing.assert_array_equal(a[R.h.pop["KC"]], kc)

    def test_leverage_deterministic_no_w_mutation(self):
        R = self.R
        h0 = R.w_hash_now()
        x = R.leverage("KCg-d", "MBON27", (3011,))
        y = R.leverage("KCg-d", "MBON27", (3011,))
        self.assertEqual(x["baseline_mbon"], y["baseline_mbon"]); self.assertEqual(x["zeroed_mbon"], y["zeroed_mbon"])
        self.assertTrue(x["w_unchanged"]); self.assertEqual(R.w_hash_now(), h0)
        self.assertEqual(R.h.brain.refractory_steps, 3)
        R.h.brain.reset(seed=1)
        self.assertTrue(bool((R.h.brain.last_spike == -10**6).all()))


if __name__ == "__main__":
    unittest.main()
