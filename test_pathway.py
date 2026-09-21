"""Experiment 15 tests: connectome traversal, subset resolution, leak-free eval.

CPU only; no GPU or FlyBrain simulation required. The connectivity functions run
on tiny handcrafted graphs, and the leak-free evaluation runs on synthetic data.
"""
import json
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from flymon.pathway import (row_normalized_magnitude, visual_influence, hop_distance,
                            input_contributor_count)
from flymon.generalization import split_plan, instance_blocks
from flymon.retina import load_mapping

ROOT = Path(__file__).resolve().parent
EXP5 = ROOT / "results" / "experiment-05-retinotopic"
EXP15 = ROOT / "results" / "experiment-15-biological-pathway"


def chain_graph():
    """0 -> 1 -> 2 -> 3 as W[post, pre] (current flows pre to post)."""
    n = 4
    rows, cols, data = [], [], []
    for pre, post in [(0, 1), (1, 2), (2, 3)]:
        rows.append(post); cols.append(pre); data.append(1.0)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


class TraversalDirectionTests(unittest.TestCase):
    def test_hop_distance_is_downstream(self):
        W = chain_graph()
        d = hop_distance(W, np.array([0]), max_hops=8)
        self.assertEqual(d[0], 0)
        self.assertEqual(d[1], 1)
        self.assertEqual(d[2], 2)
        self.assertEqual(d[3], 3)

    def test_hop_distance_does_not_flow_upstream(self):
        W = chain_graph()
        d = hop_distance(W, np.array([3]), max_hops=8)
        # node 3 is a sink; nothing else is reachable downstream from it
        self.assertEqual(d[3], 0)
        self.assertTrue(np.all(d[[0, 1, 2]] == 9))  # sentinel max_hops+1

    def test_visual_influence_decays_downstream(self):
        W = chain_graph()
        r = visual_influence(W, np.array([0]), alpha=0.85)
        # source highest, strictly decreasing along the chain
        self.assertGreater(r[0], r[1])
        self.assertGreater(r[1], r[2])
        self.assertGreater(r[2], r[3])
        self.assertGreater(r[3], 0.0)

    def test_row_normalized_rows_sum_to_one(self):
        W = sparse.csr_matrix(np.array([[0, 2.0, -1.0], [0, 0, 3.0], [0, 0, 0]]))
        A = row_normalized_magnitude(W)
        s = np.asarray(A.sum(axis=1)).ravel()
        self.assertAlmostEqual(s[0], 1.0, places=6)  # |2|+|-1| normalised
        self.assertAlmostEqual(s[1], 1.0, places=6)
        self.assertAlmostEqual(s[2], 0.0, places=6)  # no inputs

    def test_input_contributor_count(self):
        # posts 2,3 each receive from source nodes {0,1}
        rows = [2, 2, 3]; cols = [0, 1, 1]; data = [1.0, 1.0, 1.0]
        W = sparse.csr_matrix((data, (rows, cols)), shape=(4, 4))
        mask = np.zeros(4, bool); mask[[0, 1]] = True
        counts = input_contributor_count(W, mask, np.array([2, 3]))
        self.assertEqual(counts[0], 2)  # node 2 from {0,1}
        self.assertEqual(counts[1], 1)  # node 3 from {1}


class DeterminismTests(unittest.TestCase):
    def test_visual_influence_deterministic(self):
        W = chain_graph()
        a = visual_influence(W, np.array([0]))
        b = visual_influence(W, np.array([0]))
        np.testing.assert_array_equal(a, b)


class RetinaCompatibilityTests(unittest.TestCase):
    def test_mapping_loads_and_is_in_range(self):
        records, diag = load_mapping(EXP5 / "retina-mapping.json")
        self.assertGreater(len(records), 3000)
        self.assertEqual(diag["r1_6_mapped"], len(records))
        self.assertTrue(all(0.0 <= r.u <= 1.0 and 0.0 <= r.v <= 1.0 for r in records))
        self.assertTrue(all(r.eye_side in ("L", "R") for r in records))


class SubsetResolutionTests(unittest.TestCase):
    def test_audit_subsets_resolve_to_descending_neurons(self):
        if not (EXP15 / "connectivity-audit.json").exists():
            self.skipTest("connectivity audit not yet run")
        audit = json.loads((EXP15 / "connectivity-audit.json").read_text())
        dn_ids = {row["flywire_malecns_id"] for row in audit["dn_ranked"]}
        for name, ids in audit["selected_dn"].items():
            self.assertTrue(set(ids) <= dn_ids, name)      # all selected are DNs
            self.assertEqual(len(ids), len(set(ids)), name)  # no duplicates
        # top-K ordering is a strict prefix by rank
        order = [row["flywire_malecns_id"] for row in audit["dn_ranked"]]
        self.assertEqual(audit["selected_dn"]["bio_dn_top20"], order[:20])


class SplitIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.y = np.repeat(np.arange(5), 6)
        self.plan = split_plan(self.y, nseed=20)

    def test_no_instance_or_seed_leakage(self):
        for f in self.plan:
            self.assertFalse(set(f["train_i"]) & set(f["test_i"]))
            self.assertFalse(set(f["train_s"]) & set(f["test_s"]))
            for inner in f["inner"]:
                self.assertFalse(set(inner["train_i"]) & set(inner["test_i"]))
                self.assertFalse(set(inner["train_s"]) & set(inner["test_s"]))
                self.assertTrue(set(inner["train_i"] + inner["test_i"]) <= set(f["train_i"]))
                self.assertTrue(set(inner["train_s"] + inner["test_s"]) <= set(f["train_s"]))

    def test_every_instance_tested(self):
        tested = set()
        for f in self.plan:
            tested.update(f["test_i"])
        self.assertEqual(tested, set(range(len(self.y))))


if __name__ == "__main__":
    unittest.main()
