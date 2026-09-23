"""Experiment 23 tests: anatomy-only geometry transforms, ablations, leakage, oracle-freedom."""
import json
import re
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from flymon import adapted_reference as ar
from flymon import spatial_veto as sv

ROOT = Path(__file__).resolve().parent
EXP23 = ROOT / "results" / "experiment-23-spatial-inhibitory-veto"
CHEM = {"Mi1": "exc", "Mi9": "inh", "Mi4": "inh"}


def lattice(n=9, spacing=3.0):
    """n x n square lattice of columns, one eye, pixel coordinates."""
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="xy"), -1).reshape(-1, 2) * spacing
    return g.astype(float), np.array(["L"] * len(g), dtype=object)


def toy_proj(xy):
    """Two neurons: Mi1 at the lattice centre, Mi4 one column to the right (+u)."""
    n = len(xy); side = int(np.sqrt(n)); centre = (side // 2) * side + side // 2
    mi1 = np.zeros((2, n)); mi4 = np.zeros((2, n))
    mi1[:, centre] = [0.6, 0.5]
    mi1[:, centre - side] = [0.2, 0.2]
    mi4[:, centre + 1] = [-0.3, -0.25]
    mi4[:, centre + 1 + side] = [-0.1, -0.05]
    return {"T4a": {"Mi1": sparse.csr_matrix(mi1), "Mi4": sparse.csr_matrix(mi4)}}


class Transforms(unittest.TestCase):
    def setUp(self):
        self.xy, self.sides = lattice()
        self.proj = toy_proj(self.xy)

    def offset(self, proj):
        return sv.offsets(proj["T4a"], self.xy)[2]

    def test_native_unchanged(self):
        new, _ = sv.transform(self.proj, self.xy, self.sides, "native", subtypes=("T4a",))
        self.assertIs(new["T4a"]["Mi4"], self.proj["T4a"]["Mi4"])
        self.assertIs(new["T4a"]["Mi1"], self.proj["T4a"]["Mi1"])

    def test_reversal_reverses_offset(self):
        before = self.offset(self.proj)
        new, _ = sv.transform(self.proj, self.xy, self.sides, "reversed", subtypes=("T4a",))
        after = self.offset(new)
        self.assertTrue(np.all(before[:, 0] > 0))
        self.assertTrue(np.all(after[:, 0] < 0))
        np.testing.assert_allclose(after, -before, atol=1.0)

    def test_colocation_reduces_offset(self):
        new, _ = sv.transform(self.proj, self.xy, self.sides, "colocated", subtypes=("T4a",))
        self.assertTrue(np.all(np.linalg.norm(self.offset(new), axis=1)
                               < np.linalg.norm(self.offset(self.proj), axis=1)))

    def test_weights_and_counts_preserved(self):
        for mode in ("reversed", "colocated"):
            new, _ = sv.transform(self.proj, self.xy, self.sides, mode, subtypes=("T4a",))
            a, b = new["T4a"]["Mi4"], self.proj["T4a"]["Mi4"]
            np.testing.assert_allclose(np.sort(a.data), np.sort(b.data))
            np.testing.assert_array_equal(np.diff(a.indptr), np.diff(b.indptr))
            np.testing.assert_allclose(a.sum(axis=1), b.sum(axis=1))
            self.assertIs(new["T4a"]["Mi1"], self.proj["T4a"]["Mi1"])

    def test_no_duplicate_columns_after_snap(self):
        new, _ = sv.transform(self.proj, self.xy, self.sides, "colocated", subtypes=("T4a",))
        m = new["T4a"]["Mi4"].tocsr()
        for i in range(m.shape[0]):
            idx = m.indices[m.indptr[i]:m.indptr[i + 1]]
            self.assertEqual(len(idx), len(set(idx)))

    def test_snap_stays_in_same_eye(self):
        xy = np.vstack([self.xy, self.xy + 1000.0])
        sides = np.array(["L"] * len(self.xy) + ["R"] * len(self.xy), dtype=object)
        p = {"T4a": {k: sparse.hstack([m, sparse.csr_matrix(m.shape)]).tocsr()
                     for k, m in toy_proj(self.xy)["T4a"].items()}}
        new, _ = sv.transform(p, xy, sides, "reversed", subtypes=("T4a",))
        self.assertTrue(np.all(sides[new["T4a"]["Mi4"].indices] == "L"))

    def test_sign_conversion_changes_sign_only(self):
        new = sv.sign_converted(self.proj, subtypes=("T4a",))
        a, b = new["T4a"]["Mi4"], self.proj["T4a"]["Mi4"]
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_allclose(a.data, -b.data)
        self.assertTrue(np.all(a.data > 0))
        np.testing.assert_allclose(self.offset(new), self.offset(self.proj))
        self.assertIs(new["T4a"]["Mi1"], self.proj["T4a"]["Mi1"])

    def test_axis_projection(self):
        d = np.array([[2.0, 0.0], [0.0, 3.0]])
        np.testing.assert_allclose(sv.axis_projection(d, ["right", "up"]), [2.0, -3.0])

    def test_deterministic(self):
        a, _ = sv.transform(self.proj, self.xy, self.sides, "reversed", subtypes=("T4a",))
        b, _ = sv.transform(self.proj, self.xy, self.sides, "reversed", subtypes=("T4a",))
        self.assertEqual((a["T4a"]["Mi4"] != b["T4a"]["Mi4"]).nnz, 0)


class Candidate(unittest.TestCase):
    def test_candidate_frozen(self):
        import run_spatial_veto_experiment as R
        c = R.candidate()
        self.assertEqual((c.reference, c.adapt, c.G, c.beta, c.r0_exc, c.r0_mi4, c.E_inh),
                         ("fixed", 40, 16.0, 2.0, 1.0, 0.0, -0.2))
        self.assertEqual(c.drop, ("Mi9",))
        self.assertEqual((c.tau_fast, c.tau_slow), (4.0, 12.0))

    def test_temporal_conditions(self):
        import run_spatial_veto_experiment as R
        geo = dict(native={}, reversed={}, colocated={}, sign={})
        conds = R.conditions(geo)
        c = conds["T1_equal_filters"][0]; self.assertEqual(c.tau_fast, c.tau_slow)
        c = conds["T2_no_filtering"][0]; self.assertEqual((c.tau_fast, c.tau_slow), (0.0, 0.0))
        for t in R.COMMON_TAUS:
            c = conds[f"T3_common_tau_{t:g}"][0]; self.assertEqual((c.tau_fast, c.tau_slow), (t, t))
        self.assertIn("Mi4", conds["S2_mi4_removed"][0].drop)
        self.assertIn("Mi1", conds["S3_mi1_removed"][0].drop)

    def test_no_filter_path_has_no_persistence(self):
        lum = np.concatenate([np.full((5, 2), 0.5), np.full((1, 2), 0.9), np.full((5, 2), 0.5)]).astype(np.float32)
        s = ar.cell_signals(lum, 0.0, 0.0, "fixed", 0.5, 0)
        self.assertEqual(float(s["Mi1"][6:].max()), 0.0)
        s4 = ar.cell_signals(lum, 4.0, 4.0, "fixed", 0.5, 0)
        self.assertGreater(float(s4["Mi1"][6:].max()), 0.0)

    def test_heldout_bank_is_fresh(self):
        import run_spatial_veto_experiment as R
        import run_motion_nonlinearity_experiment as E19
        self.assertFalse(set(R.HELDOUT_SPEEDS) & (set(E19.CAL_SPEEDS) | set(E19.HELDOUT_SPEEDS)))
        self.assertFalse(set(R.HELDOUT_WIDTHS) & set(E19.HELDOUT_WIDTHS + (18,)))
        self.assertFalse(set(R.HELDOUT_CONTRASTS) & set(E19.HELDOUT_CONTRASTS + (1.0,)))
        self.assertFalse(set(R.HELDOUT_OFFSETS) & set(E19.HELDOUT_OFFSETS))

    def test_no_per_subtype_parameters(self):
        import run_spatial_veto_experiment as R
        self.assertFalse(any(re.search(r"T[45][abcd]", k) for k in R.candidate().to_json()))

    def test_no_oracle_in_model_or_transforms(self):
        for f in ("flymon/spatial_veto.py", "flymon/adapted_reference.py"):
            src = (ROOT / f).read_text()
            self.assertNotIn("motion_nonlinearity", src)
            self.assertNotIn("OPPONENT", src)
            self.assertNotIn("apply_mechanism", src)

    def test_geometry_shuffle_permutes(self):
        import run_fixed_background_experiment as E22
        from flymon import column_motion as cm
        ctx = dict(colindex={i: i for i in range(6)},
                   proj={"T4a": {t: sparse.csr_matrix(np.arange(1, 7, dtype=float)[None, :]) for t in cm.ALL_PARTNERS}})
        p = E22.permuted_projections(ctx, 4, 3)
        self.assertTrue(any(not np.array_equal(q["T4a"]["Mi4"].toarray().ravel(), np.arange(1, 7)) for q in p))

    def test_finite_on_toy(self):
        xy, sides = lattice()
        proj = toy_proj(xy)["T4a"]
        rng = np.random.default_rng(0)
        lum = (0.5 + rng.normal(0, 0.1, (30, len(xy)))).astype(np.float32)
        cfg = ar.AdaptedConfig(adapt=40, drop=("Mi9",))
        sig = ar.cell_signals(lum, 4.0, 12.0, "fixed", 0.5, 40)
        Y, aux = ar.simulate(proj, sig, cfg, 0.2, 1.0, CHEM)
        self.assertTrue(np.isfinite(Y).all())
        for c in aux["classes"].values():
            self.assertGreaterEqual(c["gE"].min(), 0.0); self.assertGreaterEqual(c["gI"].min(), 0.0)
        Y2, aux2 = ar.simulate(sv.sign_converted({"T4a": proj}, subtypes=("T4a",))["T4a"], sig, cfg, 0.2, 1.0, CHEM)
        self.assertEqual(float(aux2["classes"]["Mi4"]["gI"].max()), 0.0)
        self.assertGreater(float(aux2["classes"]["Mi4"]["gE"].max()), 0.0)


@unittest.skipUnless((EXP23 / "heldout-results.json").exists(), "held-out not run")
class SavedArtifacts(unittest.TestCase):
    def test_saved_subtype_dsi_matches_per_neuron(self):
        h = json.loads((EXP23 / "heldout-results.json").read_text())
        z = np.load(EXP23 / "heldout-per-neuron.npz")
        for cond in ("native", "reversed", "colocated"):
            for s, v in h["conditions"][cond]["subtypes"].items():
                self.assertAlmostEqual(v["dsi"], float(z[f"{cond}_{s}_dsi"].mean()), places=10)

    def test_candidate_matches_saved(self):
        import run_spatial_veto_experiment as R
        h = json.loads((EXP23 / "heldout-results.json").read_text())
        self.assertEqual(h["candidate"], R.candidate().to_json())

    def test_not_rescaled_oracle(self):
        z = np.load(EXP23 / "heldout-per-neuron.npz")
        for s in ("T4a", "T4b", "T4c", "T4d"):
            a, b = z[f"native_{s}_dsi"], z[f"oracle_native_{s}_dsi"]
            self.assertLess(abs(np.corrcoef(a, b)[0, 1]), 0.999)

    def test_geometry_preserved_weights(self):
        g = json.loads((EXP23 / "geometry-metrics.json").read_text())
        for v in g["subtypes"].values():
            self.assertTrue(v["weight_totals_preserved"]); self.assertTrue(v["input_counts_preserved"])


if __name__ == "__main__":
    unittest.main()
