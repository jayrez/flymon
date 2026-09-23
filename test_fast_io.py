"""Generic gameplay-loop accelerations (imported from Experiment 25): equivalence tests."""
import os
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GPU = os.environ.get("FLYMON_GPU_TESTS") == "1"


class FastIO(unittest.TestCase):
    def test_fast_column_sampler_matches(self):
        from flymon import column_motion as cm
        from flymon.fast_io import FastColumnSampler
        from flymon.retina import load_mapping
        records, _ = load_mapping(ROOT / "results/experiment-05-retinotopic/retina-mapping.json")
        colindex = cm.column_index(records)
        uv = np.asarray([(r.u, r.v) for r in records], np.float32)
        fs = FastColumnSampler(records, colindex, uv)
        rng = np.random.default_rng(0)
        for _ in range(3):
            f = rng.integers(0, 256, (144, 160, 4), dtype=np.uint8); f[..., 3] = 255
            ref = cm.column_luminance(f[None], records, colindex, uv)[0]
            np.testing.assert_allclose(fs.luminance(f), ref, atol=1e-6)

    def test_device_pairs_noop_on_cpu(self):
        from flymon.fast_io import device_pairs
        class B: xp = np
        pairs = [(np.arange(3), 0.5)]
        self.assertIs(device_pairs(pairs, B()), pairs)

    @unittest.skipUnless(GPU, "set FLYMON_GPU_TESTS=1")
    def test_vector_injection_bit_identical(self):
        import run_generation_zero as G
        from flymon.fast_io import VectorInjector, device_pairs
        from flymon.frozen_t4 import T4Injection
        brain, _, _ = G.brain_setup()
        idx = np.arange(1000, 7000)
        inj = T4Injection(idx, 2.0)
        rng = np.random.default_rng(1)
        seqs = [inj.pairs(rng.random(len(idx)) * 0.4) for _ in range(5)]
        brain.reset(seed=3); a = [brain.step(inject=device_pairs(p, brain)).copy() for p in seqs for _ in range(10)]
        vi = VectorInjector(brain); brain.reset(seed=3)
        b = [vi.step(vi.prepare(p)).copy() for p in seqs for _ in range(10)]
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))


if __name__ == "__main__":
    unittest.main()
