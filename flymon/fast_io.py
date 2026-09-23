"""Engineering accelerations for the closed gameplay loop (Experiment 25).

Neither class changes model semantics:

* `FastColumnSampler` computes exactly what `column_motion.column_luminance` computes for one
  frame (sRGB -> linear Rec.709 luminance, bilinear sampling at every E5 receptor, mean per
  optic column) as one 256-entry lookup plus one sparse matrix-vector product. Differences
  come only from float32 summation order (tested <= 1e-6).
* `device_pairs` converts FlyBrain injection index arrays to device arrays once per decision
  instead of once per brain step. FlyBrain applies the same voltages to the same neurons; the
  spike output is bit-identical (tested on the GPU).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

_SRGB = np.arange(256, dtype=np.float32) / 255
LINEAR_LUT = np.where(_SRGB <= .04045, _SRGB / 12.92, ((_SRGB + .055) / 1.055) ** 2.4).astype(np.float32)
REC709 = np.asarray([.2126, .7152, .0722], np.float32)


class FastColumnSampler:
    def __init__(self, records, colindex, uv, shape=(144, 160)):
        h, w = shape
        uv = np.asarray(uv, np.float32)
        x = uv[:, 0] * (w - 1); y = uv[:, 1] * (h - 1)
        x0 = x.astype(int); y0 = y.astype(int)
        x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
        dx = x - x0; dy = y - y0
        col_of = np.array([colindex[(r.eye_side, int(r.h1), int(r.h2))] for r in records])
        n = len(colindex)
        counts = np.bincount(col_of, minlength=n).astype(np.float64)
        counts[counts == 0] = 1.0
        rows, cols, vals = [], [], []
        for (yy, xx, ww) in ((y0, x0, (1 - dx) * (1 - dy)), (y0, x1, dx * (1 - dy)),
                             (y1, x0, (1 - dx) * dy), (y1, x1, dx * dy)):
            rows.append(col_of); cols.append(yy * w + xx); vals.append(ww.astype(np.float64) / counts[col_of])
        self.matrix = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                        shape=(n, h * w))
        self.shape = shape

    def luminance(self, frame):
        """Per-column mean linear luminance of one uint8 RGBA frame (float64, n_columns)."""
        if frame.shape != self.shape + (4,) or frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 RGBA {self.shape + (4,)}")
        lin = LINEAR_LUT[frame[..., :3]] @ REC709
        return self.matrix @ lin.ravel().astype(np.float64)


def device_pairs(pairs, brain):
    """Move injection index arrays to the brain's device once (no-op on CPU brains)."""
    xp = brain.xp
    if xp is np:
        return pairs
    return [(xp.asarray(idx), amount) for idx, amount in pairs]


class VectorInjector:
    """One FlyBrain step with all injection groups applied as a single indexed add.

    Replicates `flybrain.FlyBrain.step` (flybrain 0.1.0, batch 1, no eye drive) operation by
    operation; injection groups are disjoint, so each neuron receives exactly the same float32
    addition as through `inject=` pairs. Spike output is bit-identical (tested on the GPU)."""

    def __init__(self, brain):
        if brain.batch != 1:
            raise ValueError("VectorInjector supports batch 1 only")
        self.brain = brain

    def prepare(self, pairs):
        xp = self.brain.xp
        if not pairs:
            return None
        idx = np.concatenate([np.asarray(i, np.int64) for i, _ in pairs])
        vals = np.concatenate([np.full(len(i), np.float32(a), np.float32) for i, a in pairs])
        return xp.asarray(idx), xp.asarray(vals)

    def step(self, prepared):
        b, xp = self.brain, self.brain.xp
        current = b.synaptic_input(b.fired) * b.gain
        b.v *= b.decay
        b.v += current + b.tonic
        b.v += (b.rng.random((b.n, 1)) < b.noise_hz * b.dt) * np.float32(b.noise_amp)
        if prepared is not None:
            idx, vals = prepared
            b.v[idx, 0] += vals
        if b.refractory_steps:
            b.v[(b.steps - b.last_spike) <= b.refractory_steps] = 0.0
        fired = xp.flatnonzero(b.v >= 1.0)
        b.v.ravel()[fired] = 0.0
        if b.refractory_steps:
            b.last_spike.ravel()[fired] = b.steps
        b.fired = fired
        b.steps += 1
        return fired if xp is np else fired.get()
