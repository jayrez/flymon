"""Column-resolved graded optic-lobe model and parametric motion stimuli (Exp 18).

Offline and opt-in: stock `flybrain.FlyBrain` and the Experiment-17 adapter are
untouched. One model time step is one rendered stimulus frame. The input is the
unchanged Experiment-5 retina sampling, pooled per optic column (neural
superposition), and the readout is a weighted sum over the *actual* MaleCNS synapses
of each T4/T5 neuron, with each presynaptic cell class given a globally specified
polarity and temporal filter. No per-neuron or per-subtype parameter is fitted, and
no direction label enters the model.

All models share a final output rectification, because a firing rate cannot be
negative; M1 is therefore a linear-nonlinear model and is described as such.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from flymon import ethology
from flymon.optic_columns import FAST_ARM, SLOW_ARM

H, W = ethology.H, ethology.W

# Cell-class polarity (literature): Mi1/Mi4 are ON, Mi9 and the Tm OFF-pathway cells
# respond to luminance decrements. Never set from results.
POLARITY = {"Mi1": +1.0, "Mi4": +1.0, "Mi9": -1.0,
            "Tm1": -1.0, "Tm2": -1.0, "Tm9": -1.0}
ALL_PARTNERS = tuple(sorted(set(sum(FAST_ARM.values(), ()) + sum(SLOW_ARM.values(), ()))))


def arm_of(cell_type):
    for fam in ("T4", "T5"):
        if cell_type in FAST_ARM[fam]:
            return "fast"
        if cell_type in SLOW_ARM[fam]:
            return "slow"
    return None


# ---------------------------------------------------------------- stimuli
def parametric_bar(direction, polarity, frames=60, speed=1.0, bar_px=18,
                   contrast=1.0, start_offset=0.0):
    """Bar of constant area sweeping at `speed` px/frame, mean luminance ~0.5.

    contrast 1.0 reproduces the Experiment-16 full-contrast convention."""
    lo, hi = 0.5 - contrast / 2.0, 0.5 + contrast / 2.0
    bg, fg = (lo, hi) if polarity == "ON" else (hi, lo)
    horizontal = direction in ("left", "right")
    span = W if horizontal else H
    travel = speed * (frames - 1)
    base = (span - travel - bar_px) / 2.0 + bar_px / 2.0 + start_offset
    base = float(np.clip(base, bar_px / 2.0, span - bar_px / 2.0 - travel))
    centres = base + speed * np.arange(frames)
    if direction in ("left", "up"):
        centres = centres[::-1]
    out = []
    for c in centres:
        g = np.full((H, W), bg, np.float32)
        a = int(round(c - bar_px / 2.0)); b = a + bar_px
        a = max(a, 0); b = min(b, span)
        if horizontal:
            g[:, a:b] = fg
        else:
            g[a:b, :] = fg
        out.append(g)
    return np.stack(out)


def parametric_grating(direction, frames=60, speed=1.0, period=32.0, phase=0.0,
                       contrast=1.0):
    """Full-field drifting sinusoid; `speed` in px/frame, `phase` in radians."""
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    sign = 1.0 if direction == "right" else -1.0
    out = []
    for i in range(frames):
        shift = sign * speed * i
        g = 0.5 + 0.5 * contrast * np.sin(2 * np.pi * (xx - shift) / period + phase)
        out.append(g.astype(np.float32))
    return np.stack(out)


def to_rgba(seq):
    return ethology._rgba(seq)


def temporal_variant(seq, variant):
    return ethology.temporal_variant(seq, variant)


# ---------------------------------------------------------------- columns
def column_index(records):
    """Map (side, h1, h2) -> column id for every retina-mapped optic column."""
    keys = sorted({(r.eye_side, int(r.h1), int(r.h2)) for r in records})
    return {k: i for i, k in enumerate(keys)}


def column_luminance(seq_rgba, records, colindex, uv=None):
    """(frames, n_columns) mean luminance per optic column."""
    from flymon.retina import bilinear_sample, linear_luminance
    if uv is None:
        uv = np.asarray([(r.u, r.v) for r in records], np.float32)
    samples = np.stack([bilinear_sample(linear_luminance(f), uv) for f in seq_rgba])
    n = len(colindex)
    sums = np.zeros((samples.shape[0], n), np.float64)
    counts = np.zeros(n)
    col_of = np.array([colindex[(r.eye_side, int(r.h1), int(r.h2))] for r in records])
    np.add.at(sums.T, col_of, samples.T)
    np.add.at(counts, col_of, 1.0)
    counts[counts == 0] = 1.0
    return sums / counts


# ---------------------------------------------------------------- filters
def low_pass(x, tau):
    """First-order low pass along time (axis 0). tau in frames; tau<=0 is identity."""
    if tau is None or tau <= 0:
        return x
    alpha = float(np.exp(-1.0 / tau))
    out = np.empty_like(x)
    acc = x[0].copy()
    out[0] = acc
    for t in range(1, x.shape[0]):
        acc = alpha * acc + (1.0 - alpha) * x[t]
        out[t] = acc
    return out


def contrast_signal(lum):
    """Subtract each column's temporal mean so a static field contributes nothing."""
    return lum - lum.mean(axis=0, keepdims=True)


@dataclass(frozen=True)
class ModelConfig:
    name: str = "M1"
    tau_fast: float = 2.0
    tau_slow: float = 5.0
    rectify_inputs: bool = False     # M2+
    output_exponent: float = 1.0     # M3
    temporal: bool = True            # M0 sets this False
    correlator: bool = False         # M_HR control only

    def to_json(self):
        return asdict(self)


def cell_signals(lum, cfg):
    """Per-partner-cell-class signal over time: (n_types, frames, n_columns)."""
    base = contrast_signal(lum)
    signals = {}
    for ctype in ALL_PARTNERS:
        arm = arm_of(ctype)
        tau = 0.0 if not cfg.temporal else (cfg.tau_fast if arm == "fast" else cfg.tau_slow)
        s = low_pass(POLARITY[ctype] * base, tau)
        if cfg.rectify_inputs:
            s = np.maximum(s, 0.0)
        signals[ctype] = s
    return signals


# ---------------------------------------------------------------- readout
def build_projections(W_sparse, cell_type, cart_hex, side, subtype_records, colindex):
    """Sparse (n_neurons x n_columns) weight matrices per subtype and partner type."""
    from scipy import sparse
    ct = cell_type.astype(str)
    sd = side.astype(str)
    proj = {}
    for sub, entry in subtype_records.items():
        recs = entry["records"]
        if not recs:
            continue
        rows = {t: [] for t in ALL_PARTNERS}
        cols = {t: [] for t in ALL_PARTNERS}
        vals = {t: [] for t in ALL_PARTNERS}
        for n, rec in enumerate(recs):
            row = W_sparse[rec.brain_index].tocoo()
            for c, v in zip(row.col, row.data):
                t = ct[c]
                if t not in ALL_PARTNERS:     # only the preregistered arm cell classes
                    continue
                if not np.isfinite(cart_hex[c, 0]) or not np.isfinite(cart_hex[c, 1]):
                    continue          # partner has no annotated optic column
                key = (sd[c], int(cart_hex[c, 0]), int(cart_hex[c, 1]))
                cid = colindex.get(key)
                if cid is None:
                    continue
                rows[t].append(n); cols[t].append(cid); vals[t].append(float(v))
        proj[sub] = {t: sparse.csr_matrix((vals[t], (rows[t], cols[t])),
                                          shape=(len(recs), len(colindex)))
                     for t in ALL_PARTNERS if rows[t]}
    return proj


def responses(proj, signals, cfg):
    """Mean rectified response per neuron for each subtype: {subtype: (n_neurons,)}."""
    out = {}
    for sub, mats in proj.items():
        total = None
        for t, m in mats.items():
            contrib = m @ signals[t].T          # (n_neurons, frames)
            total = contrib if total is None else total + contrib
        if total is None:
            continue
        rate = np.maximum(total, 0.0)           # a firing rate cannot be negative
        if cfg.output_exponent != 1.0:
            rate = rate ** cfg.output_exponent
        out[sub] = rate.mean(axis=1)
    return out


def hr_responses(proj, lum, cfg):
    """Explicit multiplicative correlator control (M_HR), separate architecture.

    Correlates each neuron's fast-arm drive with its delayed slow-arm drive."""
    base = contrast_signal(lum)
    out = {}

    def arm_drive(mats, types):
        total = None
        for t in types:
            if t not in mats:
                continue
            c = mats[t] @ (POLARITY[t] * base).T
            total = c if total is None else total + c
        return total

    for sub, mats in proj.items():
        fam = sub[:2]
        fast = arm_drive(mats, FAST_ARM[fam])
        slow = arm_drive(mats, SLOW_ARM[fam])
        if fast is None or slow is None:
            continue
        slow_d = low_pass(slow.T, cfg.tau_slow).T
        prod = fast[:, 1:] * slow_d[:, :-1] - slow_d[:, 1:] * fast[:, :-1]
        out[sub] = np.maximum(prod, 0.0).mean(axis=1)
    return out


def dsi(pref, null, eps=1e-9):
    return (pref - null) / (pref + null + eps)
