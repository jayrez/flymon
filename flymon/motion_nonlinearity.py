"""Competing minimal readout mechanisms for the Experiment-18 T4/T5 geometry.

Every model shares the Experiment-18 front end (same optic-column geometry, same
column luminance, same cell-class polarity and temporal filters, same split) and
differs *only* in how the fast-arm and slow-arm drives are combined. Nothing here
fits per-neuron or per-subtype parameters, and no response ever influences geometry.

Measured sign asymmetry (see preregistration): T4's slow arm is inhibitory (Mi9, Mi4)
while T5's slow arm is excitatory (Tm9). N0/N1 keep Experiment-18's signed-weight sum
so they reproduce it exactly; N2-N4 operate on rectified arm *magnitudes* so a single
equation applies to both pathways with the interaction stated explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

from flymon import column_motion as cm
from flymon.optic_columns import FAST_ARM, SLOW_ARM


@dataclass(frozen=True)
class Mechanism:
    """A readout rule. `family` selects the equation; the rest are global parameters."""
    family: str = "N1_LEXT"
    tau_fast: float = 2.0
    tau_slow: float = 8.0
    alpha: float = 1.0        # N3 divisive strength
    beta: float = 1.0         # N4 subtractive weight
    gamma: float = 0.5        # N4 multiplicative weight
    gate: str = "identity"    # N2 gate: identity | saturating
    delta: int = 1            # frames, decomposition controls
    label: str = ""

    def key(self):
        base = f"{self.family}|tf{self.tau_fast:g}|ts{self.tau_slow:g}"
        if self.family == "N2_COINCIDENCE":
            base += f"|{self.gate}"
        if self.family == "N3_SHUNT":
            base += f"|a{self.alpha:g}"
        if self.family == "N4_SUBUNIT":
            base += f"|b{self.beta:g}|g{self.gamma:g}"
        return base

    def to_json(self):
        return asdict(self)


SELECTABLE = ("N0_E18", "N1_LEXT", "N2_COINCIDENCE", "N3_SHUNT", "N4_SUBUNIT")
CONTROL_ONLY = ("N5_HR_CONTROL", "P_ONLY", "OPPONENT_PRODUCT",
                "P_ONLY_SIGNED", "OPPONENT_SIGNED")
# Families evaluated on SIGNED arm drives (Experiment-18 convention).
SIGNED_FAMILIES = ("N0_E18", "N1_LEXT", "N5_HR_CONTROL",
                   "P_ONLY_SIGNED", "OPPONENT_SIGNED")
SIMPLICITY = {f: i for i, f in enumerate(SELECTABLE)}


def signals_for(lum, mech):
    """Cell-class filtered signals, reusing the Experiment-18 implementation."""
    cfg = cm.ModelConfig(name=mech.family, tau_fast=mech.tau_fast, tau_slow=mech.tau_slow,
                         rectify_inputs=False, output_exponent=1.0, temporal=True)
    return cm.cell_signals(lum, cfg)


def arm_drives(proj_sub, signals, family_of_subtype, magnitude=False):
    """(F, S) drives of shape (n_neurons, frames) for one subtype.

    magnitude=True uses |w| so the arm drive is an activation rather than a signed
    synaptic contribution; the interaction is then explicit in the model equation."""
    def drive(types):
        total = None
        for t in types:
            m = proj_sub.get(t)
            if m is None:
                continue
            if magnitude:
                m = m.copy()
                m.data = np.abs(m.data)
            c = m @ signals[t].T
            total = c if total is None else total + c
        return total
    fam = family_of_subtype
    F = drive(FAST_ARM[fam])
    S = drive(SLOW_ARM[fam])
    return F, S


def _shift(x, d):
    """Delay along the time axis by d frames (edge-padded)."""
    if d <= 0:
        return x
    out = np.empty_like(x)
    out[:, :d] = x[:, :1]
    out[:, d:] = x[:, :-d]
    return out


def apply_mechanism(F, S, mech):
    """Combine arm drives into a non-negative response trace (n_neurons, frames)."""
    fam = mech.family
    if fam in ("N0_E18", "N1_LEXT"):
        return np.maximum(F + S, 0.0)
    Fr = np.maximum(F, 0.0)
    Sr = np.maximum(S, 0.0)
    if fam == "N2_COINCIDENCE":
        g = Sr if mech.gate == "identity" else Sr / (1.0 + Sr)
        return np.maximum(Fr * g, 0.0)
    if fam == "N3_SHUNT":
        return Fr / (1.0 + mech.alpha * Sr)
    if fam == "N4_SUBUNIT":
        return np.maximum(Fr - mech.beta * Sr + mech.gamma * Fr * Sr, 0.0)
    if fam == "P_ONLY":
        return np.maximum(Fr * _shift(Sr, mech.delta), 0.0)
    if fam == "OPPONENT_PRODUCT":
        return np.maximum(Fr * _shift(Sr, mech.delta) - Sr * _shift(Fr, mech.delta), 0.0)
    if fam == "P_ONLY_SIGNED":       # amendment: same product on signed drives
        return np.maximum(F * _shift(S, mech.delta), 0.0)
    if fam == "OPPONENT_SIGNED":     # amendment: completes the 2x2 decomposition
        return np.maximum(F * _shift(S, mech.delta) - S * _shift(F, mech.delta), 0.0)
    if fam == "N5_HR_CONTROL":
        # Experiment-18 antisymmetric correlator on the unrectified signed drives.
        d = cm.low_pass(S.T, mech.tau_slow).T
        prod = F[:, 1:] * d[:, :-1] - d[:, 1:] * F[:, :-1]
        out = np.zeros_like(F)
        out[:, 1:] = np.maximum(prod, 0.0)
        return out
    raise ValueError(fam)


def interaction_ablation(mech):
    """The simpler parent model obtained by removing the interaction term."""
    fam = mech.family
    if fam == "N2_COINCIDENCE":      # drop the gate -> fast arm alone
        return Mechanism(family="N4_SUBUNIT", tau_fast=mech.tau_fast, tau_slow=mech.tau_slow,
                         beta=0.0, gamma=0.0, label="ablate_interaction")
    if fam == "N3_SHUNT":            # alpha -> 0 removes the division
        return Mechanism(family="N3_SHUNT", tau_fast=mech.tau_fast, tau_slow=mech.tau_slow,
                         alpha=0.0, label="ablate_interaction")
    if fam == "N4_SUBUNIT":          # gamma -> 0 removes the multiplicative term
        return Mechanism(family="N4_SUBUNIT", tau_fast=mech.tau_fast, tau_slow=mech.tau_slow,
                         beta=mech.beta, gamma=0.0, label="ablate_interaction")
    if fam in ("P_ONLY", "OPPONENT_PRODUCT", "N5_HR_CONTROL"):
        return Mechanism(family="N1_LEXT", tau_fast=mech.tau_fast, tau_slow=mech.tau_slow,
                         label="ablate_interaction")
    return Mechanism(family="N1_LEXT", tau_fast=mech.tau_fast, tau_slow=mech.tau_slow,
                     label="ablate_interaction")


def temporal_flat(mech):
    return Mechanism(**{**mech.to_json(), "tau_slow": mech.tau_fast, "label": "temporal_flat"})


# ---------------------------------------------------------------- temporal metrics
STARTUP_FRAMES = 10          # preregistered fixed startup exclusion for metric C
EVENT_HALF_WINDOW = 6        # preregistered fixed half-window for metric D


def metric_traces(Y, window=None):
    """Mean response per neuron over a predetermined frame window (None = all frames)."""
    if window is None:
        return Y.mean(axis=1)
    lo, hi = window
    return Y[:, lo:hi].mean(axis=1)


def event_window_means(Y, crossing_frame, half=EVENT_HALF_WINDOW):
    """Metric D: mean over a fixed window around each neuron's predicted crossing frame.

    `crossing_frame` is derived from the rendered stimulus trajectory and the neuron's
    anatomical receptive-field centre only -- never from the response."""
    n, T = Y.shape
    out = np.zeros(n)
    lo = np.clip(crossing_frame - half, 0, T - 1).astype(int)
    hi = np.clip(crossing_frame + half + 1, 1, T).astype(int)
    for i in range(n):
        out[i] = Y[i, lo[i]:hi[i]].mean() if hi[i] > lo[i] else 0.0
    return out


def predicted_crossing(centre_u, centre_v, direction, frames, speed, bar_px, start_offset=0.0,
                       width=160, height=144):
    """Frame at which a parametric bar's centre crosses a receptive-field centre.

    Mirrors flymon.column_motion.parametric_bar's trajectory exactly."""
    horizontal = direction in ("left", "right")
    span = width if horizontal else height
    travel = speed * (frames - 1)
    base = (span - travel - bar_px) / 2.0 + bar_px / 2.0 + start_offset
    base = float(np.clip(base, bar_px / 2.0, span - bar_px / 2.0 - travel))
    pos = centre_u * (width - 1) if horizontal else centre_v * (height - 1)
    idx = (pos - base) / speed if speed > 0 else np.zeros_like(pos)
    if direction in ("left", "up"):
        idx = (frames - 1) - idx
    return np.clip(idx, 0, frames - 1)


def dsi(pref, null, eps=1e-9):
    return cm.dsi(pref, null, eps)
