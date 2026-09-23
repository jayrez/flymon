"""Frozen E23 T4 visual readout for closed-loop gameplay (Experiment 24 onward).

This is a *streaming* form of the Experiment-23 native candidate: the fixed-background,
pre-adapted, Mi1-excitation / Mi4-inhibition single passive compartment
(`run_spatial_veto_experiment.candidate()`, E23 commit e6a88c5). One call to
:meth:`FrozenT4Readout.update` advances every T4 neuron by exactly one game frame, with the
same arithmetic as `adapted_reference.simulate` applied to a whole sequence:

* column luminance of the frame (E5 retina, linear light) -> s = polarity (L - L0) per column;
* first-order low-pass per class (fast-arm classes tau_fast, slow-arm classes tau_slow),
  state initialised at zero contrast, which is exactly the E23 40-frame L0 pre-adaptation;
* presynaptic rate r = max(0, r0 + beta s / sigma) (excitatory classes r0 = r0_exc,
  Mi4 r0 = 0);
* conductances routed by MaleCNS weight sign, scale G / M, never negative;
* exact exponential membrane update starting at the tonic rest; output max(V - V_rest, 0).

Scientific boundary: everything in this module is **frozen sensory model**. Gameplay,
controllers and any future optimisation must treat it as read-only. The configuration is a
read-only mapping, projection arrays are write-protected, and a SHA-256 of configuration +
anatomy is exposed so runs can prove which sensory model they used.

Input classes: every MaleCNS partner class that E23's native candidate actually simulated,
i.e. all T4 projection classes except the dropped Mi9: Mi1, Mi4 **and the small Tm1 / Tm2 /
Tm9 -> T4 synapses** (excitatory, OFF polarity, tonic r0_exc). E23's report describes the
candidate as "Mi1 + Mi4 only"; the E23 code also carried these Tm inputs, and this module
reproduces the code bit-for-bit rather than the description (see E24 report).

T5 is not represented: its direction selectivity remains unresolved (E19-E23).
"""
from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import numpy as np

from flymon import column_motion as cm
from flymon import adapted_reference as ar

E23_COMMIT = "e6a88c5ae2272aad4ace8d8885cd8d55854b771d"
T4_SUBTYPES = ("T4a", "T4b", "T4c", "T4d")

# Exact E23 native candidate (verified against results/experiment-23-*/heldout-results.json
# by test_generation_zero.py). Changing any value here is a new sensory model, not gameplay.
FROZEN_E23_CONFIG = MappingProxyType({
    "level": "E23", "reference": "fixed", "background": 0.5, "adapt": 40,
    "G": 16.0, "beta": 2.0, "r0_exc": 1.0, "r0_mi9": 0.0, "r0_mi4": 0.0, "E_inh": -0.2,
    "tau_fast": 4.0, "tau_slow": 12.0, "drop": ("Mi9",), "membrane_cold": False,
})
FROZEN_CONSTANTS = MappingProxyType({"sigma_source": "E21 selected-config constants",
                                     "g_leak": 1.0, "E_leak": 0.0, "E_exc": 1.0, "c_mem": 1.0})


def _readonly(a):
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


class FrozenT4Readout:
    """Streaming frozen E23 T4 model over all anatomy-eligible T4 neurons.

    Parameters come only from the frozen configuration and E21 constants; `proj` defaults to
    the native MaleCNS projections and may be replaced only by an explicit control geometry
    (e.g. a column shuffle), which changes the recorded hash."""

    def __init__(self, ctx, constants, proj=None, geometry_label="native"):
        cfg = ar.AdaptedConfig.from_json(dict(FROZEN_E23_CONFIG))
        self.config = FROZEN_E23_CONFIG
        self._cfg = cfg
        self.geometry_label = geometry_label
        self.sigma = float(constants["sigma"])
        self.M = float(constants["M"])
        self.chemistry = MappingProxyType(dict(constants["chemistry"]))
        self.L0 = ar.measured_L0(cfg.background)
        self._records = ctx["records"]; self._colindex = ctx["colindex"]; self._uv = ctx["uv"]
        proj = proj or ctx["proj"]
        classes = tuple(c for c in cm.ALL_PARTNERS
                        if c in proj[T4_SUBTYPES[0]] and c not in cfg.drop)
        self.classes = classes
        pos, neg, brain_idx, sub_of = {c: [] for c in classes}, {c: [] for c in classes}, [], []
        from scipy import sparse
        for sub in T4_SUBTYPES:
            recs = ctx["audit"][sub]["records"]
            brain_idx.extend(r.brain_index for r in recs)
            sub_of.extend([sub] * len(recs))
            for c in classes:
                m = proj[sub][c].tocsr()
                p = m.copy(); p.data = np.where(p.data > 0, p.data, 0.0)
                n = m.copy(); n.data = np.where(n.data < 0, -n.data, 0.0)
                pos[c].append(p); neg[c].append(n)
        scale = cfg.G / self.M
        self._pos = {c: (sparse.vstack(pos[c]).tocsr() * scale) for c in classes}
        self._neg = {c: (sparse.vstack(neg[c]).tocsr() * scale) for c in classes}
        for mats in (self._pos, self._neg):
            for m in mats.values():
                m.data.setflags(write=False); m.indices.setflags(write=False); m.indptr.setflags(write=False)
        self.brain_index = _readonly(np.asarray(brain_idx, np.int64))
        self.subtype = _readonly(np.asarray(sub_of))
        self._alpha = {c: float(np.exp(-1.0 / (cfg.tau_fast if cm.arm_of(c) == "fast" else cfg.tau_slow)))
                       for c in classes}
        self._r0 = {c: (cfg.r0_exc if self.chemistry[c] == "exc" else ar.class_r0(cfg, c)) for c in classes}
        gE0 = sum(self._pos[c] @ np.full(self._pos[c].shape[1], self._r0[c]) for c in classes)
        gI0 = sum(self._neg[c] @ np.full(self._neg[c].shape[1], self._r0[c]) for c in classes)
        k = FROZEN_CONSTANTS
        self._g_rest = k["g_leak"] + gE0 + gI0
        self._V_rest = _readonly((k["g_leak"] * k["E_leak"] + gE0 * k["E_exc"] + gI0 * cfg.E_inh) / self._g_rest)
        self.n = len(self.brain_index)
        self.reset()

    # -- identity -------------------------------------------------------------------------
    def config_json(self):
        return dict(e23_commit=E23_COMMIT, model=dict(self.config), constants=dict(FROZEN_CONSTANTS),
                    sigma=self.sigma, M=self.M, chemistry=dict(self.chemistry), L0_measured=self.L0,
                    geometry=self.geometry_label, n_t4=int(self.n), input_classes=list(self.classes),
                    t5="not represented; T5 direction selectivity unresolved")

    def sha256(self):
        h = hashlib.sha256(json.dumps(self.config_json(), sort_keys=True, default=list).encode())
        for mats in (self._pos, self._neg):
            for c in self.classes:
                m = mats[c]
                for a in (m.data, m.indices, m.indptr):
                    h.update(np.ascontiguousarray(a).tobytes())
        h.update(self.brain_index.tobytes())
        return h.hexdigest()

    # -- dynamics ---------------------------------------------------------------------------
    def reset(self):
        """Adapted start: filters at zero contrast (L = L0), membrane at tonic rest."""
        ncol = self._pos[self.classes[0]].shape[1]
        self._acc = {c: np.zeros(ncol) for c in self.classes}
        self._V = self._V_rest.copy()
        self.frames = 0

    def column_luminance(self, frame_rgba):
        return cm.column_luminance(frame_rgba[None], self._records, self._colindex, self._uv)[0]

    def update_from_luminance(self, lum):
        """Advance one frame given per-column linear luminance; returns T4 responses."""
        s = lum.astype(np.float64) - self.L0
        k = FROZEN_CONSTANTS
        gE = np.zeros(self.n); gI = np.zeros(self.n)
        for c in self.classes:
            a = self._alpha[c]
            self._acc[c] = a * self._acc[c] + (1.0 - a) * cm.POLARITY[c] * s
            r = np.maximum(0.0, self._r0[c] + self._cfg.beta * self._acc[c] / self.sigma)
            gE += self._pos[c] @ r
            gI += self._neg[c] @ r
        gtot = k["g_leak"] + gE + gI
        Vinf = (k["g_leak"] * k["E_leak"] + gE * k["E_exc"] + gI * self._cfg.E_inh) / gtot
        self._V = Vinf + (self._V - Vinf) * np.exp(-gtot / k["c_mem"])
        self.frames += 1
        return np.maximum(self._V - self._V_rest, 0.0)

    def update(self, frame_rgba):
        return self.update_from_luminance(self.column_luminance(frame_rgba))


class T4Injection:
    """Frozen bridge from T4 responses to FlyBrain voltage injection into MaleCNS T4 cells.

    voltage = clip(gain * response, 0, cap), quantised to `levels` steps (the E3/E14
    injection convention: FlyBrain.step accepts one scalar per index group). `gain` is fixed
    before gameplay from E23 calibration stimuli only (see run_generation_zero.py)."""

    def __init__(self, brain_index, gain, cap=0.8, levels=16):
        self.brain_index = np.asarray(brain_index, np.int64)
        self.gain, self.cap, self.levels = float(gain), float(cap), int(levels)

    def voltages(self, response):
        v = np.clip(self.gain * np.asarray(response, float), 0.0, self.cap)
        step = self.cap / self.levels
        return np.rint(v / step) * step

    def pairs(self, response):
        v = self.voltages(response)
        return [(self.brain_index[v == value], float(value)) for value in np.unique(v) if value > 0]

    def config_json(self):
        return dict(gain=self.gain, cap=self.cap, levels=self.levels, target="MaleCNS T4a-d (E18 audit records)")
