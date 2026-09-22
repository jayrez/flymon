"""Minimal sign-preserving conductance / compartmental dendrite models (Experiment 20).

Experiment 19 showed the MaleCNS T4/T5 geometry carries usable directional
information, but only an *antisymmetric opponent* readout on *sign-preserving* drives
recovered it (held-out mean DSI +0.3562, 8/8 subtypes). Experiment 20 asks whether
plausible membrane dynamics can produce an equivalent interaction **without ever
evaluating that opponent expression**.

Sign handling (the crux)
------------------------
Physical conductances are non-negative; the sign of a synapse's effect comes from its
reversal potential, not from a negative conductance. For each presynaptic cell class
`t` with MaleCNS weight `w` and polarity-filtered signal `s_t`:

* presynaptic activation `a_t = ReLU(s_t)`  (a firing rate cannot be negative --
  this rectification is at the *presynaptic cell*, not on the pooled arm drive)
* `w > 0` routes `|w| * a_t` into the **excitatory** conductance `g_E >= 0`
* `w < 0` routes `|w| * a_t` into the **inhibitory** conductance `g_I >= 0`

This differs fundamentally from Experiment 19's "magnitude drives", which pooled
`|w|` across *all* partners and thereby erased the excitatory/inhibitory distinction.
Here the magnitude sets conductance size while the connectome sign sets the channel,
so T4's inhibitory slow arm (Mi9, Mi4) and T5's excitatory slow arm (Tm9) remain
mechanistically distinct with no subtype-specific coefficients.

Membrane
--------
    C dV/dt = -g_L (V - E_L) - g_E(t)(V - E_E) - g_I(t)(V - E_I) - g_c (V - V_other)

integrated with exponential Euler on sub-steps within each stimulus frame (exact for
piecewise-constant conductances; coupling is treated semi-implicitly).

Compartment assignment is a **model hypothesis**: MaleCNS v1.0 carries no subcellular
or dendritic-compartment annotation for T4/T5, so which arm lands on which compartment
is not an anatomical fact and is tested by the arm-swap ablation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from flymon import column_motion as cm
from flymon.optic_columns import FAST_ARM, SLOW_ARM

SUBSTEPS = 8          # membrane sub-steps per stimulus frame
MODELS = ("C0_SINGLE", "C1_TWOCOMP", "C2_TWOCOMP_NL")


@dataclass(frozen=True)
class ConductanceConfig:
    """Global parameters. Deliberately few, and never per-subtype or per-neuron."""
    model: str = "C1_TWOCOMP"
    tau_fast: float = 4.0
    tau_slow: float = 12.0
    g_syn: float = 1.0         # synaptic conductance scale
    g_leak: float = 1.0
    g_couple: float = 0.5      # inter-compartment coupling
    E_leak: float = 0.0
    E_exc: float = 1.0
    E_inh: float = -0.2
    threshold: float = 0.0     # C2 output threshold
    c_mem: float = 1.0
    swap_arms: bool = False    # ablation A6
    magnitude_only: bool = False   # ablation A4 (destroy synaptic sign)
    neutral_inhibition: bool = False  # ablation A5 (E_inh -> E_leak)
    readout: str = "A"         # which compartment is the output

    def key(self):
        k = (f"{self.model}|tf{self.tau_fast:g}|ts{self.tau_slow:g}"
             f"|gs{self.g_syn:g}|Ei{self.E_inh:g}")
        if self.model != "C0_SINGLE":
            k += f"|gc{self.g_couple:g}"
        if self.model == "C2_TWOCOMP_NL":
            k += f"|th{self.threshold:g}"
        return k

    def to_json(self):
        return asdict(self)


def arm_conductances(proj_sub, signals, family, cfg):
    """(g_E, g_I) per arm: dicts 'fast'/'slow' of (n_neurons, frames), all >= 0.

    Conductance magnitude uses |w|; the *channel* (excitatory vs inhibitory) is chosen
    by the sign of the MaleCNS weight, so synaptic identity is preserved."""
    out = {}
    for arm, types in (("fast", FAST_ARM[family]), ("slow", SLOW_ARM[family])):
        gE = None
        gI = None
        for t in types:
            m = proj_sub.get(t)
            if m is None:
                continue
            a = np.maximum(signals[t], 0.0).T           # presynaptic rate, (cols, frames)->T
            pos = m.copy(); pos.data = np.where(pos.data > 0, pos.data, 0.0)
            neg = m.copy(); neg.data = np.where(neg.data < 0, -neg.data, 0.0)
            ce = pos @ a
            ci = neg @ a
            if cfg.magnitude_only:                      # A4: all input becomes excitatory
                ce = ce + ci
                ci = np.zeros_like(ci)
            gE = ce if gE is None else gE + ce
            gI = ci if gI is None else gI + ci
        if gE is None:
            continue
        out[arm] = (cfg.g_syn * gE, cfg.g_syn * gI)
    return out


def _reversals(cfg):
    E_inh = cfg.E_leak if cfg.neutral_inhibition else cfg.E_inh
    return cfg.E_exc, E_inh


def integrate(conductances, cfg):
    """Integrate the membrane equation; returns (V_out, V_A, V_B) each (n, frames)."""
    E_exc, E_inh = _reversals(cfg)
    fast = conductances.get("fast")
    slow = conductances.get("slow")
    if fast is None or slow is None:
        return None, None, None
    if cfg.swap_arms:
        fast, slow = slow, fast
    n, T = fast[0].shape
    dt = 1.0 / SUBSTEPS

    if cfg.model == "C0_SINGLE":
        gE = fast[0] + slow[0]
        gI = fast[1] + slow[1]
        V = np.full(n, cfg.E_leak)
        out = np.empty((n, T))
        for k in range(T):
            e, i = gE[:, k], gI[:, k]
            gtot = cfg.g_leak + e + i
            Vinf = (cfg.g_leak * cfg.E_leak + e * E_exc + i * E_inh) / gtot
            decay = np.exp(-dt * SUBSTEPS * gtot / cfg.c_mem)
            V = Vinf + (V - Vinf) * decay
            out[:, k] = V
        return out, out, out

    # two compartments: A receives the fast arm, B the slow arm, coupled by g_couple
    gEA, gIA = fast
    gEB, gIB = slow
    VA = np.full(n, cfg.E_leak)
    VB = np.full(n, cfg.E_leak)
    outA = np.empty((n, T)); outB = np.empty((n, T))
    gc = cfg.g_couple
    for k in range(T):
        eA, iA = gEA[:, k], gIA[:, k]
        eB, iB = gEB[:, k], gIB[:, k]
        gA = cfg.g_leak + eA + iA + gc
        gB = cfg.g_leak + eB + iB + gc
        IA = cfg.g_leak * cfg.E_leak + eA * E_exc + iA * E_inh
        IB = cfg.g_leak * cfg.E_leak + eB * E_exc + iB * E_inh
        for _ in range(SUBSTEPS):
            # semi-implicit: each compartment relaxes toward a target set by the other
            VinfA = (IA + gc * VB) / gA
            VinfB = (IB + gc * VA) / gB
            VA = VinfA + (VA - VinfA) * np.exp(-dt * gA / cfg.c_mem)
            VB = VinfB + (VB - VinfB) * np.exp(-dt * gB / cfg.c_mem)
        outA[:, k] = VA
        outB[:, k] = VB
    Vout = outA if cfg.readout == "A" else outB
    return Vout, outA, outB


def response(Vout, cfg):
    """Non-negative output rate from membrane depolarisation above rest."""
    if Vout is None:
        return None
    dep = Vout - cfg.E_leak
    if cfg.model == "C2_TWOCOMP_NL":
        return np.maximum(dep - cfg.threshold, 0.0)
    return np.maximum(dep, 0.0)


def simulate(proj_sub, signals, family, cfg):
    g = arm_conductances(proj_sub, signals, family, cfg)
    Vout, VA, VB = integrate(g, cfg)
    return response(Vout, cfg), dict(V_A=VA, V_B=VB, conductances=g)


def signals_for(lum, cfg):
    return cm.cell_signals(lum, cm.ModelConfig(
        name=cfg.model, tau_fast=cfg.tau_fast, tau_slow=cfg.tau_slow,
        rectify_inputs=False, output_exponent=1.0, temporal=True))


def temporal_flat(cfg):
    return ConductanceConfig(**{**cfg.to_json(), "tau_slow": cfg.tau_fast})


def decouple(cfg):
    return ConductanceConfig(**{**cfg.to_json(), "g_couple": 0.0})


def destroy_sign(cfg):
    return ConductanceConfig(**{**cfg.to_json(), "magnitude_only": True})


def neutralise_inhibition(cfg):
    return ConductanceConfig(**{**cfg.to_json(), "neutral_inhibition": True})


def swap(cfg):
    return ConductanceConfig(**{**cfg.to_json(), "swap_arms": True})
