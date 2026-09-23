"""Tonic shunting inhibition and release from inhibition (Experiment 21).

Experiment 20 mapped each presynaptic signal to ``max(s, 0)`` on a temporally
mean-centred contrast signal, so inhibitory conductance started at zero and could never
be *released*; measured E20 inhibition also peaked at <0.1 x g_L, a near-linear regime.
This module represents a non-zero operating point instead: every presynaptic partner
fires at a class-level tonic rate ``r0_c`` that the stimulus modulates up *or down*,

    r_j(t) = max(0, r0_c + beta * s_c(col_j, t) / sigma)            (linear, D1/D2)
    r_j(t) = 2 r0_c * logistic(2 beta s_c / (sigma r0_c))           (bounded, D3)

and each synapse's conductance is routed by the sign of its MaleCNS weight, so
conductances are never negative while an OFF-polarity inhibitory cell such as Mi9 drops
*below* its tonic level during ON motion. A single passive compartment integrates

    C dV/dt = -g_L (V - E_L) - g_E (V - E_E) - g_I (V - E_I)

exactly per frame. Nothing here evaluates an opponent product or imports the Experiment-19
mechanism code; direction selectivity, if any, must come from conductance dynamics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from flymon import column_motion as cm

FAMILIES = ("D1_UNIFORM_TONIC", "D2_TONIC_INHIBITION", "D3_BOUNDED_TONIC_INHIBITION")
T4_CLASSES = ("Mi1", "Mi9", "Mi4")
T5_CLASSES = ("Tm1", "Tm2", "Tm9")


@dataclass(frozen=True)
class TonicConfig:
    family: str = "D2_TONIC_INHIBITION"
    tau_fast: float = 4.0
    tau_slow: float = 12.0
    G: float = 4.0              # synaptic conductance scale, in units of g_L
    beta: float = 1.0           # modulation per unit signal SD
    r0_exc: float = 0.0         # tonic rate of excitatory classes
    r0_inh: float = 1.0         # tonic rate of inhibitory classes
    E_inh: float = 0.0          # 0 = pure shunt, <0 = hyperpolarising
    bounded: bool = False       # D3 logistic transfer for inhibitory classes
    # ablations
    clamp_inhibition: bool = False
    no_release: bool = False
    neutral_inhibition: bool = False
    drop_classes: tuple = field(default_factory=tuple)
    g_leak: float = 1.0
    E_leak: float = 0.0
    E_exc: float = 1.0
    c_mem: float = 1.0

    def key(self):
        return (f"{self.family}|G{self.G:g}|b{self.beta:g}|r0e{self.r0_exc:g}"
                f"|r0i{self.r0_inh:g}|Ei{self.E_inh:g}")

    def to_json(self):
        d = asdict(self)
        d["drop_classes"] = list(self.drop_classes)
        return d

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        d["drop_classes"] = tuple(d.get("drop_classes", ()))
        return cls(**d)

    def replace(self, **kw):
        d = self.to_json(); d.update(kw)
        return TonicConfig.from_json(d)


def class_chemistry(proj):
    """'exc' or 'inh' per presynaptic class, from the sign of its summed MaleCNS weight."""
    total = {}
    for mats in proj.values():
        for t, m in mats.items():
            total[t] = total.get(t, 0.0) + float(m.sum())
    return {t: ("exc" if v > 0 else "inh") for t, v in total.items()}


def normaliser(proj, subtypes=("T4a", "T4b", "T4c", "T4d"), classes=T4_CLASSES):
    """Anatomy-only constant M: median per-neuron summed |w| over the T4 classes."""
    per = []
    for s in subtypes:
        if s not in proj:
            continue
        tot = None
        for t in classes:
            m = proj[s].get(t)
            if m is None:
                continue
            v = np.asarray(abs(m).sum(axis=1)).ravel()
            tot = v if tot is None else tot + v
        if tot is not None:
            per.append(tot)
    return float(np.median(np.concatenate(per)))


def signals_for(lum, cfg):
    return cm.cell_signals(lum, cm.ModelConfig(
        name=cfg.family, tau_fast=cfg.tau_fast, tau_slow=cfg.tau_slow,
        rectify_inputs=False, output_exponent=1.0, temporal=True))


def presynaptic_rate(s, r0, cfg, sigma, inhibitory):
    """Rate of each column's presynaptic cell over time; never negative."""
    if inhibitory and cfg.clamp_inhibition:
        return np.full_like(s, r0)
    if inhibitory and cfg.bounded and r0 > 0:
        z = 2.0 * cfg.beta * s / (sigma * r0)
        r = 2.0 * r0 / (1.0 + np.exp(-z))
    else:
        r = np.maximum(0.0, r0 + cfg.beta * s / sigma)
    if inhibitory and cfg.no_release:
        r = np.maximum(r, r0)
    return r


def conductances(proj_sub, signals, cfg, sigma, M, chemistry):
    """Per-class excitatory/inhibitory conductance traces plus tonic baselines.

    Returns {class: dict(gE=(n,T), gI=(n,T), gE0=(n,), gI0=(n,))}; all values >= 0."""
    out = {}
    for t, m in proj_sub.items():
        if t in cfg.drop_classes or t not in signals:
            continue
        inhibitory = chemistry.get(t) == "inh"
        r0 = cfg.r0_inh if inhibitory else cfg.r0_exc
        rate = presynaptic_rate(signals[t], r0, cfg, sigma, inhibitory)     # (T, cols)
        pos = m.copy(); pos.data = np.where(pos.data > 0, pos.data, 0.0)
        neg = m.copy(); neg.data = np.where(neg.data < 0, -neg.data, 0.0)
        scale = cfg.G / M
        gE = scale * (pos @ rate.T)
        gI = scale * (neg @ rate.T)
        gE0 = scale * r0 * np.asarray(pos.sum(axis=1)).ravel()
        gI0 = scale * r0 * np.asarray(neg.sum(axis=1)).ravel()
        out[t] = dict(gE=gE, gI=gI, gE0=gE0, gI0=gI0)
    return out


def integrate(cond, cfg):
    """Exact exponential update per frame. Returns response and mechanism traces."""
    if not cond:
        return None, None
    first = next(iter(cond.values()))
    n, T = first["gE"].shape
    gE = sum(c["gE"] for c in cond.values()); gI = sum(c["gI"] for c in cond.values())
    gE0 = sum(c["gE0"] for c in cond.values()); gI0 = sum(c["gI0"] for c in cond.values())
    E_inh = cfg.E_leak if cfg.neutral_inhibition else cfg.E_inh
    g_rest = cfg.g_leak + gE0 + gI0
    V_rest = (cfg.g_leak * cfg.E_leak + gE0 * cfg.E_exc + gI0 * E_inh) / g_rest
    V = V_rest.copy()
    Vt = np.empty((n, T))
    for k in range(T):
        e, i = gE[:, k], gI[:, k]
        gtot = cfg.g_leak + e + i
        Vinf = (cfg.g_leak * cfg.E_leak + e * cfg.E_exc + i * E_inh) / gtot
        V = Vinf + (V - Vinf) * np.exp(-gtot / cfg.c_mem)
        Vt[:, k] = V
    response = np.maximum(Vt - V_rest[:, None], 0.0)
    aux = dict(V=Vt, V_rest=V_rest, g_total=cfg.g_leak + gE + gI, g_rest=g_rest,
               R_in=1.0 / (cfg.g_leak + gE + gI), R_rest=1.0 / g_rest, gE=gE, gI=gI)
    return response, aux


def simulate(proj_sub, signals, cfg, sigma, M, chemistry):
    cond = conductances(proj_sub, signals, cfg, sigma, M, chemistry)
    response, aux = integrate(cond, cfg)
    if aux is not None:
        aux["classes"] = cond
    return response, aux


def temporal_flat(cfg):
    return cfg.replace(tau_slow=cfg.tau_fast)
