"""Fixed adapting-background contrast reference (Experiment 22).

Every earlier column-motion experiment builds its contrast signal as
``s(x, t) = L(x, t) - mean_t L(x, t)`` (``column_motion.contrast_signal``): the zero point
is each stimulus sequence's own temporal mean, and the temporal filters start from the
first frame. Experiment 21's exploratory tonic-excitation lead came with fully inverted
ON/OFF preference, and one candidate explanation is exactly this per-sequence reference.
This module provides an explicit alternative without touching the canonical E18-E21 path:

* ``reference="sequence_mean"`` -- the unchanged canonical convention (control R0)
* ``reference="fixed"``         -- ``s = L - L0`` for a declared adapting luminance L0

plus an optional neutral pre-adaptation period at L0, prepended to the stimulus, during
which filters and the membrane settle. Adaptation frames are removed before scoring.

Luminance scale (verified, not assumed): the retina decodes sRGB into *linear* light
before column sampling, so display mid-gray 0.5 is linear ~0.21. Backgrounds here are
specified in **linear** light; bars and gratings are placed symmetrically around the
background with Weber contrast +/-c, and L0 is the linear luminance the pipeline actually
reads back from a rendered uniform frame (uint8 quantisation included).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from flymon import column_motion as cm
from flymon import tonic_disinhibition as td
from flymon.retina import linear_luminance

REFERENCES = ("sequence_mean", "fixed")
ADAPT_FRAMES = 40          # > 3 x tau_slow (12): filters fully settled at L0
PRIMARY_BACKGROUND = 0.5   # linear light


# ------------------------------------------------------------------ luminance scale
def srgb_encode(linear):
    """Inverse of the retina's sRGB decode: linear light -> display value in [0, 1]."""
    x = np.clip(np.asarray(linear, float), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def rendered_linear(display):
    """Exact linear luminance the pipeline sees for a uniform frame at this display value."""
    frame = cm.to_rgba(np.full((1, cm.H, cm.W), float(display), np.float32))[0]
    return float(linear_luminance(frame)[0, 0])


def display_for(linear):
    return float(srgb_encode(linear))


def measured_L0(background=PRIMARY_BACKGROUND):
    """L0 = linear luminance actually read back for the rendered background."""
    return rendered_linear(display_for(background))


def weber_levels(background, contrast):
    """Symmetric bar levels (linear) around the background: background * (1 +/- c),
    with the amplitude capped so both stay inside [0, 1]."""
    amp = contrast * min(background, 1.0 - background)
    return background + amp, background - amp


# ------------------------------------------------------------------ stimuli
def gray_bar(direction, polarity, frames=60, speed=1.0, bar_px=18, contrast=1.0,
             start_offset=0.0, background=PRIMARY_BACKGROUND):
    """E19 bar trajectory re-rendered on the adapting background.

    The bar mask is taken from ``column_motion.parametric_bar`` itself (a full-contrast ON
    bar, thresholded), so trajectories are identical to the E19 bank's."""
    mask = cm.parametric_bar(direction, "ON", frames, speed, bar_px, 1.0, start_offset) > 0.5
    hi, lo = weber_levels(background, contrast)
    fg = display_for(hi if polarity == "ON" else lo)
    return np.where(mask, fg, display_for(background)).astype(np.float32)


def gray_grating(direction, frames=60, speed=1.0, period=32.0, phase=0.0, contrast=1.0,
                 background=PRIMARY_BACKGROUND):
    """E19 sinusoid with its modulation applied in linear light around the background."""
    g = cm.parametric_grating(direction, frames, speed, period, phase, 1.0)
    amp = contrast * min(background, 1.0 - background)
    return srgb_encode(background + amp * (2.0 * g - 1.0)).astype(np.float32)


def uniform(frames=60, background=PRIMARY_BACKGROUND):
    return np.full((frames, cm.H, cm.W), display_for(background), np.float32)


def full_field_step(polarity, frames=60, step_at=20, contrast=0.5,
                    background=PRIMARY_BACKGROUND):
    """Background for `step_at` frames, then a full-field increment or decrement."""
    seq = uniform(frames, background)
    hi, lo = weber_levels(background, contrast)
    seq[step_at:] = display_for(hi if polarity == "ON" else lo)
    return seq


# ------------------------------------------------------------------ signal path
def contrast(lum, reference, L0=None):
    if reference == "sequence_mean":
        return cm.contrast_signal(lum)
    if reference == "fixed":
        if L0 is None:
            raise ValueError("fixed reference needs L0")
        return lum - L0
    raise ValueError(reference)


def prepend_adaptation(lum, L0, adapt):
    """(frames, cols) -> (adapt + frames, cols) with neutral L0 frames first."""
    if not adapt:
        return lum
    return np.concatenate([np.full((adapt, lum.shape[1]), L0, lum.dtype), lum], axis=0)


def cell_signals(lum, tau_fast, tau_slow, reference="fixed", L0=None, adapt=0,
                 temporal=True):
    """Polarity-applied, temporally filtered per-class signals, (adapt + frames, cols).

    With ``reference='sequence_mean'`` and ``adapt=0`` this is bit-identical to
    ``column_motion.cell_signals`` (and therefore to E21's ``signals_for``). Adaptation
    frames are *kept* here so the membrane can settle through them too; callers drop them
    with :func:`drop_adaptation` before scoring."""
    if reference == "sequence_mean" and adapt:
        raise ValueError("pre-adaptation is only defined for a fixed reference")
    if reference == "sequence_mean":
        return cm.cell_signals(lum, cm.ModelConfig(
            name="E22_R0", tau_fast=tau_fast, tau_slow=tau_slow, rectify_inputs=False,
            output_exponent=1.0, temporal=temporal))
    base = contrast(prepend_adaptation(lum, L0, adapt), "fixed", L0)
    out = {}
    for ctype in cm.ALL_PARTNERS:
        tau = 0.0 if not temporal else (tau_fast if cm.arm_of(ctype) == "fast" else tau_slow)
        out[ctype] = cm.low_pass(cm.POLARITY[ctype] * base, tau)
    return out


def drop_adaptation(x, adapt, axis=-1):
    if not adapt:
        return x
    idx = [slice(None)] * x.ndim
    idx[axis] = slice(adapt, None)
    return x[tuple(idx)]


# ------------------------------------------------------------------ model
@dataclass(frozen=True)
class AdaptedConfig:
    """E21 tonic single compartment + an explicit contrast reference.

    Class-level tonic rates only (never per subtype or per neuron). ``r0_mi9`` and
    ``r0_mi4`` are kept separate so the two inhibitory inputs can be decomposed."""
    level: str = "A1"
    reference: str = "fixed"
    background: float = PRIMARY_BACKGROUND
    adapt: int = 0
    G: float = 16.0
    beta: float = 2.0
    r0_exc: float = 1.0
    r0_mi9: float = 0.0
    r0_mi4: float = 0.0
    E_inh: float = -0.2
    tau_fast: float = 4.0
    tau_slow: float = 12.0
    drop: tuple = ()
    membrane_cold: bool = False    # membrane starts at E_leak instead of its tonic rest

    def key(self):
        k = (f"{self.level}|{self.reference}|L{self.background:g}|ad{self.adapt}|G{self.G:g}"
             f"|b{self.beta:g}|r0e{self.r0_exc:g}|mi9{self.r0_mi9:g}|mi4{self.r0_mi4:g}")
        if self.drop:
            k += "|drop" + "+".join(self.drop)
        if self.tau_slow != 12.0 or self.tau_fast != 4.0:
            k += f"|tau{self.tau_fast:g}/{self.tau_slow:g}"
        if self.membrane_cold:
            k += "|Vcold"
        return k

    def to_json(self):
        d = asdict(self); d["drop"] = list(self.drop)
        return d

    @classmethod
    def from_json(cls, d):
        d = dict(d); d["drop"] = tuple(d.get("drop", ()))
        return cls(**d)

    def replace(self, **kw):
        d = self.to_json(); d.update(kw)
        return AdaptedConfig.from_json(d)

    def tonic(self, r0_inh=0.0):
        """E21 TonicConfig carrying the shared constants (D1 linear transfer)."""
        return td.TonicConfig(family="D1_UNIFORM_TONIC", tau_fast=self.tau_fast,
                              tau_slow=self.tau_slow, G=self.G, beta=self.beta,
                              r0_exc=self.r0_exc, r0_inh=r0_inh, E_inh=self.E_inh)


def e21_a1():
    """The E21 exploratory A1 configuration expressed in this module (control R0)."""
    return AdaptedConfig(level="A0", reference="sequence_mean", adapt=0, G=16.0, beta=2.0,
                         r0_exc=1.0, r0_mi9=0.0, r0_mi4=0.0, E_inh=-0.2)


def class_r0(cfg, cls):
    return {"Mi9": cfg.r0_mi9, "Mi4": cfg.r0_mi4}.get(cls, 0.0)


def conductances(proj_sub, signals, cfg, sigma, M, chemistry):
    """E21 conductances, class by class, with per-class inhibitory tonic rates."""
    out = {}
    for cls, m in proj_sub.items():
        if cls in cfg.drop or cls not in signals:
            continue
        tc = cfg.tonic(r0_inh=class_r0(cfg, cls))
        out.update(td.conductances({cls: m}, signals, tc, sigma, M, chemistry))
    return out


def integrate(cond, cfg):
    """E21 exact per-frame integration; optional cold membrane (V0 = E_leak)."""
    tc = cfg.tonic()
    if not cfg.membrane_cold:
        return td.integrate(cond, tc)
    if not cond:
        return None, None
    first = next(iter(cond.values()))
    n, T = first["gE"].shape
    gE = sum(c["gE"] for c in cond.values()); gI = sum(c["gI"] for c in cond.values())
    gE0 = sum(c["gE0"] for c in cond.values()); gI0 = sum(c["gI0"] for c in cond.values())
    g_rest = tc.g_leak + gE0 + gI0
    V_rest = (tc.g_leak * tc.E_leak + gE0 * tc.E_exc + gI0 * tc.E_inh) / g_rest
    V = np.full(n, tc.E_leak, float)
    Vt = np.empty((n, T))
    for k in range(T):
        e, i = gE[:, k], gI[:, k]
        gtot = tc.g_leak + e + i
        Vinf = (tc.g_leak * tc.E_leak + e * tc.E_exc + i * tc.E_inh) / gtot
        V = Vinf + (V - Vinf) * np.exp(-gtot / tc.c_mem)
        Vt[:, k] = V
    response = np.maximum(Vt - V_rest[:, None], 0.0)
    aux = dict(V=Vt, V_rest=V_rest, g_total=tc.g_leak + gE + gI, g_rest=g_rest,
               R_in=1.0 / (tc.g_leak + gE + gI), R_rest=1.0 / g_rest, gE=gE, gI=gI)
    return response, aux


def simulate(proj_sub, signals, cfg, sigma, M, chemistry):
    """Response (n, frames) with adaptation frames already removed, plus traces."""
    cond = conductances(proj_sub, signals, cfg, sigma, M, chemistry)
    response, aux = integrate(cond, cfg)
    if response is None:
        return None, None
    response = drop_adaptation(response, cfg.adapt)
    for k in ("V", "g_total", "R_in", "gE", "gI"):
        aux[k] = drop_adaptation(aux[k], cfg.adapt)
    for c in cond.values():
        c["gE"] = drop_adaptation(c["gE"], cfg.adapt)
        c["gI"] = drop_adaptation(c["gI"], cfg.adapt)
    aux["classes"] = cond
    return response, aux


def temporal_flat(cfg):
    return cfg.replace(tau_slow=cfg.tau_fast)
