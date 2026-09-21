"""Deterministic ethological visual-motor stimuli for Experiment 16.

Synthetic Game Boy-sized (144x160) RGBA framebuffer sequences probing the motion,
looming and optic-flow features the fly optic lobe is built to encode -- not scene
identity. Luminance is controlled so direction / order cannot be read from mean
brightness: within each polarity the moving-bar stimuli have a constant bright area
per frame, and every temporal control (frozen / shuffle / reverse) reuses the exact
frames of its source sequence, so its luminance histogram is identical.

No game state, RAM, reward, or action label is involved.
"""
from __future__ import annotations

import numpy as np

H, W = 144, 160
FRAMES = 10
BAR = 18            # moving-bar thickness (px)
DISC_R0, DISC_R1 = 12.0, 60.0   # looming disc radius range
GRATING_PERIOD = 32.0
SHUFFLE_SEED = 1600


def _rgba(gray: np.ndarray) -> np.ndarray:
    """(...,H,W) luminance in [0,1] -> (...,H,W,4) uint8 RGBA."""
    g = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    rgb = np.repeat(g[..., None], 3, axis=-1)
    alpha = np.full(g.shape + (1,), 255, np.uint8)
    return np.concatenate([rgb, alpha], axis=-1)


def _bar_positions(n, span):
    # sweep the bar centre across the axis, staying fully inside the frame
    lo, hi = BAR / 2, span - BAR / 2
    return np.linspace(lo, hi, n)


def _moving_bar(direction: str, polarity: str, n=FRAMES):
    """Bar of constant area sweeping across a uniform background.

    polarity 'ON' = bright bar on dark background; 'OFF' = dark bar on bright."""
    bg, fg = (0.0, 1.0) if polarity == "ON" else (1.0, 0.0)
    frames = []
    horizontal = direction in ("left", "right")
    span = W if horizontal else H
    pos = _bar_positions(n, span)
    if direction in ("right", "down"):
        centres = pos
    else:  # left / up sweep the opposite way (mirror), identical luminance
        centres = pos[::-1]
    for c in centres:
        g = np.full((H, W), bg, np.float32)
        if horizontal:
            x0 = int(round(c - BAR / 2)); g[:, x0:x0 + BAR] = fg
        else:
            y0 = int(round(c - BAR / 2)); g[y0:y0 + BAR, :] = fg
        frames.append(g)
    return np.stack(frames)


def _disc(sign: str, radii, n=FRAMES):
    """Disc of given radius per frame. sign 'dark' = dark disc on bright bg."""
    bg, fg = (1.0, 0.0) if sign == "dark" else (0.0, 1.0)
    yy, xx = np.mgrid[:H, :W]
    cy, cx = (H - 1) / 2, (W - 1) / 2
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    frames = []
    for r in radii:
        g = np.full((H, W), bg, np.float32)
        g[dist <= r] = fg
        frames.append(g)
    return np.stack(frames)


def _grating(kind: str, n=FRAMES):
    """Full-field drifting sinusoid: horizontal flow or radial expansion/contraction."""
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    cy, cx = (H - 1) / 2, (W - 1) / 2
    frames = []
    for i in range(n):
        phase = 2 * np.pi * i / n
        if kind in ("flow_left", "flow_right"):
            s = -1 if kind == "flow_left" else 1
            g = 0.5 + 0.5 * np.sin(2 * np.pi * xx / GRATING_PERIOD - s * phase * 2)
        else:  # radial: expansion (outward) or contraction (inward)
            s = 1 if kind == "flow_expand" else -1
            rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            g = 0.5 + 0.5 * np.sin(2 * np.pi * rad / GRATING_PERIOD - s * phase * 2)
        frames.append(g.astype(np.float32))
    return np.stack(frames)


def _static(kind: str, n=FRAMES):
    if kind == "dark": g = np.zeros((H, W), np.float32)
    elif kind == "gray": g = np.full((H, W), 0.5, np.float32)
    elif kind == "bright": g = np.ones((H, W), np.float32)
    elif kind == "vedge":
        g = np.zeros((H, W), np.float32); g[:, W // 2:] = 1.0
    elif kind == "hedge":
        g = np.zeros((H, W), np.float32); g[H // 2:, :] = 1.0
    elif kind == "dark_disc":
        return _disc("dark", [np.mean([DISC_R0, DISC_R1])] * n, n)
    else:
        raise ValueError(kind)
    return np.stack([g] * n)


def base_sequence(name: str) -> np.ndarray:
    """Float (FRAMES, H, W) luminance sequence for a base stimulus name."""
    if name in ("dark", "gray", "bright", "vedge", "hedge", "dark_disc"):
        return _static(name)
    if name.startswith("on_"):
        return _moving_bar(name[3:], "ON")
    if name.startswith("off_"):
        return _moving_bar(name[4:], "OFF")
    if name == "loom_dark":
        return _disc("dark", np.linspace(DISC_R0, DISC_R1, FRAMES))
    if name == "loom_bright":
        return _disc("bright", np.linspace(DISC_R0, DISC_R1, FRAMES))
    if name == "recede_dark":
        return _disc("dark", np.linspace(DISC_R1, DISC_R0, FRAMES))
    if name == "recede_bright":
        return _disc("bright", np.linspace(DISC_R1, DISC_R0, FRAMES))
    if name in ("flow_left", "flow_right", "flow_expand", "flow_contract"):
        return _grating(name)
    raise ValueError(name)


def shuffle_order(n=FRAMES):
    """Deterministic non-identity permutation of frame indices."""
    order = np.random.default_rng(SHUFFLE_SEED).permutation(n)
    if np.array_equal(order, np.arange(n)):        # guarantee a real shuffle
        order = np.roll(order, 1)
    return order


def temporal_variant(seq: np.ndarray, variant: str) -> np.ndarray:
    if variant == "normal":
        return seq
    if variant == "frozen_first":
        return np.repeat(seq[:1], len(seq), axis=0)
    if variant == "frozen_mid":
        return np.repeat(seq[len(seq) // 2:len(seq) // 2 + 1], len(seq), axis=0)
    if variant == "shuffle":
        return seq[shuffle_order(len(seq))]
    if variant == "reverse":
        return seq[::-1]
    raise ValueError(variant)


# Stimulus catalogue: base names, and which get temporal controls.
STATIC = ("dark", "gray", "bright", "vedge", "hedge", "dark_disc")
ON_MOTION = ("on_left", "on_right", "on_up", "on_down")
OFF_MOTION = ("off_left", "off_right", "off_up", "off_down")
LOOM = ("loom_dark", "loom_bright", "recede_dark", "recede_bright")
FLOW = ("flow_left", "flow_right", "flow_expand", "flow_contract")
TEMPORAL_SOURCES = ("off_left", "off_right", "on_left", "loom_dark", "flow_expand")
TEMPORAL_VARIANTS = ("frozen_first", "frozen_mid", "shuffle", "reverse")


def catalogue():
    """Ordered dict name -> uint8 RGBA (FRAMES,H,W,4) for the full stimulus set."""
    out = {}
    for name in (*STATIC, *ON_MOTION, *OFF_MOTION, *LOOM, *FLOW):
        out[name] = _rgba(base_sequence(name))
    for src in TEMPORAL_SOURCES:
        seq = base_sequence(src)
        for v in TEMPORAL_VARIANTS:
            out[f"{src}__{v}"] = _rgba(temporal_variant(seq, v))
    return out


def mean_luminance(seq_rgba: np.ndarray) -> float:
    return float(seq_rgba[..., :3].mean() / 255.0)
