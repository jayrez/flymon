"""Inspectable 2-D framebuffer features and a declared projection surrogate.

FlyBrain has no receptive-field coordinates for LPLC2 or LC10a. Grid samples
are assigned to metadata-selected cells deterministically; this is not an
anatomical retinotopy claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .vision import preprocess


@dataclass(frozen=True)
class SpatiotemporalConfig:
    grid_height: int = 18
    grid_width: int = 20
    frames: int = 10
    game_frame_interval: int = 4
    neural_steps_per_frame: int = 20
    darkness_gain: float = 0.55
    change_gain: float = 0.75
    voltage_cap: float = 0.8
    voltage_levels: int = 16


def spatial_grid(frame: np.ndarray, config: SpatiotemporalConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return full-resolution grayscale and exact 8x8 pooled image."""
    gray, _ = preprocess(frame)
    if 144 % config.grid_height or 160 % config.grid_width or config.grid_width % 2:
        raise ValueError("Grid must divide 144x160 and have an even width")
    h, w = 144 // config.grid_height, 160 // config.grid_width
    grid = gray.reshape(config.grid_height, h, config.grid_width, w).mean(axis=(1, 3))
    return gray, grid.astype(np.float32)


def temporal_features(grids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Darkness and absolute change, with no invented change at the first frame."""
    if grids.ndim != 3 or not np.all(np.isfinite(grids)):
        raise ValueError("Expected finite [time, height, width] grids")
    darkness = np.clip(1.0 - grids, 0.0, 1.0).astype(np.float32)
    change = np.zeros_like(grids, dtype=np.float32)
    change[1:] = np.abs(np.diff(grids, axis=0))
    return darkness, change


class SpatialProjection:
    """Map grid coordinates to LPLC2/LC10a cells without claiming true RFs."""

    def __init__(self, brain, config: SpatiotemporalConfig):
        self.config = config
        self.populations = {}
        self.grid_indices = {}
        half = config.grid_width // 2
        for kind in ("LPLC2", "LC10a"):
            for side in ("L", "R"):
                ids = brain.cells([kind], side)
                if not len(ids) or not np.all(brain.superclass[ids] == "visual_projection"):
                    raise RuntimeError(f"Missing visual-projection metadata for {kind} {side}")
                if len(ids) > config.grid_height * half:
                    raise ValueError("More target cells than grid locations in one half")
                # Each class evenly samples its own half of the 2-D grid.
                positions = np.rint(np.linspace(0, config.grid_height * half - 1, len(ids))).astype(int)
                y, x_local = np.divmod(positions, half)
                x = x_local + (0 if side == "L" else half)
                self.populations[(kind, side)] = ids
                self.grid_indices[(kind, side)] = (y, x)
        self.ids = np.concatenate(list(self.populations.values())).astype(np.int32)
        if len(np.unique(self.ids)) != len(self.ids):
            raise RuntimeError("Overlapping projection populations")

    def encode(self, darkness: np.ndarray, change: np.ndarray) -> np.ndarray:
        if darkness.shape != (self.config.grid_height, self.config.grid_width) or change.shape != darkness.shape:
            raise ValueError("Unexpected spatial feature shape")
        out = []
        for (kind, side), (y, x) in self.grid_indices.items():
            source = change if kind == "LPLC2" else darkness
            gain = self.config.change_gain if kind == "LPLC2" else self.config.darkness_gain
            out.append(np.clip(gain * source[y, x], 0, self.config.voltage_cap))
        raw = np.concatenate(out).astype(np.float32)
        # FlyBrain.step accepts one scalar voltage per target index group.
        # Uniform quantization permits grouping cells with equal voltage.
        step = self.config.voltage_cap / self.config.voltage_levels
        return (np.rint(raw / step) * step).astype(np.float32)

    def injection_pairs(self, vector: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Group cells by scalar voltage, as supported by FlyBrain.step."""
        if vector.shape != (len(self.ids),):
            raise ValueError("Projection vector length mismatch")
        return [(self.ids[vector == value], float(value))
                for value in np.unique(vector) if value > 0]


def vector_stats(vector: np.ndarray) -> dict:
    raw = np.ascontiguousarray(vector, dtype=np.float32)
    return dict(sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                min=float(raw.min()), max=float(raw.max()), mean=float(raw.mean()),
                std=float(raw.std()), nonzero=int(np.count_nonzero(raw)))
