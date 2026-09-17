"""Pokémon framebuffer to MaleCNS photoreceptor drive.

The temporal blend matches flybrain.eyes.Eyes: 0.45*luminance +
1.6*absolute change, clipped to [0, 1]. Eyes itself renders Blob objects,
so it cannot directly accept our measured screen panorama.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisionConfig:
    luminance_gain: float = 0.45
    change_gain: float = 1.6
    onset_background: float = 0.9


def preprocess(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RGBA [144,160,4] -> luminance image and vertical-mean panorama [160].

    RGB uses Rec.709 weights. Dividing by 255 retains absolute brightness;
    per-image min/max normalization would erase screen brightness differences.
    X=0 stays left, X=159 stays right.
    """
    if frame.shape != (144, 160, 4) or frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGBA [144,160,4], got {frame.shape} {frame.dtype}")
    rgb = frame[..., :3].astype(np.float32) / 255.0
    gray = np.clip(rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32), 0, 1)
    panorama = gray.mean(axis=0).astype(np.float32)
    return gray, panorama


class PanoramaEyes:
    def __init__(self, azimuth: np.ndarray, config: VisionConfig = VisionConfig()):
        self.azimuth = np.asarray(azimuth, dtype=np.float32)
        if self.azimuth.ndim != 1 or not np.all(np.isfinite(self.azimuth)):
            raise ValueError("Expected a finite one-dimensional receptor azimuth array")
        if np.any((self.azimuth < -1) | (self.azimuth > 1)):
            raise ValueError("Receptor azimuths must lie in [-1,1]")
        self.config = config
        self.previous: np.ndarray | None = None

    def reset(self, background: float | None = None) -> None:
        """Set a common pre-stimulus screen for controlled image onset."""
        value = self.config.onset_background if background is None else background
        self.previous = np.full(len(self.azimuth), value, dtype=np.float32)

    def project(self, panorama: np.ndarray) -> np.ndarray:
        panorama = np.asarray(panorama, dtype=np.float32)
        if panorama.shape != (160,) or not np.all(np.isfinite(panorama)):
            raise ValueError("Expected a finite 160-column panorama")
        x = np.linspace(-1, 1, 160, dtype=np.float32)
        return np.interp(self.azimuth, x, np.clip(panorama, 0, 1)).astype(np.float32)

    def drive(self, panorama: np.ndarray) -> np.ndarray:
        luminance = self.project(panorama)
        change = np.zeros_like(luminance) if self.previous is None else np.abs(luminance - self.previous)
        self.previous = luminance
        return np.clip(self.config.luminance_gain * luminance + self.config.change_gain * change, 0, 1).astype(np.float32)


def save_debug_artifacts(prefix, frame: np.ndarray, gray: np.ndarray,
                         panorama: np.ndarray, azimuth: np.ndarray,
                         onset_drive: np.ndarray, steady_drive: np.ndarray) -> None:
    """Save spatially ordered inspection images; receptor drive is sorted by azimuth."""
    prefix = str(prefix)
    Image.fromarray(frame, "RGBA").save(prefix + "-original.png")
    Image.fromarray(np.rint(gray * 255).astype(np.uint8), "L").save(prefix + "-grayscale.png")
    strip = np.tile(np.rint(panorama * 255).astype(np.uint8), (48, 1))
    Image.fromarray(strip, "L").save(prefix + "-panorama.png")
    order = np.argsort(azimuth, kind="stable")
    drives = np.vstack([np.tile(onset_drive[order], (24, 1)),
                        np.tile(steady_drive[order], (24, 1))])
    Image.fromarray(np.rint(drives * 255).astype(np.uint8), "L").save(prefix + "-receptor-drive.png")
