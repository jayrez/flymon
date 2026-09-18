"""Deterministic framebuffer adapter for upstream flybrain.eyes.FeatureDetectors.

A dark connected component supplies object geometry. The biological channel
mapping, gains, clipping, and side selection remain in upstream FlyBrain.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from flybrain.eyes import FeatureDetectors

from .vision import preprocess


@dataclass(frozen=True)
class FeatureConfig:
    dark_threshold: float = 0.5
    min_component_pixels: int = 16


@dataclass
class FeatureObservation:
    gray: np.ndarray
    dark_mask: np.ndarray
    component_mask: np.ndarray
    bbox: tuple[int, int, int, int] | None  # left, top, right-exclusive, bottom-exclusive
    area: int
    center_x: float | None
    dx: float | None
    size: float | None

    @property
    def side(self) -> str | None:
        return None if self.dx is None else ("L" if self.dx < 0 else "R")

    @property
    def opp(self) -> tuple[float, float] | None:
        return None if self.dx is None else (self.dx, self.size)


def extract_component(frame: np.ndarray, config: FeatureConfig = FeatureConfig()) -> FeatureObservation:
    """Preprocess a framebuffer and select its largest dark component."""
    gray, _ = preprocess(frame)
    return extract_from_gray(gray, config)


def extract_from_gray(gray: np.ndarray, config: FeatureConfig = FeatureConfig()) -> FeatureObservation:
    """Select the largest 8-connected dark region using one fixed threshold."""
    if gray.shape != (144, 160) or not np.all(np.isfinite(gray)):
        raise ValueError("Expected finite 144x160 grayscale image")
    dark = gray < config.dark_threshold
    labels, count = ndimage.label(dark, structure=np.ones((3, 3), np.uint8))
    if not count:
        return FeatureObservation(gray, dark, np.zeros_like(dark), None, 0, None, None, None)
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    chosen = int(np.argmax(areas))
    area = int(areas[chosen])
    if area < config.min_component_pixels:
        return FeatureObservation(gray, dark, np.zeros_like(dark), None, 0, None, None, None)
    mask = labels == chosen
    ys, xs = np.nonzero(mask)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    center_x = float(xs.mean())
    return FeatureObservation(gray, dark, mask, (left, top, right, bottom), area,
                              center_x, center_x - 79.5, sqrt(area))


def detector_for(brain, observation: FeatureObservation) -> FeatureDetectors:
    """Create a fresh upstream detector and prime a zero-size prior object.

    The priming call is not applied to the brain. It makes the first real call
    an object-appearance transient; later calls see unchanged geometry.
    """
    detector = FeatureDetectors(brain)
    if observation.opp is not None:
        detector.inject(opp=(observation.dx, 0.0))
    return detector


def target_populations(detector: FeatureDetectors) -> dict[str, np.ndarray]:
    """Return metadata-selected LPLC2 and LC10a populations on both sides."""
    return {"LPLC2": np.concatenate([detector.cells["loom"][s] for s in "LR"]),
            "LC10a": np.concatenate([detector.cells["chase"][s] for s in "LR"])}


def save_feature_artifacts(prefix, frame: np.ndarray, observation: FeatureObservation,
                           onset_amounts: dict[str, float], steady_amounts: dict[str, float]) -> None:
    """Save original, grayscale, component mask, and one map per used class."""
    prefix = str(prefix)
    Image.fromarray(frame, "RGBA").save(prefix + "-original.png")
    Image.fromarray(np.rint(observation.gray * 255).astype(np.uint8), "L").save(prefix + "-grayscale.png")
    Image.fromarray(observation.dark_mask.astype(np.uint8) * 255, "L").save(prefix + "-dark-mask.png")
    Image.fromarray(observation.component_mask.astype(np.uint8) * 255, "L").save(prefix + "-selected-component.png")
    for label, channel, amounts in (("LPLC2", "loom", onset_amounts),
                                    ("LC10a", "chase", steady_amounts)):
        # Only side information is available to upstream FeatureDetectors; this
        # visualization shows the half-screen driven and the source component.
        panel = Image.new("RGB", (160, 144), (0, 0, 0))
        pixels = np.asarray(panel).copy()
        pixels[:, :80, 0] = round(255 * min(1.0, amounts.get(channel + "L", 0.0) / 0.8))
        pixels[:, 80:, 0] = round(255 * min(1.0, amounts.get(channel + "R", 0.0) / 0.8))
        panel = Image.fromarray(pixels, "RGB")
        if observation.bbox:
            draw = ImageDraw.Draw(panel)
            left, top, right, bottom = observation.bbox
            draw.rectangle((left, top, right - 1, bottom - 1), outline=(255, 255, 255), width=1)
        panel.save(prefix + f"-{label}-projection.png")
