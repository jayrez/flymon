"""Experiment 8 baseline-relative, DNa02-only directional BCI.

Only DNa02 rates and a frozen pretrial baseline enter the controller. Pixels,
encoder values, labels, game state, and RAM are outside this interface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SteeringBaseline:
    left_mean_hz: float
    right_mean_hz: float
    left_variance_hz2: float
    right_variance_hz2: float
    difference_mean_hz: float
    difference_variance_hz2: float
    windows: int


@dataclass(frozen=True)
class BaselineSteeringConfig:
    threshold_multiplier: float
    sd_floor_hz: float
    left_evidence_side: str = "L"
    cooldown_decisions: int = 1

    def __post_init__(self) -> None:
        if self.threshold_multiplier <= 0 or self.sd_floor_hz <= 0:
            raise ValueError("threshold multiplier and SD floor must be positive")
        if self.left_evidence_side not in ("L", "R"):
            raise ValueError("left_evidence_side must be L or R")
        if self.cooldown_decisions < 0:
            raise ValueError("cooldown cannot be negative")


def estimate_baseline(windows: list[Mapping[str, float]]) -> SteeringBaseline:
    if len(windows) < 2:
        raise ValueError("at least two baseline windows required")
    left = np.asarray([float(x["DNa02_L"]) for x in windows])
    right = np.asarray([float(x["DNa02_R"]) for x in windows])
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("baseline rates must be finite")
    difference = left - right
    return SteeringBaseline(float(left.mean()), float(right.mean()),
        float(left.var(ddof=1)), float(right.var(ddof=1)),
        float(difference.mean()), float(difference.var(ddof=1)), len(windows))


class BaselineRelativeSteeringController:
    """Frozen-baseline LEFT/RIGHT/NONE threshold decoder."""
    def __init__(self, baseline: SteeringBaseline, config: BaselineSteeringConfig,
                 swap_mapping: bool = False):
        self.baseline, self.config, self.swap_mapping = baseline, config, swap_mapping
        self.threshold_hz = config.threshold_multiplier * max(
            math.sqrt(max(0.0, baseline.difference_variance_hz2)), config.sd_floor_hz)
        self.reset()

    def reset(self) -> None:
        self.cooldown_remaining = 0

    def decode(self, rates_hz: Mapping[str, float]) -> tuple[str | None, dict]:
        left, right = float(rates_hz["DNa02_L"]), float(rates_hz["DNa02_R"])
        if not math.isfinite(left) or not math.isfinite(right) or min(left, right) < 0:
            raise ValueError("DNa02 rates must be finite and nonnegative")
        delta_l, delta_r = left - self.baseline.left_mean_hz, right - self.baseline.right_mean_hz
        anatomical = delta_l - delta_r
        signal = anatomical if self.config.left_evidence_side == "L" else -anatomical
        raw_action = "LEFT" if signal > self.threshold_hz else "RIGHT" if signal < -self.threshold_hz else None
        if self.swap_mapping and raw_action is not None:
            raw_action = "RIGHT" if raw_action == "LEFT" else "LEFT"
        if self.cooldown_remaining:
            action, reason = None, "cooldown"
            self.cooldown_remaining -= 1
        else:
            action, reason = raw_action, "threshold" if raw_action else "deadband"
            if action: self.cooldown_remaining = self.config.cooldown_decisions
        return action, {"raw_left_hz": left, "raw_right_hz": right,
            "delta_left_hz": delta_l, "delta_right_hz": delta_r,
            "anatomical_L_minus_R_hz": anatomical, "steering_signal_hz": signal,
            "threshold_hz": self.threshold_hz, "raw_action": raw_action,
            "selected_action": action, "reason": reason,
            "cooldown_remaining": self.cooldown_remaining,
            "swap_mapping": self.swap_mapping, "baseline": asdict(self.baseline)}
