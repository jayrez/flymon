"""Fixed, inspectable descending-neuron to Game Boy directional BCI.

The controller deliberately has no framebuffer, game-state, label, or RAM API.
It consumes only aggregate candidate-DN firing rates for one decision window.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping


ACTIONS = ("LEFT", "RIGHT", "UP", "DOWN")


@dataclass(frozen=True)
class ControllerConfig:
    smoothing_tau_s: float = 0.15
    decision_duration_s: float = 0.20
    steering_deadband_hz: float = 1.0
    forward_threshold_hz: float = 1.0
    backward_threshold_hz: float = 1.0
    minimum_activity_hz: float = 1.0
    minimum_hold_decisions: int = 1
    cooldown_decisions: int = 1
    # Calibration assigns which anatomical side raises screen-LEFT evidence.
    left_evidence_side: str = "L"

    def __post_init__(self) -> None:
        if self.smoothing_tau_s <= 0 or self.decision_duration_s <= 0:
            raise ValueError("positive timing values required")
        if self.left_evidence_side not in ("L", "R"):
            raise ValueError("left_evidence_side must be L or R")
        if min(self.steering_deadband_hz, self.forward_threshold_hz,
               self.backward_threshold_hz, self.minimum_activity_hz) < 0:
            raise ValueError("thresholds cannot be negative")
        if self.minimum_hold_decisions < 1 or self.cooldown_decisions < 0:
            raise ValueError("invalid hold/cooldown")


class FixedDNController:
    """Exponential smoothing plus fixed threshold competition.

    Expected keys are DNa02_L/R, DNg100, and MDN. DNp01 is accepted and
    reported in raw activity but intentionally does not map to a button.
    """

    required = frozenset({"DNa02_L", "DNa02_R", "DNg100", "MDN"})

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.smoothed = {key: 0.0 for key in (*self.required, "DNp01")}
        self.last_action: str | None = None
        self.hold_remaining = 0
        self.cooldown_remaining = 0

    def decode(self, rates_hz: Mapping[str, float]) -> tuple[str | None, dict]:
        missing = self.required - rates_hz.keys()
        if missing:
            raise KeyError(f"missing DN rates: {sorted(missing)}")
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in rates_hz.values()):
            raise ValueError("DN rates must be finite and nonnegative")
        alpha = 1.0 - math.exp(-self.config.decision_duration_s /
                               self.config.smoothing_tau_s)
        for key in self.smoothed:
            raw = float(rates_hz.get(key, 0.0))
            self.smoothed[key] += alpha * (raw - self.smoothed[key])

        if self.hold_remaining > 0 and self.last_action is not None:
            self.hold_remaining -= 1
            return self.last_action, self.snapshot("minimum_hold")
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.last_action = None
            return None, self.snapshot("cooldown")

        left_side = self.config.left_evidence_side
        right_side = "R" if left_side == "L" else "L"
        steer = self.smoothed[f"DNa02_{left_side}"] - self.smoothed[f"DNa02_{right_side}"]
        scores = {
            "LEFT": max(0.0, steer - self.config.steering_deadband_hz),
            "RIGHT": max(0.0, -steer - self.config.steering_deadband_hz),
            "UP": max(0.0, self.smoothed["DNg100"] - self.config.forward_threshold_hz),
            "DOWN": max(0.0, self.smoothed["MDN"] - self.config.backward_threshold_hz),
        }
        if max(self.smoothed.values()) < self.config.minimum_activity_hz or max(scores.values()) <= 0:
            self.last_action = None
            return None, self.snapshot("deadband", scores)
        # Fixed tie order gives steering precedence, then forward, then backward.
        action = max(ACTIONS, key=lambda a: (scores[a], -ACTIONS.index(a)))
        self.last_action = action
        self.hold_remaining = self.config.minimum_hold_decisions - 1
        self.cooldown_remaining = self.config.cooldown_decisions
        return action, self.snapshot("selected", scores)

    def snapshot(self, reason: str, scores: Mapping[str, float] | None = None) -> dict:
        return {
            "smoothed_rates_hz": dict(self.smoothed),
            "scores": dict(scores or {}),
            "reason": reason,
            "last_action": self.last_action,
            "hold_remaining": self.hold_remaining,
            "cooldown_remaining": self.cooldown_remaining,
            "config": asdict(self.config),
        }
