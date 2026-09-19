"""Shared Experiment 7 metadata, encoding, and spike aggregation utilities."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

from .spatiotemporal import SpatiotemporalConfig, SpatialProjection, spatial_grid


MOTOR_TYPES = ("DNa02", "DNg100", "MDN", "DNp01")
E7_CONFIG = SpatiotemporalConfig(frames=1, neural_steps_per_frame=10)


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def resolve_motor_populations(brain, data_dir: Path, cell_types: Iterable[str] = MOTOR_TYPES) -> tuple[dict, dict[str, np.ndarray]]:
    meta = np.load(data_dir / "brain.npz")
    if not np.array_equal(meta["cell_type"], brain.cell_type):
        raise RuntimeError("FlyBrain and brain.npz metadata order differs")
    requested = tuple(cell_types)
    if len(set(requested)) != len(requested): raise ValueError("duplicate motor cell type")
    populations, records = {}, []
    for typ in requested:
        idx = brain.cells([typ]).astype(np.int32)
        if not len(idx) or not np.all(brain.superclass[idx] == "descending_neuron"):
            raise RuntimeError(f"Missing descending-neuron metadata for {typ}")
        populations[typ] = idx
        for i in idx:
            records.append({"cell_type": typ, "flywire_malecns_id": int(meta["ids"][i]),
                            "brain_index": int(i), "side": str(brain.side[i]),
                            "superclass": str(brain.superclass[i]), "count": int(len(idx))})
    return {"model": "MaleCNS v1.0", "neurons": records,
            "counts": {k: int(len(v)) for k, v in populations.items()}}, populations


def verify_motor_neurons(brain, data_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    return resolve_motor_populations(brain, data_dir, MOTOR_TYPES)


class OnlineE3Encoder:
    """Experiment 3 spatial encoder with explicit one-frame temporal history."""
    def __init__(self, projection: SpatialProjection, config: SpatiotemporalConfig = E7_CONFIG):
        self.projection, self.config, self.previous = projection, config, None

    def reset(self) -> None:
        self.previous = None

    def encode(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        _, grid = spatial_grid(frame, self.config)
        darkness = np.clip(1.0 - grid, 0, 1).astype(np.float32)
        change = np.zeros_like(grid) if self.previous is None else np.abs(grid - self.previous)
        self.previous = grid.copy()
        vector = self.projection.encode(darkness, change)
        return vector, {"grid_sha256": sha256_array(grid), "grid_mean": float(grid.mean()),
                        "grid_std": float(grid.std()), "change_mean": float(change.mean()),
                        "encoder_sha256": sha256_array(vector), "encoder_mean": float(vector.mean()),
                        "encoder_nonzero": int(np.count_nonzero(vector))}


def candidate_slots(brain, populations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {typ: np.isin(np.arange(brain.n), idx) for typ, idx in populations.items()}


def run_neural_window(brain, projection: SpatialProjection, vector: np.ndarray | None,
                      populations: dict[str, np.ndarray], steps: int = 10,
                      cache: dict | None = None) -> tuple[dict, dict]:
    per_cell = {typ: np.zeros(len(idx), np.int32) for typ, idx in populations.items()}
    cache = {} if cache is None else cache
    slots = cache.setdefault("motor_slots", {})
    if not slots:
        for typ, idx in populations.items():
            slot = np.full(brain.n, -1, np.int32); slot[idx] = np.arange(len(idx)); slots[typ] = slot
    vp = {f"{kind}_{side}": np.zeros(len(idx), np.int32)
          for (kind, side), idx in projection.populations.items()}
    vp_slots = cache.setdefault("vp_slots", {})
    if not vp_slots:
        for key, idx in projection.populations.items():
            slot = np.full(brain.n, -1, np.int32); slot[idx] = np.arange(len(idx)); vp_slots[key] = slot
    inject = () if vector is None else projection.injection_pairs(vector)
    for _ in range(steps):
        fired = brain.step(inject=inject)
        for typ, slot in slots.items():
            hit = slot[fired]; hit = hit[hit >= 0]
            per_cell[typ] += np.bincount(hit, minlength=len(per_cell[typ])).astype(np.int32)
        for key, slot in vp_slots.items():
            hit = slot[fired]; hit = hit[hit >= 0]
            vp[f"{key[0]}_{key[1]}"] += np.bincount(hit, minlength=len(vp[f"{key[0]}_{key[1]}"])).astype(np.int32)
    seconds = steps * float(brain.dt)
    rates = {}
    for typ, counts in per_cell.items():
        for side in ("L", "R"):
            mask = brain.side[populations[typ]] == side
            rates[f"{typ}_{side}"] = float(counts[mask].sum() / seconds)
        rates[typ] = float(counts.sum() / seconds)
    controller_rates = dict(rates)
    return ({"per_cell_counts": {k: v.tolist() for k, v in per_cell.items()},
             "group_rates_hz": rates, "controller_rates_hz": controller_rates},
            {k: int(v.sum()) for k, v in vp.items()})


def synthetic_frame(kind: str) -> np.ndarray:
    gray = np.full((144, 160), 192, np.uint8)
    if kind == "left": gray[:, :80] = 16
    elif kind == "right": gray[:, 80:] = 16
    elif kind == "uniform": gray[:] = 128
    else: raise ValueError(kind)
    rgba = np.empty((144, 160, 4), np.uint8)
    rgba[..., :3] = gray[..., None]; rgba[..., 3] = 255
    return rgba
