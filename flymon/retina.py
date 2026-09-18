"""Inferred two-dimensional MaleCNS R1-R6 retinal geometry.

The contact-weighted column inference, axial-grid embedding, overlapping eye
viewports, and bilinear sampling follow DoomFly (nftechie/doomfly, MIT),
independently recomputed here from the local MaleCNS v1.0 source tables.
The display projection is an experimental interface, not a measured eye pose.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np


@dataclass(frozen=True)
class Receptor:
    flywire_id: int
    brain_index: int
    visual_index: int
    eye_side: str
    h1: float
    h2: float
    x: float
    y: float
    u: float
    v: float
    mapping_confidence: float
    selected_column_weight: int
    total_eligible_weight: int


def infer_retina(brain_npz: Path, annotations: Path, connections: Path) -> tuple[list[Receptor], dict]:
    """Derive R1-R6 columns from all outgoing contacts onto annotated L1/L2/L3."""
    import pyarrow.compute as pc
    import pyarrow.feather as feather

    started = time.perf_counter()
    meta = np.load(brain_npz)
    ids, types, visual = meta["ids"], meta["cell_type"].astype(str), meta["visual"]
    receptor_visual_slots = np.flatnonzero(types[visual] == "R1-6")
    receptor_indices = visual[receptor_visual_slots]
    receptor_ids = ids[receptor_indices]

    ann = feather.read_table(annotations, columns=["bodyId", "type", "rootSide", "somaSide",
                                                   "assignedOlHex1", "assignedOlHex2"])
    body = ann["bodyId"].to_numpy()
    typ = np.asarray(ann["type"].to_pylist(), dtype=object)
    h1 = ann["assignedOlHex1"].to_numpy(zero_copy_only=False)
    h2 = ann["assignedOlHex2"].to_numpy(zero_copy_only=False)
    root = np.asarray(ann["rootSide"].to_pylist(), dtype=object)
    soma = np.asarray(ann["somaSide"].to_pylist(), dtype=object)
    anchors_mask = np.isin(typ, ["L1", "L2", "L3"]) & ~np.isnan(h1) & ~np.isnan(h2)
    anchor_ids = body[anchors_mask]
    anchor_lookup = {int(b): (float(a), float(c)) for b, a, c in zip(anchor_ids, h1[anchors_mask], h2[anchors_mask])}
    ann_side = {int(b): str(r or s or "").upper() for b, r, s in zip(body, root, soma)}

    edge = feather.read_table(connections, columns=["body_pre", "body_post", "weight"], memory_map=True)
    # Arrow filtering avoids converting the complete 151M-edge table to NumPy.
    receptor_set = __import__("pyarrow").array(receptor_ids)
    anchor_set = __import__("pyarrow").array(anchor_ids)
    edge = edge.filter(pc.and_(pc.is_in(edge["body_pre"], value_set=receptor_set),
                               pc.is_in(edge["body_post"], value_set=anchor_set)))
    pre = edge["body_pre"].to_numpy(); post = edge["body_post"].to_numpy(); weight = edge["weight"].to_numpy()
    counts: dict[int, dict[tuple[float, float], int]] = {}
    for a, b, w in zip(pre, post, weight):
        col = anchor_lookup[int(b)]
        by_col = counts.setdefault(int(a), {})
        by_col[col] = by_col.get(col, 0) + int(w)

    raw = []
    id_to_index = {int(b): i for i, b in enumerate(ids)}
    visual_by_index = {int(i): k for k, i in enumerate(visual)}
    for rid in receptor_ids:
        by_col = counts.get(int(rid))
        if not by_col:
            continue
        selected = max(by_col, key=by_col.get)
        total = sum(by_col.values())
        bi = id_to_index[int(rid)]
        raw.append(dict(flywire_id=int(rid), brain_index=bi, visual_index=visual_by_index[bi],
                        eye_side=ann_side[int(rid)], h1=selected[0], h2=selected[1],
                        x=selected[0] - .5 * selected[1], y=math.sqrt(3) / 2 * selected[1],
                        mapping_confidence=by_col[selected] / total,
                        selected_column_weight=by_col[selected], total_eligible_weight=total))
    for side in ("L", "R"):
        group = [r for r in raw if r["eye_side"] == side]
        xy = np.asarray([(r["x"], r["y"]) for r in group], np.float64)
        z = (xy - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-12)
        for r, q in zip(group, z):
            r["u"] = .60 * q[0] if side == "L" else .40 + .60 * (1 - q[0])
            r["v"] = 1 - q[1]
    records = [Receptor(**r) for r in raw if r["eye_side"] in ("L", "R")]
    confidence = np.asarray([r.mapping_confidence for r in records])
    def digest(path):
        h=hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b""): h.update(chunk)
        return h.hexdigest()
    diagnostics = dict(source_sha256={annotations.name:digest(annotations),connections.name:digest(connections)}, total_visual=int(len(visual)), r1_6_total=int(len(receptor_ids)),
        r7_count=int(np.sum(types[visual] == "R7")), r8_count=int(np.sum(types[visual] == "R8")),
        unresolved_visual=int(np.sum(~np.isin(types[visual], ["R1-6", "R7", "R8"]))),
        r1_6_mapped=len(records), r1_6_unmapped=int(len(receptor_ids) - len(records)),
        left_mapped=sum(r.eye_side == "L" for r in records), right_mapped=sum(r.eye_side == "R" for r in records),
        median_mapping_confidence=float(np.median(confidence)), below_0_8=int(np.sum(confidence < .8)),
        setup_seconds=time.perf_counter() - started,
        method="modal optic column weighted by every R1-R6 to L1/L2/L3 contact",
        projection={"horizontal":"L: 0..0.6; R mirrored into 0.4..1", "vertical":"hex y maximum maps to screen top",
                    "left_viewport":[0, .6], "right_viewport":[.4, 1], "overlap":[.4, .6]})
    return records, diagnostics


def save_mapping(path: Path, records: list[Receptor], diagnostics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"diagnostics": diagnostics, "receptors": [asdict(r) for r in records]}, indent=2) + "\n")


def load_mapping(path: Path) -> tuple[list[Receptor], dict]:
    data = json.loads(path.read_text())
    return [Receptor(**x) for x in data["receptors"]], data["diagnostics"]


def linear_luminance(frame: np.ndarray) -> np.ndarray:
    if frame.shape != (144, 160, 4) or frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGBA (144,160,4), got {frame.shape} {frame.dtype}")
    rgb = frame[..., :3].astype(np.float32) / 255
    rgb = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    return (rgb @ np.asarray([.2126, .7152, .0722], np.float32)).astype(np.float32)


def bilinear_sample(luminance: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = luminance.shape; x = uv[:, 0] * (w - 1); y = uv[:, 1] * (h - 1)
    x0 = x.astype(int); y0 = y.astype(int); x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    dx = x - x0; dy = y - y0
    return (((1-dx)*(1-dy)*luminance[y0,x0] + dx*(1-dy)*luminance[y0,x1] +
             (1-dx)*dy*luminance[y1,x0] + dx*dy*luminance[y1,x1])).astype(np.float32)


def full_eye_drive(visual_count: int, records: list[Receptor], samples: np.ndarray,
                   previous: np.ndarray | None = None, temporal: bool = False,
                   luminance_gain: float = .45, change_gain: float = 1.6) -> np.ndarray:
    values = samples if not temporal else np.clip(luminance_gain*samples + change_gain*(0 if previous is None else np.abs(samples-previous)), 0, 1)
    drive = np.zeros(visual_count, np.float32)
    drive[np.asarray([r.visual_index for r in records])] = np.clip(values, 0, 1)
    return drive


def synthetic_frame(name: str, tick: int = 0) -> np.ndarray:
    y, x = np.mgrid[:144, :160]
    if name == "black": mask = np.zeros((144,160), bool)
    elif name == "white": mask = np.ones((144,160), bool)
    elif name == "left": mask = x < 80
    elif name == "right": mask = x >= 80
    elif name == "vertical": mask = (x // 10) % 2 == 0
    elif name == "horizontal": mask = (y // 9) % 2 == 0
    elif name == "motion_left": mask = ((x + tick * 4) // 10) % 2 == 0
    elif name == "motion_right": mask = ((x - tick * 4) // 10) % 2 == 0
    else: raise ValueError(name)
    rgb = np.repeat((mask * 255).astype(np.uint8)[...,None], 3, axis=2)
    return np.concatenate([rgb, np.full((144,160,1), 255, np.uint8)], axis=2)
