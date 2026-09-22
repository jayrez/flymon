"""Optic-column coordinates and T4/T5 receptive-field geometry (Experiment 18).

Column coordinates come from the released MaleCNS `assignedOlHex1/2` annotations and
are converted to the same axial cartesian frame `flymon/retina.py` already uses
(x = h1 - 0.5*h2, y = sqrt(3)/2*h2). Only *directly annotated* coordinates are used;
nothing here reads a stimulus, a response, or a class label, so the geometry it
produces cannot leak evaluation information.

T4/T5 cells themselves carry no hex annotation in MaleCNS v1.0. That is not a problem
for this measurement: the quantity of interest is the offset vector between the
weighted centroids of two *input arms*, which is translation invariant and therefore
independent of where the postsynaptic cell sits.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Arm membership is fixed from literature + connectome sign, never from results.
# T4 (ON): Mi1 fast; Mi9/Mi4 slow & sign-inverting.  T5 (OFF): Tm1/Tm2 fast; Tm9 slow.
FAST_ARM = {"T4": ("Mi1",), "T5": ("Tm1", "Tm2")}
SLOW_ARM = {"T4": ("Mi9", "Mi4"), "T5": ("Tm9",)}
# Present in the pathway but carrying no hex annotation in v1.0 (documented, excluded):
UNRESOLVED_PARTNERS = ("Tm3", "CT1")
SUBTYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
MIN_PARTNERS = 3


def load_column_coords(brain_npz: Path, annotations: Path):
    """Per-neuron optic-column coordinates: (hex, cartesian), NaN where unannotated."""
    import pyarrow.feather as feather
    meta = np.load(brain_npz)
    ids = meta["ids"]
    table = feather.read_table(annotations, columns=["bodyId", "assignedOlHex1", "assignedOlHex2"])
    body = table["bodyId"].to_numpy()
    h1 = table["assignedOlHex1"].to_numpy(zero_copy_only=False)
    h2 = table["assignedOlHex2"].to_numpy(zero_copy_only=False)
    good = ~(np.isnan(h1) | np.isnan(h2))
    index = {int(b): i for i, b in enumerate(ids)}
    hexes = np.full((len(ids), 2), np.nan)
    for b, a, c in zip(body[good], h1[good], h2[good]):
        i = index.get(int(b))
        if i is not None:
            hexes[i] = (a, c)
    cart = np.column_stack([hexes[:, 0] - 0.5 * hexes[:, 1],
                            np.sqrt(3) / 2 * hexes[:, 1]])
    return hexes, cart


def screen_direction(vector, eye_side):
    """Map an eye-space offset vector to screen (du, dv).

    The Experiment-5 retina places the left eye directly (u grows with x) and mirrors
    the right eye into the display (u shrinks with x); v reverses y for both."""
    dx, dy = float(vector[0]), float(vector[1])
    du = dx if eye_side == "L" else -dx
    return np.array([du, -dy])


def cardinal(du_dv):
    """Nearest cardinal screen direction name for a (du, dv) vector."""
    du, dv = du_dv
    if abs(du) >= abs(dv):
        return "right" if du >= 0 else "left"
    return "down" if dv >= 0 else "up"


def circular_stats(angles):
    """Circular mean (degrees, 0-360) and resultant concentration R in [0,1]."""
    if len(angles) == 0:
        return float("nan"), 0.0
    z = np.mean(np.exp(1j * np.asarray(angles)))
    return float(np.degrees(np.angle(z)) % 360.0), float(np.abs(z))


@dataclass
class ReceptiveField:
    brain_index: int
    flywire_id: int
    cell_type: str
    side: str
    n_fast: int
    n_slow: int
    fast_weight: float
    slow_weight: float
    fast_centroid: tuple
    slow_centroid: tuple
    offset: tuple            # eye-space (fast - slow)
    angle: float             # radians, eye space
    magnitude: float
    screen_offset: tuple
    predicted_direction: str


def audit_receptive_fields(W, cell_type, side, cart, ids, subtypes=SUBTYPES,
                           min_partners=MIN_PARTNERS, column_permutation=None):
    """Per-neuron fast/slow input geometry for each T4/T5 subtype.

    `column_permutation` optionally remaps presynaptic coordinates (the geometry
    ablation): a dict {cell_type: permuted index array} applied to partner columns."""
    ct = cell_type.astype(str)
    sd = side.astype(str)
    coords = cart if column_permutation is None else _permuted(cart, ct, column_permutation)
    resolved = ~np.isnan(coords[:, 0])
    out = {}
    for sub in subtypes:
        family = sub[:2]
        fast_types = set(FAST_ARM[family])
        slow_types = set(SLOW_ARM[family])
        targets = np.flatnonzero(ct == sub)
        records = []
        total_weight = 0.0
        resolved_weight = 0.0
        for i in targets:
            row = W[i].tocoo()
            fc, fw, sc, sw = [], [], [], []
            for c, v in zip(row.col, row.data):
                total_weight += abs(float(v))
                tc = ct[c]
                if not resolved[c] or (tc not in fast_types and tc not in slow_types):
                    continue
                resolved_weight += abs(float(v))
                if tc in fast_types and v > 0:
                    fc.append(coords[c]); fw.append(abs(float(v)))
                elif tc in slow_types:
                    sc.append(coords[c]); sw.append(abs(float(v)))
            if len(fc) < min_partners or len(sc) < min_partners:
                continue
            f = np.average(np.asarray(fc), axis=0, weights=fw)
            s = np.average(np.asarray(sc), axis=0, weights=sw)
            v = f - s
            scr = screen_direction(v, sd[i])
            records.append(ReceptiveField(
                brain_index=int(i), flywire_id=int(ids[i]), cell_type=sub, side=str(sd[i]),
                n_fast=len(fc), n_slow=len(sc), fast_weight=float(sum(fw)), slow_weight=float(sum(sw)),
                fast_centroid=tuple(map(float, f)), slow_centroid=tuple(map(float, s)),
                offset=tuple(map(float, v)), angle=float(np.arctan2(v[1], v[0])),
                magnitude=float(np.linalg.norm(v)), screen_offset=tuple(map(float, scr)),
                predicted_direction=cardinal(scr)))
        out[sub] = dict(records=records, n_total=int(len(targets)), n_eligible=len(records),
                        resolved_weight_fraction=float(resolved_weight / total_weight) if total_weight else 0.0)
    return out


def _permuted(cart, ct, permutation):
    coords = cart.copy()
    for typ, order in permutation.items():
        idx = np.flatnonzero(ct == typ)
        if len(idx) == len(order):
            coords[idx] = cart[idx][order]
    return coords


def shuffle_columns(ct, types, rng):
    """Geometry ablation: permute column assignments *within* each cell type, which
    preserves the marginal distribution of columns but destroys which partner sits
    where relative to the postsynaptic cell."""
    return {t: rng.permutation(int(np.sum(ct.astype(str) == t))) for t in types}


def subtype_summary(audit):
    """Circular summary per subtype, plus screen-direction prediction."""
    summary = {}
    for sub, entry in audit.items():
        recs = entry["records"]
        if not recs:
            summary[sub] = dict(n_eligible=0)
            continue
        angles = np.array([r.angle for r in recs])
        mean_deg, R = circular_stats(angles)
        mags = np.array([r.magnitude for r in recs])
        by_side = {}
        for s in ("L", "R"):
            a = [r.angle for r in recs if r.side == s]
            if a:
                m, rr = circular_stats(a)
                by_side[s] = dict(n=len(a), mean_angle_deg=m, concentration=rr)
        votes = {}
        for r in recs:
            votes[r.predicted_direction] = votes.get(r.predicted_direction, 0) + 1
        summary[sub] = dict(
            n_total=entry["n_total"], n_eligible=entry["n_eligible"],
            eligible_fraction=entry["n_eligible"] / max(entry["n_total"], 1),
            resolved_weight_fraction=entry["resolved_weight_fraction"],
            mean_angle_deg=mean_deg, concentration=R,
            mean_magnitude=float(mags.mean()), median_magnitude=float(np.median(mags)),
            by_side=by_side, predicted_direction_votes=votes,
            predicted_direction=max(votes, key=votes.get))
    return summary
