"""Anatomy-only geometry manipulations of the Mi1 / Mi4 inputs to T4 (Experiment 23).

Experiment 22 found that, under a fixed adapting background, T4 direction selectivity
needs Mi1 excitation, Mi4 inhibition, tonic excitation and some temporal persistence, but
not a slower Mi4. The candidate explanation is a *spatially offset inhibitory veto*: Mi1
and Mi4 sample different screen positions, so one motion direction overlaps excitation
with inhibition more than the other.

This module changes only *where* each T4 neuron's Mi4 inputs sit, never their weights,
number, sign or timing. For neuron i with |w|-weighted screen centroids c1 (Mi1) and c4
(Mi4), every Mi4 input column p is translated rigidly and re-snapped to the nearest
unused retina column of the same eye:

* reversal:    p -> p - 2 (c4 - c1)     (the Mi1->Mi4 offset vector is negated)
* co-location: p -> p - (c4 - c1)       (the Mi4 centroid lands on the Mi1 centroid)

Rigid translation keeps the Mi4 field's shape and spread; the only change is its
position relative to Mi1. Everything is computed from MaleCNS weights and the E5 retina
column positions; no response, label or direction metric is used. Coordinates are
screen pixels (u * (W - 1), v * (H - 1)), matching how stimuli are sampled.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from flymon import column_motion as cm

DIRECTION_VECTORS = {"right": (1.0, 0.0), "left": (-1.0, 0.0),
                     "down": (0.0, 1.0), "up": (0.0, -1.0)}
KNN = 12


def column_pixels(col_uv):
    """(n_cols, 2) normalised screen uv -> pixel coordinates."""
    return np.asarray(col_uv, float) * np.array([cm.W - 1, cm.H - 1], float)


def column_sides(colindex):
    """Eye side ('L'/'R') of every column index."""
    sides = np.empty(len(colindex), dtype=object)
    for (side, _h1, _h2), i in colindex.items():
        sides[i] = side
    return sides


def weighted_centroids(mat, xy):
    """|w|-weighted centroid of each row's input columns; NaN for empty rows."""
    a = abs(mat).tocsr()
    w = np.asarray(a.sum(axis=1)).ravel()
    c = a @ xy
    out = np.full((mat.shape[0], 2), np.nan)
    ok = w > 0
    out[ok] = c[ok] / w[ok, None]
    return out, w


def column_spacing(xy, sides):
    """Median nearest-neighbour distance between retina columns (pixels)."""
    d = []
    for s in np.unique(sides):
        pts = xy[sides == s]
        if len(pts) > 1:
            dist, _ = cKDTree(pts).query(pts, k=2)
            d.append(dist[:, 1])
    return float(np.median(np.concatenate(d)))


def translate_rows(mat, shifts, xy, sides):
    """Rigidly translate each row's inputs by shifts[i] and snap to retina columns.

    Each translated input goes to the nearest column of the same eye not already used by
    that row, so the input count, every weight and the row total are preserved exactly.
    Returns (new csr matrix, per-input snap error in pixels)."""
    mat = mat.tocsr()
    trees = {s: (cKDTree(xy[sides == s]), np.flatnonzero(sides == s)) for s in np.unique(sides)}
    rows, cols, vals, errs = [], [], [], []
    for i in range(mat.shape[0]):
        lo, hi = mat.indptr[i], mat.indptr[i + 1]
        idx, dat = mat.indices[lo:hi], mat.data[lo:hi]
        if not len(idx):
            continue
        shift = shifts[i] if np.all(np.isfinite(shifts[i])) else np.zeros(2)
        used = set()
        order = np.argsort(-np.abs(dat), kind="stable")          # heaviest inputs placed first
        for k in order:
            j, v = idx[k], dat[k]
            tree, members = trees[sides[j]]
            target = xy[j] + shift
            kk = min(KNN, len(members))
            dist, nn = tree.query(target, k=kk)
            dist, nn = np.atleast_1d(dist), np.atleast_1d(nn)
            pick = next(((members[n], d) for n, d in zip(nn, dist) if members[n] not in used), None)
            if pick is None:                                      # extremely crowded: widen search
                dist, nn = tree.query(target, k=len(members))
                pick = next((members[n], d) for n, d in zip(nn, dist) if members[n] not in used)
            used.add(pick[0])
            rows.append(i); cols.append(pick[0]); vals.append(v); errs.append(pick[1])
    new = sparse.csr_matrix((vals, (rows, cols)), shape=mat.shape)
    return new, np.asarray(errs, float)


def offsets(proj_sub, xy):
    """Per-neuron Mi1 and Mi4 centroids and the Mi1->Mi4 offset vector (pixels)."""
    c1, w1 = weighted_centroids(proj_sub["Mi1"], xy)
    c4, w4 = weighted_centroids(proj_sub["Mi4"], xy)
    return c1, c4, c4 - c1, w1, w4


def transform(proj, xy, sides, mode, subtypes=("T4a", "T4b", "T4c", "T4d")):
    """Return (new projection dict, per-subtype diagnostics) with Mi4 moved.

    mode: 'native' (copy), 'reversed' or 'colocated'. Only the Mi4 matrices of the
    listed subtypes change; every other class and subtype is shared unchanged."""
    factor = {"native": 0.0, "colocated": 1.0, "reversed": 2.0}[mode]
    new = {s: dict(m) for s, m in proj.items()}
    diag = {}
    for s in subtypes:
        if s not in proj or "Mi4" not in proj[s] or "Mi1" not in proj[s]:
            continue
        c1, c4, d, _w1, _w4 = offsets(proj[s], xy)
        shifts = -factor * np.nan_to_num(d)
        if mode == "native":
            new[s]["Mi4"] = proj[s]["Mi4"]
            errs = np.zeros(proj[s]["Mi4"].nnz)
        else:
            new[s]["Mi4"], errs = translate_rows(proj[s]["Mi4"], shifts, xy, sides)
        _c1, c4n, dn, _, _ = offsets(new[s], xy)
        diag[s] = dict(offset_before=d, offset_after=dn, snap_error=errs)
    return new, diag


def sign_converted(proj, cls="Mi4", subtypes=("T4a", "T4b", "T4c", "T4d")):
    """Flip the synaptic sign of one class (inhibitory -> excitatory); geometry untouched."""
    new = {s: dict(m) for s, m in proj.items()}
    for s in subtypes:
        if s in proj and cls in proj[s]:
            m = proj[s][cls].copy()
            m.data = -m.data
            new[s][cls] = m
    return new


def axis_projection(d, directions):
    """Project offset vectors onto each neuron's anatomy-predicted screen direction."""
    u = np.array([DIRECTION_VECTORS[x] for x in directions], float)
    return np.einsum("ij,ij->i", d, u)
