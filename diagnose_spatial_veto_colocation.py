"""EXPLORATORY (calibration split only; not preregistered; run after the E23 held-out):
why does co-locating the Mi4 centroid with Mi1 leave a residual *negative* T4 DSI?

Tests whether the residual comes from the Mi1/Mi4 filter asymmetry (equal filters),
from translation-with-snapping itself (a 'null translation' that shifts Mi4 by +d and back
by -d through two snaps), or from fractional shifts along the offset axis (0, 0.5, 1, 1.5, 2
times the offset removed). Held-out data are not touched; nothing here changes E23's verdict.
"""
import json

import numpy as np

import run_fixed_background_experiment as E22
import run_spatial_veto_experiment as R
from flymon import spatial_veto as sv

ctx, const = E22.contexts()
xy = sv.column_pixels(ctx["col_uv"]); sides = sv.column_sides(ctx["colindex"])
sim = E22.make_sim(ctx, const, E22.gray_bank("calibration", 0.5))
c0 = R.candidate()
rows = {}


def shifted(factor):
    new = {s: dict(m) for s, m in ctx["proj"].items()}
    for s in R.T4:
        _c1, _c4, d, _, _ = sv.offsets(ctx["proj"][s], xy)
        new[s]["Mi4"], _ = sv.translate_rows(ctx["proj"][s]["Mi4"], -factor * np.nan_to_num(d), xy, sides)
    return new


def roundtrip():
    new = {s: dict(m) for s, m in ctx["proj"].items()}
    for s in R.T4:
        _c1, _c4, d, _, _ = sv.offsets(ctx["proj"][s], xy)
        a, _ = sv.translate_rows(ctx["proj"][s]["Mi4"], np.nan_to_num(d), xy, sides)
        new[s]["Mi4"], _ = sv.translate_rows(a, -np.nan_to_num(d), xy, sides)
    return new


for f in (0.0, 0.5, 1.0, 1.5, 2.0):
    p = ctx["proj"] if f == 0.0 else shifted(f)
    r, _ = R.evaluate(sim, c0, p)
    rows[f"offset_removed_x{f:g}"] = dict(t4_mean=r["t4_mean"], subtypes={k: v["dsi"] for k, v in r["subtypes"].items()})
    print(f"offset removed x{f:g}: T4 {r['t4_mean']:+.4f}", flush=True)
col = shifted(1.0)
for name, cfg in (("colocated_equal_filters", c0.replace(tau_slow=c0.tau_fast)),
                  ("colocated_common_tau_12", c0.replace(tau_fast=12.0, tau_slow=12.0))):
    r, _ = R.evaluate(sim, cfg, col)
    rows[name] = dict(t4_mean=r["t4_mean"], subtypes={k: v["dsi"] for k, v in r["subtypes"].items()})
    print(f"{name}: T4 {r['t4_mean']:+.4f}", flush=True)
r, _ = R.evaluate(sim, c0, roundtrip())
rows["snap_roundtrip_native"] = dict(t4_mean=r["t4_mean"], subtypes={k: v["dsi"] for k, v in r["subtypes"].items()})
print(f"snap round-trip (native position, two snaps): T4 {r['t4_mean']:+.4f}", flush=True)
(R.RESULTS / "exploratory-colocation-diagnostic.json").write_text(json.dumps(dict(
    label="EXPLORATORY - calibration split only; run after held-out; not preregistered; does not alter the verdict",
    rows=rows), indent=2) + "\n")
