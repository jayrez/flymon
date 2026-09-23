"""Experiment 23 analysis: preregistered criteria, category, per-neuron reversal, figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from analyze_tonic_disinhibition_experiment import _svg, bars, boot_ci, histogram, holm, panel_traces, sign_p
from flymon import spatial_veto as sv

ROOT = Path(__file__).resolve().parent
R = ROOT / "results" / "experiment-23-spatial-inhibitory-veto"
CAP = ROOT / "captures" / "experiment-23"
E22R = ROOT / "results" / "experiment-22-fixed-background-adaptation"
T4 = ("T4a", "T4b", "T4c", "T4d")


def load(p):
    return json.loads(p.read_text())


def criteria(h):
    c = h["conditions"]; nat, rev, col = c["native"], c["reversed"], c["colocated"]
    n = nat["t4_mean"]
    st = h["static"]
    crit = dict(
        N_native=bool(n >= 0.15 and nat["t4_positive"] == 4 and nat["t4_on_correct"] == 4),
        R_reversal=bool(rev["t4_mean"] <= -0.10 and rev["t4_negative"] >= 3 and rev["t4_on_correct"] == 4),
        C_colocation=bool(abs(col["t4_mean"]) < 0.05 or (n > 0 and abs(col["t4_mean"]) <= 0.30 * n)),
        G_geometry=bool(h["geometry_null"]["residual_fraction"] is not None
                        and h["geometry_null"]["residual_fraction"] < 0.30),
        I_mi4_inhibition=bool(c["S1_mi4_excitatory"]["t4_mean"] <= 0.5 * n),
        P_persistence=bool(abs(c["T2_no_filtering"]["t4_mean"]) < 0.05
                           or (n > 0 and abs(c["T2_no_filtering"]["t4_mean"]) <= 0.30 * n)),
        S_static=bool(abs(st["native"]["static_t4_dsi"]) <= 0.05 and abs(st["reversed"]["static_t4_dsi"]) <= 0.05
                      and st["native"]["static_t4_on_correct"] == 4))
    vals = dict(native=n, native_positive=nat["t4_positive"], native_on=nat["t4_on_correct"],
                reversed=rev["t4_mean"], reversed_negative=rev["t4_negative"], reversed_on=rev["t4_on_correct"],
                colocated=col["t4_mean"], colocated_reduction=(1 - abs(col["t4_mean"]) / n) if n > 0 else None,
                geometry_residual=h["geometry_null"]["residual_fraction"],
                mi4_excitatory=c["S1_mi4_excitatory"]["t4_mean"], no_filtering=c["T2_no_filtering"]["t4_mean"],
                static_native=st["native"]["static_t4_dsi"], static_reversed=st["reversed"]["static_t4_dsi"])
    return crit, vals


def category(h, crit, v):
    c = h["conditions"]; n = v["native"]
    if n < 0.10 or v["native_positive"] < 3 or v["native_on"] < 4:
        return "E", "FAIL"
    causal = all(crit[k] for k in ("R_reversal", "C_colocation", "G_geometry", "I_mi4_inhibition",
                                   "P_persistence", "S_static"))
    if causal and crit["N_native"]:
        return "A", "PASS"
    if causal and 0.10 <= n < 0.15:
        return "A-partial", "PARTIAL"
    geo_dep = crit["G_geometry"] or crit["C_colocation"]
    if geo_dep and v["reversed"] <= 0.5 * n and not crit["R_reversal"]:
        return "B", "PARTIAL"
    if geo_dep and v["reversed"] > 0.5 * n:
        return "C", "FAIL"
    if c["S2_mi4_removed"]["t4_mean"] <= 0.5 * n and not geo_dep and not crit["R_reversal"]:
        return "D", "FAIL"
    return "E", "FAIL"


def progression(cat, v):
    if cat == "A" and v["reversed_negative"] == 4:
        return "GO"
    if cat == "A" or cat == "A-partial":
        return "CONDITIONAL GO"
    return "NO-GO"


def main():
    import run_fixed_background_experiment as E22
    CAP.mkdir(parents=True, exist_ok=True)
    h = load(R / "heldout-results.json"); gm = load(R / "geometry-metrics.json")
    cal = load(R / "calibration-verification.json")
    z = np.load(R / "heldout-per-neuron.npz")
    crit, v = criteria(h)
    cat, verdict = category(h, crit, v)
    prog = progression(cat, v)

    # ---- per-subtype statistics ----
    stats = {}
    for cond, sign in (("native", 1), ("reversed", -1)):
        rows, ps = {}, []
        for s in T4:
            d, e = z[f"{cond}_{s}_dsi"], z[f"{cond}_{s}_effect"]
            rows[s] = dict(n=int(len(d)), dsi=float(d.mean()), ci=boot_ci(d), p_sign=sign_p(sign * e))
            ps.append(rows[s]["p_sign"])
        for s, a in zip(T4, holm(ps)):
            rows[s]["p_holm"] = float(a)
            rows[s]["significant_expected_sign"] = bool(a < 0.05 and sign * rows[s]["dsi"] > 0)
        stats[cond] = rows

    # ---- per-neuron reversal and spatial-axis analysis (needs anatomy) ----
    ctx, _const = E22.contexts()
    xy = sv.column_pixels(ctx["col_uv"])
    pn = {}
    allN, allR, allM = [], [], []
    for s in T4:
        dn, dr = z[f"native_{s}_dsi"], z[f"reversed_{s}_dsi"]
        en = z[f"native_{s}_effect"]
        recs = ctx["audit"][s]["records"]
        assert len(dn) == len(recs)
        _c1, _c4, d, _w1, _w4 = sv.offsets(ctx["proj"][s], xy)
        has = np.isfinite(d[:, 0])
        proj_ax = sv.axis_projection(np.nan_to_num(d), [r.predicted_direction for r in recs])
        mag = np.linalg.norm(np.nan_to_num(d), axis=1)
        pos = (dn > 0) & (en != 0)
        row = dict(n=int(len(dn)), n_with_mi4=int(has.sum()),
                   pearson=float(pearsonr(dn, dr)[0]), spearman=float(spearmanr(dn, dr)[0]),
                   median_change=float(np.median(dr - dn)),
                   sign_reversal_fraction=float(np.mean(dr[pos] < 0)) if pos.any() else None,
                   sign_reversal_fraction_mi4=float(np.mean(dr[pos & has] < 0)) if (pos & has).any() else None,
                   mean_dsi_with_mi4=float(dn[has].mean()), mean_dsi_without_mi4=float(dn[~has].mean()),
                   reversed_mean_dsi_with_mi4=float(dr[has].mean()),
                   spearman_dsi_vs_projection=float(spearmanr(dn[has], proj_ax[has])[0]),
                   spearman_dsi_vs_magnitude=float(spearmanr(dn[has], mag[has])[0]))
        pn[s] = row
        allN.append(dn); allR.append(dr); allM.append(has)
    N_, R_, M_ = np.concatenate(allN), np.concatenate(allR), np.concatenate(allM)
    pn["all_T4"] = dict(pearson=float(pearsonr(N_, R_)[0]), spearman=float(spearmanr(N_, R_)[0]),
                        median_change=float(np.median(R_ - N_)),
                        sign_reversal_fraction=float(np.mean(R_[N_ > 0] < 0)),
                        sign_reversal_fraction_mi4=float(np.mean(R_[(N_ > 0) & M_] < 0)))
    oc = h["oracle"]
    orc_corr = {s: float(pearsonr(z[f"native_{s}_dsi"], z[f"oracle_native_{s}_dsi"])[0]) for s in T4}
    e22 = load(E22R / "heldout-results.json")
    out_stats = dict(subtypes=stats, per_neuron=pn, oracle_per_neuron_pearson=orc_corr,
                     unit="neuron (deterministic front end)", bootstrap=10000,
                     correction="Holm over the four T4 subtypes, per condition")
    (R / "statistics.json").write_text(json.dumps(out_stats, indent=2) + "\n")
    failed = [k for k, ok in crit.items() if not ok]
    if cat == "A":
        support = "strongly supported"
    elif crit["R_reversal"] and crit["G_geometry"] and crit["I_mi4_inhibition"] and crit["P_persistence"]:
        support = "partially supported (direction reverses with Mi4 geometry; failed: " + ", ".join(failed) + ")"
    elif cat in ("A-partial", "B"):
        support = "partially supported"
    else:
        support = "unsupported"
    ver = dict(scientific_verdict=verdict, mechanistic_category=cat, gameplay_progression=prog,
               criteria=crit, criterion_values=v, failed_criteria=failed,
               category_note=("E reached by fall-through: the preregistered ladder has no rule for "
                              "'all causal criteria met except co-location'" if cat == "E" and v["native"] >= 0.10
                              and v["native_positive"] >= 3 and v["native_on"] == 4 else None),
               spatial_veto_support_assessment=support,
               support_assessment_basis="analyst interpretation, not preregistered; the formal verdict is the "
                                        "preregistered category above")
    (R / "verdict.json").write_text(json.dumps(ver, indent=2) + "\n")

    # ---------------- figures ----------------
    c = h["conditions"]
    sub = "T4b"
    recs = ctx["audit"][sub]["records"]
    geo_rev, _ = sv.transform(ctx["proj"], xy, sv.column_sides(ctx["colindex"]), "reversed", subtypes=(sub,))
    geo_col, _ = sv.transform(ctx["proj"], xy, sv.column_sides(ctx["colindex"]), "colocated", subtypes=(sub,))
    _c1, _c4, dd, _, _ = sv.offsets(ctx["proj"][sub], xy)
    k = int(np.nanargmax(np.where(np.isfinite(dd[:, 0]), np.linalg.norm(np.nan_to_num(dd), axis=1)
                                  * (np.array([r.predicted_direction for r in recs]) == "right"), -1)))

    def field(path, title, maps):
        m1 = ctx["proj"][sub]["Mi1"].tocsr()[k]
        cen = np.average(xy[m1.indices], axis=0, weights=np.abs(m1.data))
        W, H, S = 900, 330, 22
        p = _svg(W, H, title)
        for j, (lab, m4) in enumerate(maps):
            ox = 40 + j * 290; oy = 70
            p.append(f'<text x="{ox}" y="{oy-12}" font-size="12">{lab}</text>')
            p.append(f'<rect x="{ox}" y="{oy}" width="250" height="220" fill="none" stroke="#ccc"/>')
            for col, mat, colr in (("Mi1", m1, "#27ae60"), ("Mi4", m4.tocsr()[k], "#c0392b")):
                for jj, w in zip(mat.indices, mat.data):
                    x = ox + 125 + (xy[jj, 0] - cen[0]) * S / 3.35; y = oy + 110 + (xy[jj, 1] - cen[1]) * S / 3.35
                    r = 3 + 14 * abs(w) / max(abs(m1.data).max(), 1e-9)
                    p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{colr}" fill-opacity="0.45"/>')
            p.append(f'<text x="{ox+4}" y="{oy+214}" font-size="10">green Mi1, red Mi4; preferred = right</text>')
        path.write_text("\n".join(p + ["</svg>"]))

    field(CAP / "01-native-geometry.svg", f"Exp23 native Mi1/Mi4 inputs, example {sub} neuron (screen px, 1 column = 3.35 px)",
          [("native", ctx["proj"][sub]["Mi4"])])
    field(CAP / "02-native-vs-reversed.svg", "Exp23 Mi4 reversal: p -> p - 2(c4 - c1), snapped to retina columns",
          [("native", ctx["proj"][sub]["Mi4"]), ("reversed", geo_rev[sub]["Mi4"])])
    field(CAP / "03-colocation.svg", "Exp23 Mi4 co-location: p -> p - (c4 - c1)",
          [("native", ctx["proj"][sub]["Mi4"]), ("co-located", geo_col[sub]["Mi4"])])
    lab, val = [], []
    for s in T4:
        for mode in ("native", "reversed", "colocated"):
            lab.append(f"{s} {mode}"); val.append(gm["subtypes"][s][mode]["projection_median_px"])
    bars(CAP / "04-spatial-offsets.svg", lab, val,
         "Exp23 median Mi1->Mi4 offset projected on the predicted preferred direction (px)")
    bars(CAP / "05-native-subtype-dsi.svg", list(T4), [stats["native"][s]["dsi"] for s in T4],
         "Exp23 native held-out DSI by T4 subtype", ref=0.15, cis=[stats["native"][s]["ci"] for s in T4])
    bars(CAP / "06-reversed-subtype-dsi.svg", list(T4), [stats["reversed"][s]["dsi"] for s in T4],
         "Exp23 reversed-Mi4 held-out DSI by T4 subtype", ref=-0.10, cis=[stats["reversed"][s]["ci"] for s in T4])
    # 07 scatter
    W, H, L = 520, 520, 60
    lo, hi = -1.0, 1.0; X = lambda a: L + (W - 2 * L) * (a - lo) / (hi - lo); Y = lambda b: H - L - (H - 2 * L) * (b - lo) / (hi - lo)
    p = _svg(W, H, "Exp23 per-neuron DSI: native (x) vs reversed (y), all T4")
    p += [f'<line x1="{X(0)}" y1="{Y(lo)}" x2="{X(0)}" y2="{Y(hi)}" stroke="#999"/>',
          f'<line x1="{X(lo)}" y1="{Y(0)}" x2="{X(hi)}" y2="{Y(0)}" stroke="#999"/>',
          f'<line x1="{X(lo)}" y1="{Y(hi)}" x2="{X(hi)}" y2="{Y(lo)}" stroke="#c0392b" stroke-dasharray="4"/>']
    rng = np.random.default_rng(0)
    sel = rng.choice(len(N_), size=min(2500, len(N_)), replace=False)
    for a, b in zip(N_[sel], R_[sel]):
        p.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="1.6" fill="#275e85" fill-opacity="0.35"/>')
    p.append(f'<text x="{L}" y="{H-20}" font-size="11">Pearson {pn["all_T4"]["pearson"]:+.2f}; '
             f'native&gt;0 that reverse: {pn["all_T4"]["sign_reversal_fraction"]:.2f}</text>')
    (CAP / "07-native-vs-reversed-per-neuron.svg").write_text("\n".join(p + ["</svg>"]))
    bars(CAP / "08-colocation-effect.svg", ["native", "co-located", "reversed"],
         [c["native"]["t4_mean"], c["colocated"]["t4_mean"], c["reversed"]["t4_mean"]],
         "Exp23 held-out T4 mean DSI: native vs co-located vs reversed Mi4")
    histogram(CAP / "09-geometry-null.svg", z["null_samples"], h["geometry_null"]["observed"],
              "Exp23 geometry shuffle null, 200 permutations (held-out speed axis, T4)")
    bars(CAP / "10-mi4-inhibitory-vs-excitatory.svg", ["Mi4 inhibitory (native)", "Mi4 excitatory"],
         [c["native"]["t4_mean"], c["S1_mi4_excitatory"]["t4_mean"]], "Exp23 Mi4 sign conversion, held-out T4 mean DSI")
    bars(CAP / "11-removals.svg", ["native", "Mi4 removed", "Mi1 removed"],
         [c["native"]["t4_mean"], c["S2_mi4_removed"]["t4_mean"], c["S3_mi1_removed"]["t4_mean"]],
         "Exp23 removals, held-out T4 mean DSI (Mi1 removed: amplitude "
         f"{c['S3_mi1_removed']['amplitude']:.2g})")
    tn = ["native", "T1_equal_filters", "T2_no_filtering"] + [k for k in c if k.startswith("T3")] + ["reversed", "T1R_reversed_equal_filters"]
    bars(CAP / "12-temporal-filtering.svg", tn, [c[k]["t4_mean"] for k in tn], "Exp23 temporal tests, held-out T4 mean DSI")
    lab, val = [], []
    for mode in ("native", "reversed", "colocated"):
        for s in T4:
            o = c[mode]["on_off"][s]; lab.append(f"{mode} {s}"); val.append(o["on"] - o["off"])
    bars(CAP / "13-on-off-specificity.svg", lab, val, "Exp23 T4 ON minus OFF mean response (should stay > 0)")
    comp = [("E19 oracle, E19 bank (saved)", e22["oracle_comparison"]["oracle_e19_bank_saved"]["t4_mean"]),
            ("E19 oracle, E23 bank native", oc["native"]["t4_mean"]),
            ("E19 oracle, E23 bank reversed Mi4", oc["reversed"]["t4_mean"]),
            ("E22 strong (R2, E22 bank)", e22["reference_conditions"]["E21_params"]["R2"]["t4_mean"]),
            ("E22 frozen (E22 bank)", e22["heldout"]["summary"]["t4_mean"]),
            ("E23 native", c["native"]["t4_mean"]), ("E23 reversed", c["reversed"]["t4_mean"])]
    bars(CAP / "14-e23-vs-e22-vs-oracle.svg", [a for a, _ in comp], [b for _, b in comp],
         "Exp23 vs E22 vs E19 oracle, held-out T4 mean DSI (oracle is a benchmark only)")
    tz = np.load(R / "traces.npz")
    panel_traces(CAP / "15-pref-null-traces.svg",
                 [(f"{m} {s}", [("pref", tz[f"{m}_{s}_pref"], "#275e85"), ("null", tz[f"{m}_{s}_null"], "#c0392b")])
                  for m in ("native", "reversed") for s in ("T4a", "T4b", "T4c")],
                 "Exp23 held-out ON bars 1.25 px/frame, neuron-averaged output (pref = anatomy-predicted)")

    print(json.dumps(ver, indent=2))
    print(json.dumps(pn, indent=1))
    for cond in ("native", "reversed"):
        for s in T4:
            r = stats[cond][s]
            print(f"{cond} {s} DSI {r['dsi']:+.4f} CI [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p_holm {r['p_holm']:.3g}")
    print("oracle corr", orc_corr)


if __name__ == "__main__":
    main()
