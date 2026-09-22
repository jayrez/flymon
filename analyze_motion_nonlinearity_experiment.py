"""Experiment 19 analysis: mechanism discrimination, gates, verdict, figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "experiment-19-motion-nonlinearity"
CAPTURES = ROOT / "captures" / "experiment-19"
RNG = np.random.default_rng(19072026)
NBOOT = 10000
SUBTYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
E18_MEAN_ABS_DSI = 0.0446
THRESH_DSI, THRESH_SIGN, MIN_COHERENT = 0.15, 0.70, 6


def boot_ci(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 2
    idx = RNG.integers(0, len(x), size=(NBOOT, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def sign_p(effect):
    e = np.asarray(effect, float); r = e[e != 0]
    if len(r) == 0:
        return 1.0
    k = int((r > 0).sum()); n = len(r)
    z = (k - n / 2) / np.sqrt(n / 4)
    from math import erfc
    tail = 0.5 * erfc(abs(z) / np.sqrt(2))
    return float(min(1.0, 2 * tail))


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def bar_svg(path, labels, values, title, ref=None, cis=None):
    from xml.sax.saxutils import escape
    width, height = 900, 70 + len(labels) * 26
    vals = list(values) + [0.0] + ([ref] if ref else [])
    lo, hi = min(vals), max(vals); span = max(hi - lo, 1e-12); x0, xw = 300, 500
    X = lambda v: x0 + xw * (v - lo) / span
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="15" y="24" font-size="15">{escape(title)}</text>',
         f'<line x1="{X(0.0):.1f}" y1="38" x2="{X(0.0):.1f}" y2="{height-10}" stroke="#999"/>']
    if ref:
        p.append(f'<line x1="{X(ref):.1f}" y1="38" x2="{X(ref):.1f}" y2="{height-10}" stroke="#c0392b" stroke-dasharray="4"/>')
    for i, (lb, v) in enumerate(zip(labels, values)):
        y = 42 + i * 26; a, b = sorted([X(0.0), X(v)])
        col = "#275e85" if v >= 0 else "#c0392b"
        p += [f'<text x="292" y="{y+13}" text-anchor="end" font-size="11">{escape(str(lb))}</text>',
              f'<rect x="{a:.1f}" y="{y}" width="{max(b-a,0.5):.1f}" height="16" fill="{col}"/>']
        if cis and cis[i] and np.isfinite(cis[i][0]):
            p.append(f'<line x1="{X(cis[i][0]):.1f}" y1="{y+8}" x2="{X(cis[i][1]):.1f}" y2="{y+8}" stroke="#111"/>')
        p.append(f'<text x="{width-8}" y="{y+13}" font-size="11" text-anchor="end">{v:+.4f}</text>')
    path.write_text("\n".join(p + ["</svg>"]))


def trace_svg(path, series, title):
    from xml.sax.saxutils import escape
    Wd, H, L, T = 880, 340, 60, 46
    n = max(len(v) for _, v, _ in series)
    ymax = max(float(np.max(np.abs(v))) for _, v, _ in series) or 1.0
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{H}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="15" y="24" font-size="15">{escape(title)}</text>',
         f'<line x1="{L}" y1="{H-40}" x2="{Wd-20}" y2="{H-40}" stroke="#333"/>',
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-40}" stroke="#333"/>',
         f'<text x="6" y="{T+10}" font-size="10">{ymax:.3g}</text>',
         f'<text x="6" y="{H-42}" font-size="10">0</text>',
         f'<text x="{L}" y="{H-24}" font-size="10">frame 0</text>',
         f'<text x="{Wd-70}" y="{H-24}" font-size="10">frame {n-1}</text>']
    for k, (name, v, col) in enumerate(series):
        pts = " ".join(f"{L+(Wd-20-L)*i/max(n-1,1):.1f},{(H-40)-(H-40-T)*float(x)/ymax:.1f}"
                       for i, x in enumerate(v))
        p.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.8" points="{pts}"/>')
        p.append(f'<text x="{Wd-230}" y="{T+14+k*15}" font-size="11" fill="{col}">{escape(name)}</text>')
    path.write_text("\n".join(p + ["</svg>"]))


def per_subtype_stats(pn, prefix):
    rows, ps = {}, []
    for s in SUBTYPES:
        kd, ke = f"{prefix}_{s}_dsi", f"{prefix}_{s}_effect"
        if kd not in pn.files:
            continue
        d, e = pn[kd], pn[ke]
        resp = e != 0
        rows[s] = dict(n=int(len(d)), dsi_mean=float(d.mean()), dsi_ci=boot_ci(d),
                       effect_mean=float(e.mean()), responsive_fraction=float(resp.mean()),
                       sign_consistency_responsive=float((e[resp] > 0).mean()) if resp.any() else 0.0,
                       p_sign=sign_p(e))
        ps.append(rows[s]["p_sign"])
    if rows:
        for s, a in zip(list(rows), holm(ps)):
            rows[s]["p_holm"] = float(a)
    return rows


def summarise(rows):
    if not rows:
        return dict(mean_dsi=0.0, mean_abs_dsi=0.0, n_coherent=0, mean_sign=0.0)
    d = [v["dsi_mean"] for v in rows.values()]
    sg = [v["sign_consistency_responsive"] for v in rows.values()]
    return dict(mean_dsi=float(np.mean(d)), mean_abs_dsi=float(np.mean(np.abs(d))),
                mean_sign=float(np.mean(sg)),
                n_coherent=int(sum(1 for x, y in zip(d, sg) if x > 0 and y >= THRESH_SIGN)),
                min_dsi=float(min(d)), all_same_sign=bool(all(x > 0 for x in d) or all(x < 0 for x in d)))


def main():
    CAPTURES.mkdir(parents=True, exist_ok=True)
    cal = json.loads((RESULTS / "calibration-results.json").read_text())
    sel = json.loads((RESULTS / "selected-model.json").read_text())
    held = json.loads((RESULTS / "heldout-results.json").read_text())
    pn = np.load(RESULTS / "heldout-per-neuron.npz")
    opp = None
    op_path = RESULTS / "opponent-signed-per-neuron.npz"
    if op_path.exists():
        opp = np.load(op_path)

    stats = {m: per_subtype_stats(pn, m) for m in ("A_whole_mean", "C_motion_window", "D_event_window")}
    summary = {m: summarise(r) for m, r in stats.items()}
    opp_stats = per_subtype_stats(opp, "A_whole_mean") if opp is not None else {}
    opp_sum = summarise(opp_stats) if opp_stats else held["control_mechanisms"]["OPPONENT_SIGNED"]["summary"]

    # ---- H0: did extending tau rescue N1? ----
    n1 = {k: v for k, v in cal.items() if v["mechanism"]["family"] == "N1_LEXT"}
    h0 = dict(best_extended=max(n1.values(), key=lambda v: v["mean_abs_dsi"])["mean_abs_dsi"],
              best_extended_coherent=max(n1.values(), key=lambda v: v["mean_abs_dsi"])["n_coherent"],
              e18_tau8=next((v["mean_abs_dsi"] for k, v in n1.items() if k.endswith("tf2|ts8")), None),
              rescued=False)
    h0["rescued"] = bool(h0["best_extended"] >= THRESH_DSI and h0["best_extended_coherent"] >= MIN_COHERENT)

    # ---- H1: did a transient exist that averaging hid? ----
    h1 = dict(A=summary["A_whole_mean"], C=summary["C_motion_window"], D=summary["D_event_window"])
    h1["transient_rescue"] = bool(
        (summary["C_motion_window"]["mean_dsi"] >= THRESH_DSI and summary["C_motion_window"]["n_coherent"] >= MIN_COHERENT)
        or (summary["D_event_window"]["mean_dsi"] >= THRESH_DSI and summary["D_event_window"]["n_coherent"] >= MIN_COHERENT))

    # ---- 2x2 mechanism decomposition ----
    cm_ = held["control_mechanisms"]
    decomposition = {
        "signed_product_only": cm_["P_ONLY_SIGNED"]["summary"],
        "signed_opponent": dict(opp_sum),
        "magnitude_product_only": cm_["P_ONLY"]["summary"],
        "magnitude_opponent": cm_["OPPONENT_PRODUCT"]["summary"],
        "e18_hr_formulation": cm_["N5_HR_CONTROL"]["summary"],
        "linear_extended": cm_["N1_LEXT"]["summary"],
    }

    # ---- verdict ----
    prim = summary["A_whole_mean"]
    onoff = held["on_off"]
    onoff_ok = (all(onoff[s]["on"] > onoff[s]["off"] for s in onoff if s.startswith("T4"))
                and all(onoff[s]["off"] > onoff[s]["on"] for s in onoff if s.startswith("T5")))
    bio_pass = (prim["mean_abs_dsi"] >= THRESH_DSI and prim["n_coherent"] >= MIN_COHERENT
                and prim["min_dsi"] > -0.05)
    opp_pass = (opp_sum["mean_abs_dsi"] >= THRESH_DSI and opp_sum["n_coherent"] >= MIN_COHERENT)

    if bio_pass:
        fam = sel["mechanism"]["family"]
        category = {"N1_LEXT": "A", "N2_COINCIDENCE": "C", "N3_SHUNT": "D", "N4_SUBUNIT": "E"}.get(fam, "H")
        verdict = "PASS"
    elif h0["rescued"]:
        verdict, category = "PASS", "A"
    elif h1["transient_rescue"]:
        verdict, category = "PARTIAL", "B"
    elif opp_pass:
        verdict, category = "FAIL", "F"
    else:
        verdict, category = "FAIL", "G"

    gate = dict(verdict=verdict, category=category,
                thresholds=dict(mean_abs_dsi=THRESH_DSI, sign=THRESH_SIGN, min_coherent=MIN_COHERENT),
                selected=sel["selected"], selected_met_calibration=sel["met_preregistered_threshold"],
                primary_metric_A=prim, hypothesis_H0=h0, hypothesis_H1=h1,
                decomposition=decomposition, opponent_signed_passes=bool(opp_pass),
                biological_model_passes=bool(bio_pass), on_off_specificity_ok=bool(onoff_ok),
                e18_reference_mean_abs_dsi=E18_MEAN_ABS_DSI,
                geometry_null=held["ablations"]["geometry_shuffle"],
                proceed_to_experiment20=bool(bio_pass and onoff_ok))
    (RESULTS / "gate-status.json").write_text(json.dumps(gate, indent=2) + "\n")
    (RESULTS / "statistics.json").write_text(json.dumps(
        dict(selected=stats, selected_summary=summary, opponent_signed=opp_stats,
             opponent_signed_summary=opp_sum, decomposition=decomposition), indent=2) + "\n")

    # ---- figures ----
    fam_best = {}
    for k, v in cal.items():
        f = v["mechanism"]["family"]
        if f not in fam_best or v["mean_abs_dsi"] > fam_best[f]["mean_abs_dsi"]:
            fam_best[f] = v
    order = ["N0_E18", "N1_LEXT", "N2_COINCIDENCE", "N3_SHUNT", "N4_SUBUNIT",
             "P_ONLY", "OPPONENT_PRODUCT", "N5_HR_CONTROL"]
    order = [f for f in order if f in fam_best]
    bar_svg(CAPTURES / "model-family-dsi.svg", order, [fam_best[f]["mean_dsi"] for f in order],
            "Exp19 calibration mean DSI by family (signed; controls included)", ref=THRESH_DSI)
    dk = list(decomposition)
    bar_svg(CAPTURES / "mechanism-decomposition.svg", dk,
            [decomposition[k]["mean_dsi"] for k in dk],
            "Exp19 held-out mean DSI: product vs opponent x magnitude vs signed", ref=THRESH_DSI)
    subs = [s for s in SUBTYPES if s in stats["A_whole_mean"]]
    bar_svg(CAPTURES / "subtype-heldout-dsi.svg", subs,
            [stats["A_whole_mean"][s]["dsi_mean"] for s in subs],
            f"Exp19 held-out DSI by subtype, selected {sel['selected']}",
            cis=[stats["A_whole_mean"][s]["dsi_ci"] for s in subs])
    if opp_stats:
        osubs = [s for s in SUBTYPES if s in opp_stats]
        bar_svg(CAPTURES / "opponent-signed-subtype-dsi.svg", osubs,
                [opp_stats[s]["dsi_mean"] for s in osubs],
                "Exp19 held-out DSI by subtype, OPPONENT_SIGNED (control architecture)",
                cis=[opp_stats[s]["dsi_ci"] for s in osubs])
    mk = ["A_whole_mean", "C_motion_window", "D_event_window"]
    bar_svg(CAPTURES / "temporal-metric-comparison.svg", mk,
            [summary[m]["mean_dsi"] for m in mk],
            "Exp19 held-out mean DSI by temporal metric (tests the hidden-transient hypothesis)",
            ref=THRESH_DSI)
    ax = held["by_axis"]
    bar_svg(CAPTURES / "speed-vs-dsi.svg", list(ax),
            [ax[a]["summary"]["mean_dsi"] for a in ax],
            "Exp19 held-out mean DSI by generalisation axis")
    ab = held["ablations"]
    alabels = ["selected (observed)", "temporal_flat", "interaction_ablation", "geometry null mean"]
    avals = [prim["mean_dsi"], ab["temporal_flat"]["summary"]["mean_dsi"],
             ab["interaction_ablation"]["summary"]["mean_dsi"], ab["geometry_shuffle"]["null_mean"]]
    bar_svg(CAPTURES / "ablations.svg", alabels, avals, "Exp19 ablations (held-out mean DSI)")
    oo = [s for s in SUBTYPES if s in onoff]
    bar_svg(CAPTURES / "on-off-specificity.svg", oo,
            [onoff[s]["on"] - onoff[s]["off"] for s in oo],
            "Exp19 ON minus OFF response (T4 should be >0, T5 <0)")
    sens = held.get("resolved_weight_sensitivity", {})
    if sens:
        bar_svg(CAPTURES / "resolved-weight-vs-dsi.svg", list(sens),
                [sens[k]["mean_dsi"] for k in sens],
                "Exp19 mean DSI vs minimum resolved-weight fraction")

    tp = RESULTS / "time-resolved-results.npz"
    if tp.exists():
        z = np.load(tp)
        for fam, col in (("T5a", "#275e85"), ("T4a", "#27ae60")):
            if f"{fam}_pref" in z.files:
                P, N = z[f"{fam}_pref"], z[f"{fam}_null"]
                trace_svg(CAPTURES / f"time-resolved-{fam.lower()}.svg",
                          [("preferred", P, "#275e85"), ("null", N, "#c0392b"),
                           ("preferred - null", P - N, "#111111")],
                          f"Exp19 {fam} time-resolved response, selected model")

    print(f"VERDICT {verdict}  category {category}")
    print(f"  selected {sel['selected']} (met calibration: {sel['met_preregistered_threshold']})")
    print(f"  primary A: meanDSI {prim['mean_dsi']:+.4f} |DSI| {prim['mean_abs_dsi']:.4f} "
          f"coherent {prim['n_coherent']}/8  -> passes {bio_pass}")
    print(f"  H0 extended tau rescued: {h0['rescued']} (best |DSI| {h0['best_extended']:.4f}, "
          f"coherent {h0['best_extended_coherent']}/8)")
    print(f"  H1 transient rescue: {h1['transient_rescue']} "
          f"(C {summary['C_motion_window']['mean_dsi']:+.4f}, D {summary['D_event_window']['mean_dsi']:+.4f})")
    print("  decomposition (held-out mean DSI, coherent/8):")
    for k, v in decomposition.items():
        print(f"    {k:<24} {v['mean_dsi']:+.4f}  {v['n_coherent']}/8")
    g = held["ablations"]["geometry_shuffle"]
    print(f"  geometry null: mean {g['null_mean']:+.4f} sd {g['null_sd']:.4f} "
          f"observed {g['observed']:+.4f} p={g['p_empirical']:.4f}")
    print(f"  ON/OFF specificity ok: {onoff_ok}")


if __name__ == "__main__":
    main()
