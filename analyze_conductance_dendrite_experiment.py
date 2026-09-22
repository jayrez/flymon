"""Experiment 20 analysis: verdict, oracle comparison, ablations, figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "experiment-20-conductance-dendrite"
E19DIR = ROOT / "results" / "experiment-19-motion-nonlinearity"
CAPTURES = ROOT / "captures" / "experiment-20"
RNG = np.random.default_rng(20072026)
NBOOT = 10000
SUBTYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
THRESH_DSI, MIN_POS = 0.15, 7


def boot_ci(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 2
    idx = RNG.integers(0, len(x), size=(NBOOT, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def sign_p(e):
    e = np.asarray(e, float); r = e[e != 0]
    if len(r) == 0:
        return 1.0
    k = int((r > 0).sum()); n = len(r)
    from math import erfc
    z = (k - n / 2) / np.sqrt(n / 4)
    return float(min(1.0, 2 * 0.5 * erfc(abs(z) / np.sqrt(2))))


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def bar_svg(path, labels, values, title, ref=None, cis=None):
    from xml.sax.saxutils import escape
    width, height = 920, 70 + len(labels) * 26
    vals = list(values) + [0.0] + ([ref] if ref else [])
    lo, hi = min(vals), max(vals); span = max(hi - lo, 1e-12); x0, xw = 330, 470
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
        p += [f'<text x="322" y="{y+13}" text-anchor="end" font-size="11">{escape(str(lb))}</text>',
              f'<rect x="{a:.1f}" y="{y}" width="{max(b-a,0.5):.1f}" height="16" fill="{col}"/>']
        if cis and cis[i] and np.isfinite(cis[i][0]):
            p.append(f'<line x1="{X(cis[i][0]):.1f}" y1="{y+8}" x2="{X(cis[i][1]):.1f}" y2="{y+8}" stroke="#111"/>')
        p.append(f'<text x="{width-8}" y="{y+13}" font-size="11" text-anchor="end">{v:+.4f}</text>')
    path.write_text("\n".join(p + ["</svg>"]))


def trace_svg(path, series, title):
    from xml.sax.saxutils import escape
    Wd, H, L, T = 900, 330, 64, 46
    n = max(len(v) for _, v, _ in series)
    ymax = max(float(np.max(np.abs(v))) for _, v, _ in series) or 1.0
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{H}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="15" y="24" font-size="15">{escape(title)}</text>',
         f'<line x1="{L}" y1="{(H-40+T)//2}" x2="{Wd-20}" y2="{(H-40+T)//2}" stroke="#ddd"/>',
         f'<line x1="{L}" y1="{H-40}" x2="{Wd-20}" y2="{H-40}" stroke="#333"/>',
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-40}" stroke="#333"/>',
         f'<text x="4" y="{T+10}" font-size="10">{ymax:.3g}</text>',
         f'<text x="4" y="{H-42}" font-size="10">-{ymax:.3g}</text>',
         f'<text x="{L}" y="{H-24}" font-size="10">frame 0</text>',
         f'<text x="{Wd-70}" y="{H-24}" font-size="10">frame {n-1}</text>']
    mid = (H - 40 + T) / 2; half = (H - 40 - T) / 2
    for k, (name, v, col) in enumerate(series):
        pts = " ".join(f"{L+(Wd-20-L)*i/max(n-1,1):.1f},{mid-half*float(x)/ymax:.1f}"
                       for i, x in enumerate(v))
        p.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.8" points="{pts}"/>')
        p.append(f'<text x="{Wd-250}" y="{T+14+k*15}" font-size="11" fill="{col}">{escape(name)}</text>')
    path.write_text("\n".join(p + ["</svg>"]))


def hist_svg(path, samples, observed, title):
    from xml.sax.saxutils import escape
    Wd, H, L = 860, 300, 60
    lo = min(float(np.min(samples)), observed); hi = max(float(np.max(samples)), observed)
    span = max(hi - lo, 1e-12)
    counts, edges = np.histogram(samples, bins=30, range=(lo, hi))
    cmax = max(counts.max(), 1)
    X = lambda v: L + (Wd - 20 - L) * (v - lo) / span
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{H}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="15" y="24" font-size="15">{escape(title)}</text>',
         f'<line x1="{L}" y1="{H-40}" x2="{Wd-20}" y2="{H-40}" stroke="#333"/>']
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        h = (H - 80) * c / cmax
        p.append(f'<rect x="{X(e0):.1f}" y="{H-40-h:.1f}" width="{max(X(e1)-X(e0)-1,1):.1f}" '
                 f'height="{h:.1f}" fill="#9bb7cc"/>')
    p += [f'<line x1="{X(observed):.1f}" y1="40" x2="{X(observed):.1f}" y2="{H-40}" stroke="#c0392b" stroke-width="2"/>',
          f'<text x="{X(observed):.1f}" y="36" font-size="11" fill="#c0392b" text-anchor="middle">observed {observed:+.4f}</text>',
          f'<text x="{L}" y="{H-22}" font-size="10">{lo:+.4f}</text>',
          f'<text x="{Wd-60}" y="{H-22}" font-size="10">{hi:+.4f}</text>']
    path.write_text("\n".join(p + ["</svg>"]))


def schematic_svg(path, cfg):
    from xml.sax.saxutils import escape
    p = ['<svg xmlns="http://www.w3.org/2000/svg" width="880" height="330">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<text x="15" y="26" font-size="15">Exp20 conductance dendrite: two compartments, sign set by reversal potential</text>',
         '<rect x="90" y="80" width="200" height="90" rx="10" fill="#eef4f8" stroke="#275e85"/>',
         '<text x="190" y="110" font-size="13" text-anchor="middle">compartment A</text>',
         '<text x="190" y="132" font-size="11" text-anchor="middle">fast arm (Mi1 | Tm1,Tm2)</text>',
         '<text x="190" y="152" font-size="11" text-anchor="middle">V_A  (readout)</text>',
         '<rect x="560" y="80" width="200" height="90" rx="10" fill="#f8f0ee" stroke="#c0392b"/>',
         '<text x="660" y="110" font-size="13" text-anchor="middle">compartment B</text>',
         '<text x="660" y="132" font-size="11" text-anchor="middle">slow arm (Mi9,Mi4 | Tm9)</text>',
         '<text x="660" y="152" font-size="11" text-anchor="middle">V_B</text>',
         f'<line x1="290" y1="125" x2="560" y2="125" stroke="#333" stroke-width="2"/>',
         f'<text x="425" y="116" font-size="12" text-anchor="middle">g_c = {cfg["g_couple"]}</text>',
         '<text x="425" y="146" font-size="10" text-anchor="middle">coupling current</text>',
         '<text x="90" y="205" font-size="12">w &gt; 0 -&gt; g_E &#8805; 0 with E_E = +1 (depolarising)</text>',
         f'<text x="90" y="226" font-size="12">w &lt; 0 -&gt; g_I &#8805; 0 with E_I = {cfg["E_inh"]} (hyperpolarising)</text>',
         '<text x="90" y="247" font-size="12">C dV/dt = -g_L(V-E_L) - g_E(V-E_E) - g_I(V-E_I) - g_c(V-V_other)</text>',
         f'<text x="90" y="268" font-size="12">output = ReLU(V_A - E_L - {cfg["threshold"]})</text>',
         '<text x="90" y="296" font-size="11" fill="#555">T4: fast excitatory, slow inhibitory.  T5: both arms excitatory.  No subtype coefficients.</text>',
         '</svg>']
    path.write_text("\n".join(p))


def main():
    CAPTURES.mkdir(parents=True, exist_ok=True)
    cal = json.loads((RESULTS / "calibration-results.json").read_text())
    sel = json.loads((RESULTS / "selected-config.json").read_text())
    held = json.loads((RESULTS / "heldout-results.json").read_text())
    z = np.load(RESULTS / "heldout-per-neuron.npz")
    e19 = json.loads((E19DIR / "gate-status.json").read_text())

    # per-subtype stats with CIs and Holm
    def stats(prefix):
        rows, ps = {}, []
        for s in SUBTYPES:
            kd, ke = f"{prefix}_{s}_dsi", f"{prefix}_{s}_effect"
            if kd not in z.files:
                continue
            d, e = z[kd], z[ke]
            resp = e != 0
            rows[s] = dict(n=int(len(d)), dsi_mean=float(d.mean()), dsi_ci=boot_ci(d),
                           effect_mean=float(e.mean()),
                           responsive_fraction=float(resp.mean()),
                           sign_consistency_responsive=float((e[resp] > 0).mean()) if resp.any() else 0.0,
                           p_sign=sign_p(e))
            ps.append(rows[s]["p_sign"])
        if rows:
            for s, a in zip(list(rows), holm(ps)):
                rows[s]["p_holm"] = float(a)
        return rows

    cond = stats("cond"); orc = stats("oracle")
    hs = held["heldout"]["summary"]
    osum = held["oracle"]["summary"]

    passed = (hs["mean_dsi"] >= THRESH_DSI and hs["n_positive"] >= MIN_POS
              and hs["t4_mean"] > 0 and hs["t5_mean"] > 0)
    onoff = held["on_off"]
    onoff_ok = (all(onoff[s]["on"] > onoff[s]["off"] for s in onoff if s.startswith("T4"))
                and all(onoff[s]["off"] > onoff[s]["on"] for s in onoff if s.startswith("T5")))
    fam = held["families"]
    improves = fam["C2_TWOCOMP_NL"]["summary"]["mean_dsi"] > fam["C0_SINGLE"]["summary"]["mean_dsi"]
    ab_all = held["ablations"]
    obs = hs["mean_dsi"]

    def rel_change(name):
        return abs(ab_all[name]["summary"]["mean_dsi"] - obs) / (abs(obs) + 1e-12)

    gsh_ = ab_all["geometry_shuffle"]
    # Fraction of the observed effect that survives complete geometry destruction.
    geometry_residual = abs(gsh_["null_mean"]) / (abs(gsh_["observed"]) + 1e-12)
    mechanism = dict(
        arm_swap_relative_change=rel_change("arm_swapped"),
        sign_destruction_relative_change=rel_change("sign_destroyed"),
        inhibition_neutralisation_relative_change=rel_change("inhibition_neutralised"),
        coupling_removal_relative_change=rel_change("coupling_removed"),
        temporal_flat_relative_change=rel_change("temporal_flat"),
        geometry_shuffle_residual_fraction=float(geometry_residual))
    # A genuine spatially ordered, sign-dependent opponent mechanism must depend on
    # which arm occupies which compartment, on synaptic sign, and on the true geometry.
    mechanism["arm_order_matters"] = bool(mechanism["arm_swap_relative_change"] >= 0.20)
    mechanism["sign_matters"] = bool(mechanism["sign_destruction_relative_change"] >= 0.50)
    mechanism["geometry_matters"] = bool(geometry_residual <= 0.50)
    mechanism["mechanism_supported"] = bool(mechanism["arm_order_matters"]
                                            and mechanism["sign_matters"]
                                            and mechanism["geometry_matters"])
    if passed:
        model = sel["config"]["model"]
        category = {"C0_SINGLE": "A", "C1_TWOCOMP": "B", "C2_TWOCOMP_NL": "C"}[model]
        verdict = "PASS"
    elif hs["mean_dsi"] > 0 and improves and mechanism["mechanism_supported"]:
        verdict, category = "FAIL", "D"
    else:
        verdict, category = "FAIL", "E"
    naive_category = "D" if (hs["mean_dsi"] > 0 and improves) else "E"

    gate = dict(verdict=verdict, category=category,
                thresholds=dict(mean_dsi=THRESH_DSI, min_positive=MIN_POS),
                selected=sel["selected"], met_calibration=sel["met_preregistered_threshold"],
                heldout=hs, oracle=osum, families={k: v["summary"] for k, v in fam.items()},
                on_off_specificity_ok=bool(onoff_ok),
                oracle_ratio=float(hs["mean_dsi"] / osum["mean_dsi"]) if osum["mean_dsi"] else None,
                mechanism_diagnostics=mechanism,
                naive_improvement_only_category=naive_category,
                category_rule_note=(
                    "The preregistration lists categories D and E but specifies no precedence "
                    "between them. The first implementation used an unpreregistered heuristic "
                    "(C2 > C0 improvement => D). That heuristic would give category "
                    f"'{naive_category}'. It is superseded here by the mechanistic criteria "
                    "recorded in mechanism_diagnostics, because 'conductance dynamics improve "
                    "selectivity' (D) is only meaningful if the improvement depends on arm "
                    "order, synaptic sign and the true geometry. This is a change to an "
                    "implementation heuristic, not to any preregistered threshold; the primary "
                    "endpoint and its thresholds are unchanged."),
                e19_oracle_reference=e19["decomposition"]["signed_opponent"],
                geometry_null=held["ablations"]["geometry_shuffle"],
                proceed_to_vp_dn=bool(passed and onoff_ok))
    (RESULTS / "gate-status.json").write_text(json.dumps(gate, indent=2) + "\n")
    (RESULTS / "statistics.json").write_text(json.dumps(
        dict(conductance=cond, oracle=orc, heldout_summary=hs, oracle_summary=osum,
             similarity_per_neuron=held["oracle_similarity_per_neuron"],
             similarity_temporal=held["oracle_similarity_temporal"]), indent=2) + "\n")

    # ---------------- figures ----------------
    schematic_svg(CAPTURES / "architecture-schematic.svg", sel["config"])
    fam_best = {}
    for k, v in cal.items():
        m = v["config"]["model"]
        if m not in fam_best or v["mean_dsi"] > fam_best[m]["mean_dsi"]:
            fam_best[m] = v
    order = [m for m in ("C0_SINGLE", "C1_TWOCOMP", "C2_TWOCOMP_NL") if m in fam_best]
    bar_svg(CAPTURES / "calibration-model-family.svg", order,
            [fam_best[m]["mean_dsi"] for m in order],
            "Exp20 calibration: best mean DSI per model family", ref=THRESH_DSI)
    top = sorted(cal.items(), key=lambda kv: -kv[1]["mean_dsi"])[:12]
    bar_svg(CAPTURES / "calibration-parameter-sweep.svg", [k for k, _ in top],
            [v["mean_dsi"] for _, v in top],
            "Exp20 calibration: top 12 parameter sets", ref=THRESH_DSI)
    subs = [s for s in SUBTYPES if s in cond]
    bar_svg(CAPTURES / "heldout-subtype-dsi.svg", subs, [cond[s]["dsi_mean"] for s in subs],
            f"Exp20 held-out DSI by subtype ({sel['selected']})",
            cis=[cond[s]["dsi_ci"] for s in subs], ref=THRESH_DSI)
    bar_svg(CAPTURES / "conductance-vs-oracle.svg",
            [f"{s} cond" for s in subs] + [f"{s} oracle" for s in subs],
            [cond[s]["dsi_mean"] for s in subs] + [orc[s]["dsi_mean"] for s in subs],
            "Exp20 conductance model vs E19 OPPONENT_SIGNED oracle (held-out DSI)")
    ab = held["ablations"]
    alabels = ["selected (observed)"] + [k for k in ab if k not in ("controls", "geometry_shuffle")] + ["geometry null mean"]
    avals = [hs["mean_dsi"]] + [ab[k]["summary"]["mean_dsi"] for k in ab if k not in ("controls", "geometry_shuffle")] \
        + [ab["geometry_shuffle"]["null_mean"]]
    bar_svg(CAPTURES / "ablations.svg", alabels, avals, "Exp20 ablations (held-out mean DSI)")
    axd = held["by_axis"]
    bar_svg(CAPTURES / "generalization.svg", list(axd),
            [axd[a]["summary"]["mean_dsi"] for a in axd],
            "Exp20 held-out mean DSI by generalisation axis", ref=THRESH_DSI)
    oo = [s for s in SUBTYPES if s in onoff]
    bar_svg(CAPTURES / "on-off-specificity.svg", oo,
            [onoff[s]["on"] - onoff[s]["off"] for s in oo],
            "Exp20 ON minus OFF response (T4 should be >0, T5 <0)")
    gsh = ab["geometry_shuffle"]
    hist_svg(CAPTURES / "geometry-shuffle-null.svg",
             RNG.normal(gsh["null_mean"], max(gsh["null_sd"], 1e-9), 2000),
             gsh["observed"], "Exp20 geometry-shuffle null (200 permutations, modelled)")
    tp = RESULTS / "time-resolved-traces.npz"
    if tp.exists():
        tz = np.load(tp)
        for sub in ("T5a", "T4a"):
            if f"{sub}_cond_pref" in tz.files:
                cp, cn = tz[f"{sub}_cond_pref"], tz[f"{sub}_cond_null"]
                trace_svg(CAPTURES / f"traces-{sub.lower()}-conductance.svg",
                          [("preferred", cp, "#275e85"), ("null", cn, "#c0392b"),
                           ("preferred - null", cp - cn, "#111111")],
                          f"Exp20 {sub} conductance-model response over time")
                op, on_ = tz[f"{sub}_orc_pref"], tz[f"{sub}_orc_null"]
                d1 = cp - cn; d2 = op - on_
                trace_svg(CAPTURES / f"traces-{sub.lower()}-vs-oracle.svg",
                          [("conductance pref-null", d1 / (np.abs(d1).max() + 1e-12), "#275e85"),
                           ("oracle pref-null", d2 / (np.abs(d2).max() + 1e-12), "#e67e22")],
                          f"Exp20 {sub} preferred-null, normalised: conductance vs oracle")

    print(f"VERDICT {verdict}  category {category}")
    print(f"  selected {sel['selected']} (met calibration {sel['met_preregistered_threshold']})")
    print(f"  held-out meanDSI {hs['mean_dsi']:+.4f}  positive {hs['n_positive']}/8  "
          f"coherent {hs['n_coherent']}/8  T4 {hs['t4_mean']:+.4f} T5 {hs['t5_mean']:+.4f}")
    print(f"  oracle   meanDSI {osum['mean_dsi']:+.4f}  coherent {osum['n_coherent']}/8  "
          f"ratio {gate['oracle_ratio']}")
    print("  families:", {k: round(v['summary']['mean_dsi'], 4) for k, v in fam.items()})
    print("  ablations:", {k: round(ab[k]['summary']['mean_dsi'], 4)
                           for k in ab if k not in ('controls', 'geometry_shuffle')})
    print(f"  geometry null mean {gsh['null_mean']:+.5f} sd {gsh['null_sd']:.5f} "
          f"observed {gsh['observed']:+.5f} z {gsh['z']:.2f} p {gsh['p_empirical']:.4f}")
    print(f"  ON/OFF ok: {onoff_ok}")
    print("  mechanism diagnostics:")
    for k in ("arm_swap_relative_change", "sign_destruction_relative_change",
              "inhibition_neutralisation_relative_change", "coupling_removal_relative_change",
              "temporal_flat_relative_change", "geometry_shuffle_residual_fraction"):
        print(f"    {k:<44} {mechanism[k]:.3f}")
    print(f"    arm_order_matters {mechanism['arm_order_matters']}  "
          f"sign_matters {mechanism['sign_matters']}  geometry_matters {mechanism['geometry_matters']}")
    print(f"    naive improvement-only category would be: {naive_category}")
    sim = held["oracle_similarity_per_neuron"]
    if sim:
        print(f"  oracle per-neuron Pearson mean {np.mean([v['pearson'] for v in sim.values()]):+.3f}")
    tsim = held["oracle_similarity_temporal"]
    if tsim:
        print(f"  oracle temporal peak xcorr mean {np.mean([v['peak_cross_correlation'] for v in tsim.values()]):+.3f}"
              f"  lags {sorted({v['lag_frames'] for v in tsim.values()})}")


if __name__ == "__main__":
    main()
