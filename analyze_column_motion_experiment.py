"""Experiment 18 analysis: geometry gate, held-out statistics, ablations, verdict.

Statistical unit is the individual T4/T5 neuron (the front end is deterministic, so
seed-based inference is not fabricated). Bootstrap over neurons; Holm correction
across the eight subtypes. CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "experiment-18-column-motion"
CAPTURES = ROOT / "captures" / "experiment-18"
RNG = np.random.default_rng(18062026)
NBOOT = 10000
SUBTYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
E17_MAX_DSI = 0.012


def boot_ci(x, n=NBOOT):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 2
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def sign_test_p(x):
    """Two-sided sign test that the per-neuron effect is centred above zero."""
    x = np.asarray(x, float)
    resp = x[x != 0]
    if len(resp) == 0:
        return 1.0
    k = int((resp > 0).sum()); n = len(resp)
    from math import comb
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2.0 ** n) if n <= 1000 else None
    if tail is None:   # normal approximation for large n
        z = (k - n / 2) / np.sqrt(n / 4)
        from math import erfc
        tail = 0.5 * erfc(z / np.sqrt(2))
    return float(min(1.0, 2 * min(tail, 1 - tail + 1e-300)))


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        val = (len(pvals) - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def bar_svg(path, labels, values, title, ref=None, cis=None):
    from xml.sax.saxutils import escape
    width, height = 860, 70 + len(labels) * 30
    lo = min(list(values) + [0.0] + ([ref] if ref else []))
    hi = max(list(values) + [0.0] + ([ref] if ref else []))
    span = max(hi - lo, 1e-12); x0, xw = 250, 520
    X = lambda v: x0 + xw * (v - lo) / span
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="15" y="26" font-size="15">{escape(title)}</text>',
         f'<line x1="{X(0.0):.1f}" y1="40" x2="{X(0.0):.1f}" y2="{height-12}" stroke="#999"/>']
    if ref:
        p.append(f'<line x1="{X(ref):.1f}" y1="40" x2="{X(ref):.1f}" y2="{height-12}" stroke="#c0392b" stroke-dasharray="4"/>')
    for i, (lb, v) in enumerate(zip(labels, values)):
        y = 46 + i * 30
        a, b = sorted([X(0.0), X(v)])
        p += [f'<text x="240" y="{y+14}" text-anchor="end" font-size="11">{escape(str(lb))}</text>',
              f'<rect x="{a:.1f}" y="{y}" width="{max(b-a,0.5):.1f}" height="18" fill="#275e85"/>']
        if cis and cis[i] and np.isfinite(cis[i][0]):
            p.append(f'<line x1="{X(cis[i][0]):.1f}" y1="{y+9}" x2="{X(cis[i][1]):.1f}" y2="{y+9}" stroke="#111" stroke-width="1.4"/>')
        p.append(f'<text x="{width-8}" y="{y+14}" font-size="11" text-anchor="end">{v:+.4f}</text>')
    path.write_text("\n".join(p + ["</svg>"]))


def polar_svg(path, summary, title):
    from xml.sax.saxutils import escape
    W = H = 460; cx = cy = 230; R = 165
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H+30}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="15" y="22" font-size="15">{escape(title)}</text>',
         f'<circle cx="{cx}" cy="{cy+20}" r="{R}" fill="none" stroke="#ccc"/>',
         f'<line x1="{cx-R}" y1="{cy+20}" x2="{cx+R}" y2="{cy+20}" stroke="#eee"/>',
         f'<line x1="{cx}" y1="{cy+20-R}" x2="{cx}" y2="{cy+20+R}" stroke="#eee"/>']
    colors = {"a": "#275e85", "b": "#c0392b", "c": "#27ae60", "d": "#8e44ad"}
    for sub, s in summary.items():
        if not s.get("n_eligible"):
            continue
        ang = np.radians(s["mean_angle_deg"]); r = R * min(s["concentration"], 1.0)
        x = cx + r * np.cos(ang); y = cy + 20 - r * np.sin(ang)
        col = colors[sub[2]]
        dash = "" if sub.startswith("T5") else ' stroke-dasharray="5,3"'
        p.append(f'<line x1="{cx}" y1="{cy+20}" x2="{x:.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="2"{dash}/>')
        p.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="11" fill="{col}">{sub}</text>')
    p.append(f'<text x="15" y="{H+24}" font-size="10">radius = concentration R; solid = T5, dashed = T4</text>')
    path.write_text("\n".join(p + ["</svg>"]))


def main():
    CAPTURES.mkdir(parents=True, exist_ok=True)
    geom = json.loads((RESULTS / "geometry-summary.json").read_text())
    cal = json.loads((RESULTS / "calibration-results.json").read_text())
    sel = json.loads((RESULTS / "selected-config.json").read_text())
    held = json.loads((RESULTS / "heldout-results.json").read_text())
    pn = np.load(RESULTS / "heldout-per-neuron.npz")

    # ---------------- geometry gate ----------------
    elig = np.mean([geom[s]["eligible_fraction"] for s in SUBTYPES])
    conc = {s: geom[s]["concentration"] for s in SUBTYPES}
    ang = {s: geom[s]["mean_angle_deg"] for s in SUBTYPES}
    def sep(a, b):
        d = abs(ang[a] - ang[b]) % 360
        return min(d, 360 - d)
    pairs = dict(T4ab=sep("T4a", "T4b"), T4cd=sep("T4c", "T4d"),
                 T5ab=sep("T5a", "T5b"), T5cd=sep("T5c", "T5d"))
    axes = dict(T4=sep("T4a", "T4c"), T5=sep("T5a", "T5c"))
    antiparallel = all(v > 150 for v in pairs.values())
    orthogonal = all(60 < v < 120 for v in axes.values())
    geom_gate = ("PASS" if (elig >= 0.60 and min(conc.values()) >= 0.30 and antiparallel and orthogonal)
                 else "PARTIAL" if (elig >= 0.60 and min(conc.values()) >= 0.15) else "FAIL")

    # ---------------- held-out statistics ----------------
    def stats_for(prefix, label):
        rows = {}
        pvals = []
        for s in SUBTYPES:
            key = f"{prefix}_{s}_dsi"
            ekey = f"{prefix}_{s}_effect"
            if key not in pn.files:
                continue
            d = pn[key]; e = pn[ekey]
            resp = e != 0
            rows[s] = dict(n=int(len(d)), dsi_mean=float(d.mean()), dsi_ci=boot_ci(d),
                           effect_mean=float(e.mean()), effect_ci=boot_ci(e),
                           responsive_fraction=float(resp.mean()),
                           sign_consistency=float((e > 0).mean()),
                           sign_consistency_responsive=float((e[resp] > 0).mean()) if resp.any() else 0.0,
                           p_sign=sign_test_p(e))
            pvals.append(rows[s]["p_sign"])
        if rows:
            adj = holm(np.array(pvals))
            for s, a in zip(list(rows), adj):
                rows[s]["p_holm"] = float(a)
        return dict(label=label, subtypes=rows,
                    mean_abs_dsi=float(np.mean([abs(v["dsi_mean"]) for v in rows.values()])) if rows else 0.0,
                    mean_sign_responsive=float(np.mean([v["sign_consistency_responsive"] for v in rows.values()])) if rows else 0.0)

    selected_stats = stats_for("selected", sel["selected"])
    abl_stats = {k: stats_for(k, k) for k in
                 ("geometry_shuffle", "temporal_flat", "no_rectification", "explicit_correlator")
                 if any(f.startswith(k) for f in pn.files)}

    # ---------------- verdict ----------------
    bio = selected_stats
    bio_ok = (bio["mean_abs_dsi"] > 3 * E17_MAX_DSI
              and bio["mean_sign_responsive"] >= 0.7
              and all(v["dsi_mean"] > 0 for v in bio["subtypes"].values()))
    hr = abl_stats.get("explicit_correlator")
    hr_ok = bool(hr and hr["mean_abs_dsi"] > 3 * E17_MAX_DSI and hr["mean_sign_responsive"] >= 0.7
                 and all(v["dsi_mean"] > 0 for v in hr["subtypes"].values()))
    onoff = held["on_off"]
    onoff_ok = (all(onoff[s]["on"] > onoff[s]["off"] for s in onoff if s.startswith("T4"))
                and all(onoff[s]["off"] > onoff[s]["on"] for s in onoff if s.startswith("T5")))
    if bio_ok:
        verdict, category = "PASS", ("B" if sel["config"]["name"] in ("M0", "M1") else "C")
    elif hr_ok:
        verdict, category = "FAIL", "D"
    elif geom_gate == "FAIL":
        verdict, category = "FAIL", "A"
    else:
        verdict, category = "FAIL", "F"

    gate = dict(verdict=verdict, category=category, geometry_gate=geom_gate,
                geometry=dict(mean_eligible_fraction=float(elig), concentration=conc,
                              antiparallel_pairs_deg=pairs, axis_separation_deg=axes,
                              antiparallel=bool(antiparallel), orthogonal=bool(orthogonal)),
                biological_model=dict(selected=sel["selected"],
                                      met_calibration_threshold=sel["met_preregistered_threshold"],
                                      mean_abs_dsi=bio["mean_abs_dsi"],
                                      mean_sign_responsive=bio["mean_sign_responsive"],
                                      passed=bool(bio_ok)),
                explicit_correlator=dict(mean_abs_dsi=hr["mean_abs_dsi"] if hr else None,
                                         mean_sign_responsive=hr["mean_sign_responsive"] if hr else None,
                                         passed=bool(hr_ok)),
                on_off_specificity_ok=bool(onoff_ok),
                experiment17_reference_max_dsi=E17_MAX_DSI,
                proceed_to_experiment19=bool(bio_ok))
    (RESULTS / "gate-status.json").write_text(json.dumps(gate, indent=2) + "\n")
    (RESULTS / "statistics.json").write_text(json.dumps(
        dict(selected=selected_stats, ablations=abl_stats, geometry_gate=geom_gate), indent=2) + "\n")
    (RESULTS / "ablation-results.json").write_text(json.dumps(
        {k: dict(mean_abs_dsi=v["mean_abs_dsi"], mean_sign_responsive=v["mean_sign_responsive"],
                 subtypes={s: dict(dsi_mean=r["dsi_mean"], sign_consistency_responsive=r["sign_consistency_responsive"])
                           for s, r in v["subtypes"].items()})
         for k, v in abl_stats.items()}, indent=2) + "\n")
    (RESULTS / "generalization-results.json").write_text(json.dumps(held["by_axis"], indent=2) + "\n")

    # ---------------- figures ----------------
    polar_svg(CAPTURES / "subtype-orientations.svg", geom,
              "Exp18 T4/T5 fast-slow input offset orientation (anatomy only)")
    t4 = [s for s in SUBTYPES if s.startswith("T4")]
    t5 = [s for s in SUBTYPES if s.startswith("T5")]
    bar_svg(CAPTURES / "t4-receptive-fields.svg", t4, [geom[s]["mean_magnitude"] for s in t4],
            "Exp18 T4 mean fast-slow offset magnitude (columns)")
    bar_svg(CAPTURES / "t5-receptive-fields.svg", t5, [geom[s]["mean_magnitude"] for s in t5],
            "Exp18 T5 mean fast-slow offset magnitude (columns)")
    order_keys = sorted(cal, key=lambda k: (cal[k]["config"]["name"], k))
    bar_svg(CAPTURES / "model-comparison-dsi.svg",
            [k.split("|")[0] + "|" + k.split("|")[1] + k.split("|")[2] for k in order_keys],
            [cal[k]["mean_abs_dsi"] for k in order_keys],
            "Exp18 calibration mean |DSI| by model (M_HR is a control, not the biological result)",
            ref=0.05)
    for fam, keys, fname in (("T4", t4, "heldout-t4-dsi.svg"), ("T5", t5, "heldout-t5-dsi.svg")):
        vals = [selected_stats["subtypes"][s]["dsi_mean"] for s in keys if s in selected_stats["subtypes"]]
        cis = [selected_stats["subtypes"][s]["dsi_ci"] for s in keys if s in selected_stats["subtypes"]]
        bar_svg(CAPTURES / fname, keys, vals, f"Exp18 held-out {fam} DSI, selected model {sel['selected']}", cis=cis)
    if "explicit_correlator" in abl_stats:
        hrs = abl_stats["explicit_correlator"]["subtypes"]
        bar_svg(CAPTURES / "explicit-correlator-dsi.svg", list(hrs),
                [hrs[s]["dsi_mean"] for s in hrs],
                "Exp18 held-out DSI, explicit correlator CONTROL (not the biological model)",
                cis=[hrs[s]["dsi_ci"] for s in hrs])
    abl_labels = ["selected"] + list(abl_stats)
    abl_vals = [selected_stats["mean_abs_dsi"]] + [abl_stats[k]["mean_abs_dsi"] for k in abl_stats]
    bar_svg(CAPTURES / "geometry-ablation.svg", abl_labels, abl_vals,
            "Exp18 mean |DSI|: selected model vs ablations and correlator control")
    ax = held["by_axis"]
    bar_svg(CAPTURES / "generalization-dsi.svg", list(ax),
            [float(np.mean([abs(v["dsi_mean"]) for v in ax[a].values()])) for a in ax],
            "Exp18 held-out mean |DSI| by generalisation axis")

    print(f"GEOMETRY GATE: {geom_gate}  (eligible {elig:.3f}, min R {min(conc.values()):.3f}, "
          f"antiparallel {antiparallel}, orthogonal {orthogonal})")
    print(f"  pair separations {pairs}  axis separations {axes}")
    print(f"biological model {sel['selected']}: mean|DSI| {bio['mean_abs_dsi']:.4f}, "
          f"sign(responsive) {bio['mean_sign_responsive']:.2f}, passed {bio_ok}")
    if hr:
        print(f"explicit correlator CONTROL: mean|DSI| {hr['mean_abs_dsi']:.4f}, "
              f"sign(responsive) {hr['mean_sign_responsive']:.2f}")
    for k, v in abl_stats.items():
        print(f"  ablation {k:<22} mean|DSI| {v['mean_abs_dsi']:.4f}")
    print(f"ON/OFF specificity ok: {onoff_ok}")
    print(f"VERDICT {verdict}  category {category}")


if __name__ == "__main__":
    main()
