"""Experiment 21 analysis: verdicts (T4 / T5 / overall), statistics, comparisons, figures."""
from __future__ import annotations

import json
from math import erfc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / "results" / "experiment-21-tonic-disinhibition"
CAP = ROOT / "captures" / "experiment-21"
E20 = ROOT / "results" / "experiment-20-conductance-dendrite"
RNG = np.random.default_rng(21072026)
SUBTYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
T4 = SUBTYPES[:4]
THRESH_T4, SUBSTANTIAL = 0.15, 0.075


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    m = x[RNG.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def sign_p(e):
    r = np.asarray(e, float); r = r[r != 0]
    if not len(r):
        return 1.0
    z = ((r > 0).sum() - len(r) / 2) / np.sqrt(len(r) / 4)
    return float(min(1.0, erfc(abs(z) / np.sqrt(2))))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); a = np.empty(len(p)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); a[i] = min(1.0, run)
    return a


# ---------------------------------------------------------------- tiny SVG helpers
def _svg(w, h, title):
    from xml.sax.saxutils import escape
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="14" y="22" font-size="14">{escape(title)}</text>']


def bars(path, labels, values, title, ref=None, cis=None, colors=None):
    from xml.sax.saxutils import escape
    W, H = 900, 64 + 24 * len(labels)
    vals = list(values) + [0.0] + ([ref] if ref is not None else [])
    lo, hi = min(vals), max(vals); sp = max(hi - lo, 1e-12); x0, xw = 300, 500
    X = lambda v: x0 + xw * (v - lo) / sp
    p = _svg(W, H, title) + [f'<line x1="{X(0):.1f}" y1="34" x2="{X(0):.1f}" y2="{H-8}" stroke="#999"/>']
    if ref is not None:
        p.append(f'<line x1="{X(ref):.1f}" y1="34" x2="{X(ref):.1f}" y2="{H-8}" stroke="#c0392b" stroke-dasharray="4"/>')
    for i, (lb, v) in enumerate(zip(labels, values)):
        y = 38 + 24 * i; a, b = sorted([X(0), X(v)])
        col = (colors[i] if colors else None) or ("#275e85" if v >= 0 else "#c0392b")
        p += [f'<text x="292" y="{y+12}" text-anchor="end" font-size="11">{escape(str(lb))}</text>',
              f'<rect x="{a:.1f}" y="{y}" width="{max(b-a, .5):.1f}" height="15" fill="{col}"/>']
        if cis and cis[i]:
            p.append(f'<line x1="{X(cis[i][0]):.1f}" y1="{y+7}" x2="{X(cis[i][1]):.1f}" y2="{y+7}" stroke="#111"/>')
        p.append(f'<text x="{W-6}" y="{y+12}" font-size="11" text-anchor="end">{v:+.4f}</text>')
    path.write_text("\n".join(p + ["</svg>"]))


def panel_traces(path, panels, title):
    """Grid of small line plots: panels = [(subtitle, [(label, array, color), ...]), ...]."""
    from xml.sax.saxutils import escape
    cols = 3; rows = (len(panels) + cols - 1) // cols
    pw, ph = 300, 190; W, H = cols * pw + 20, rows * ph + 40
    p = _svg(W, H, title)
    for i, (sub, series) in enumerate(panels):
        ox = 10 + (i % cols) * pw; oy = 34 + (i // cols) * ph
        allv = np.concatenate([np.asarray(s[1], float) for s in series])
        lo, hi = float(allv.min()), float(allv.max()); sp = max(hi - lo, 1e-12)
        L, T, Wd, Hd = ox + 40, oy + 22, pw - 60, ph - 55
        p += [f'<text x="{ox+4}" y="{oy+14}" font-size="12">{escape(sub)}</text>',
              f'<rect x="{L}" y="{T}" width="{Wd}" height="{Hd}" fill="none" stroke="#ccc"/>',
              f'<text x="{ox+2}" y="{T+9}" font-size="9">{hi:.3g}</text>',
              f'<text x="{ox+2}" y="{T+Hd}" font-size="9">{lo:.3g}</text>']
        if lo < 0 < hi:
            yz = T + Hd * (hi - 0) / sp
            p.append(f'<line x1="{L}" y1="{yz:.1f}" x2="{L+Wd}" y2="{yz:.1f}" stroke="#eee"/>')
        for j, (lab, arr, col) in enumerate(series):
            arr = np.asarray(arr, float); n = len(arr)
            pts = " ".join(f"{L + Wd * k / max(n - 1, 1):.1f},{T + Hd * (hi - v) / sp:.1f}" for k, v in enumerate(arr))
            p.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.5" points="{pts}"/>')
            p.append(f'<text x="{L + 4 + 70 * j}" y="{T + Hd + 16}" font-size="10" fill="{col}">{escape(lab)}</text>')
    path.write_text("\n".join(p + ["</svg>"]))


def histogram(path, samples, observed, title):
    W, H, L = 860, 280, 50
    s = np.asarray(samples); lo = min(s.min(), observed); hi = max(s.max(), observed); sp = max(hi - lo, 1e-12)
    c, e = np.histogram(s, bins=30, range=(lo, hi)); cm = max(c.max(), 1)
    X = lambda v: L + (W - 30 - L) * (v - lo) / sp
    p = _svg(W, H, title) + [f'<line x1="{L}" y1="{H-36}" x2="{W-30}" y2="{H-36}" stroke="#333"/>']
    for k, a, b in zip(c, e[:-1], e[1:]):
        h = (H - 80) * k / cm
        p.append(f'<rect x="{X(a):.1f}" y="{H-36-h:.1f}" width="{max(X(b)-X(a)-1, 1):.1f}" height="{h:.1f}" fill="#9bb7cc"/>')
    p += [f'<line x1="{X(observed):.1f}" y1="36" x2="{X(observed):.1f}" y2="{H-36}" stroke="#c0392b" stroke-width="2"/>',
          f'<text x="{X(observed):.1f}" y="32" font-size="11" fill="#c0392b" text-anchor="middle">observed {observed:+.4f}</text>',
          f'<text x="{L}" y="{H-18}" font-size="10">{lo:+.4f}</text>', f'<text x="{W-80}" y="{H-18}" font-size="10">{hi:+.4f}</text>']
    path.write_text("\n".join(p + ["</svg>"]))


def schematic(path, cfg):
    p = _svg(900, 330, "Exp21: single passive compartment with tonic inhibition that can be released")
    p += ['<rect x="330" y="95" width="240" height="120" rx="12" fill="#eef4f8" stroke="#275e85"/>',
          '<text x="450" y="125" font-size="13" text-anchor="middle">T4 compartment</text>',
          '<text x="450" y="148" font-size="11" text-anchor="middle">C dV/dt = -gL(V-EL) - gE(V-EE)</text>',
          '<text x="450" y="166" font-size="11" text-anchor="middle">- gI(V-EI)</text>',
          '<text x="450" y="190" font-size="11" text-anchor="middle">output = max(V - V_rest, 0)</text>',
          '<text x="40" y="100" font-size="12" fill="#27ae60">Mi1 (ON, fast, excitatory)</text>',
          '<text x="40" y="118" font-size="11">r = max(0, r0_exc + beta s/sigma), r0_exc = ' + f'{cfg["r0_exc"]}</text>',
          '<text x="40" y="165" font-size="12" fill="#c0392b">Mi9 (OFF, slow, inhibitory)</text>',
          '<text x="40" y="183" font-size="11">tonic r0_inh = ' + f'{cfg["r0_inh"]}; ON motion -&gt; below baseline</text>',
          '<text x="40" y="230" font-size="12" fill="#8e44ad">Mi4 (ON, slow, inhibitory)</text>',
          '<text x="40" y="248" font-size="11">tonic r0_inh; ON motion -&gt; above baseline</text>',
          '<line x1="290" y1="110" x2="330" y2="140" stroke="#27ae60" stroke-width="2"/>',
          '<line x1="290" y1="175" x2="330" y2="160" stroke="#c0392b" stroke-width="2"/>',
          '<line x1="290" y1="240" x2="330" y2="185" stroke="#8e44ad" stroke-width="2"/>',
          f'<text x="610" y="130" font-size="11">G = {cfg["G"]} g_L, beta = {cfg["beta"]}</text>',
          f'<text x="610" y="150" font-size="11">E_I = {cfg["E_inh"]} (0 = pure shunt)</text>',
          '<text x="610" y="170" font-size="11">conductance = G sum|w| r / M, never negative</text>',
          '<text x="610" y="190" font-size="11">channel set by MaleCNS weight sign</text>',
          '<text x="40" y="300" font-size="11" fill="#555">Frozen configuration shown. No opponent product; no per-subtype parameters.</text>']
    path.write_text("\n".join(p + ["</svg>"]))


def transfer_curves(path, cfg, sigma):
    s = np.linspace(-3 * sigma, 3 * sigma, 121)
    r0 = max(cfg["r0_inh"], 1e-9); b = cfg["beta"]
    lin = np.maximum(0.0, r0 + b * s / sigma)
    bnd = 2 * r0 / (1 + np.exp(-2 * b * s / (sigma * r0)))
    panel_traces(path, [("presynaptic rate vs signal (x = -3..+3 sigma)",
                         [("linear (D1/D2)", lin, "#275e85"), ("bounded (D3)", bnd, "#e67e22"),
                          ("tonic r0", np.full_like(s, r0), "#999999")])],
                 "Exp21 tonic conductance mapping: rate falls below r0 for negative signal")


def main():
    CAP.mkdir(parents=True, exist_ok=True)
    cal = json.loads((R / "calibration-results.json").read_text())
    sel = json.loads((R / "selected-config.json").read_text())
    held = json.loads((R / "heldout-results.json").read_text())
    abl = json.loads((R / "ablation-results.json").read_text())
    diag = json.loads((R / "gate2-diagnostic.json").read_text())
    z = np.load(R / "heldout-per-neuron.npz")
    e20 = json.loads((E20 / "gate-status.json").read_text())

    rows, ps = {}, []
    for s in SUBTYPES:
        d, e = z[f"e21_{s}_dsi"], z[f"e21_{s}_effect"]
        resp = e != 0
        rows[s] = dict(n=int(len(d)), dsi=float(d.mean()), ci=boot_ci(d),
                       signResp=float((e[resp] > 0).mean()) if resp.any() else 0.0,
                       p=sign_p(e))
        ps.append(rows[s]["p"])
    for s, a in zip(SUBTYPES, holm(ps)):
        rows[s]["p_holm"] = float(a)
    orc = {s: dict(dsi=float(z[f"oracle_{s}_dsi"].mean())) for s in SUBTYPES}

    hs = held["heldout"]["summary"]
    geo = abl["A6_geometry_shuffle"]
    onoff = held["on_off"]
    t4_on = all(onoff[s]["on"] > onoff[s]["off"] for s in T4)
    t5_off = all(onoff[s]["off"] > onoff[s]["on"] for s in SUBTYPES[4:])
    inverted_t4 = [s for s in T4 if rows[s]["dsi"] < 0 and rows[s]["p_holm"] < 0.05]
    gates_passed = sel["selection_status"] != "no_mechanism_valid_configuration"
    endpoint = dict(t4_mean_ge_015=hs["t4_mean"] >= THRESH_T4, t4_4of4_positive=hs["t4_positive"] == 4,
                    no_t4_inverted=not inverted_t4, t4_on_preference=t4_on,
                    gates_passed=gates_passed,
                    geometry_residual_le_05=(geo["residual_fraction"] is not None
                                             and geo["residual_fraction"] <= 0.5))
    t4_pass = all(endpoint.values())
    fam = sel["config"]["family"]
    if t4_pass:
        category = "B" if fam.startswith("D3") else "A"
    elif gates_passed:
        category = "C" if hs["t4_mean"] >= SUBSTANTIAL else "D"
    else:
        category = "E"
    t5_mean = hs["t5_mean"]
    t5_verdict = ("transfers" if (t5_mean >= THRESH_T4 and sum(rows[s]["dsi"] > 0 for s in SUBTYPES[4:]) == 4)
                  else "does not transfer")
    verdict = dict(overall="PASS" if t4_pass and t5_verdict == "transfers" else ("PARTIAL" if t4_pass else "FAIL"),
                   mechanistic_category=category, t4_verdict="PASS" if t4_pass else "FAIL",
                   t5_verdict=t5_verdict, endpoint=endpoint, inverted_t4_subtypes=inverted_t4,
                   selection_status=sel["selection_status"], gate_outcomes=sel["gate_outcomes"],
                   heldout=hs, oracle_t4_mean=held["oracle"]["t4_mean"],
                   oracle_all8_mean=held["oracle"]["summary"]["mean_dsi"],
                   e20=dict(C0=held["d0_e20"]["C0_SINGLE"], C2=held["d0_e20"]["C2_TWOCOMP_NL"]),
                   geometry=dict((k, geo[k]) for k in ("observed", "null_mean", "null_sd", "z",
                                                        "p_empirical", "residual_fraction")),
                   on_off=dict(t4_on=t4_on, t5_off=t5_off),
                   proceed_to_vp_dn=bool(t4_pass and t5_verdict == "transfers"))
    (R / "gate-status.json").write_text(json.dumps(verdict, indent=2) + "\n")
    (R / "statistics.json").write_text(json.dumps(dict(e21=rows, oracle=orc,
                                                       similarity=held["oracle_similarity_per_neuron"],
                                                       temporal=held["oracle_similarity_temporal"]), indent=2) + "\n")

    # ---------------- figures ----------------
    schematic(CAP / "01-model-schematic.svg", sel["config"])
    transfer_curves(CAP / "02-tonic-conductance-mapping.svg", sel["config"], sel["constants"]["sigma"])
    tz = np.load(R / "mechanism-traces.npz")
    for sub in ("T4a", "T4c"):
        if f"{sub}_pref_g_Mi9" not in tz.files:
            continue
        g = lambda k: tz[f"{sub}_{k}"]
        panel_traces(CAP / f"03-mechanism-traces-{sub.lower()}.svg", [
            ("Mi9 conductance - baseline", [("pref", g("pref_g_Mi9"), "#275e85"), ("null", g("null_g_Mi9"), "#c0392b")]),
            ("Mi1 excitation - baseline", [("pref", g("pref_g_Mi1"), "#275e85"), ("null", g("null_g_Mi1"), "#c0392b")]),
            ("Mi4 conductance - baseline", [("pref", g("pref_g_Mi4"), "#275e85"), ("null", g("null_g_Mi4"), "#c0392b")]),
            ("total conductance - rest", [("pref", g("pref_g_total"), "#275e85"), ("null", g("null_g_total"), "#c0392b")]),
            ("R_in / R_rest", [("pref", g("pref_R_in"), "#275e85"), ("null", g("null_R_in"), "#c0392b")]),
            ("output max(V - V_rest, 0)", [("pref", g("pref_response"), "#275e85"), ("null", g("null_response"), "#c0392b")]),
        ], f"Exp21 {sub} mechanism, held-out ON bars at 1.5 px/frame, frozen config (neuron-averaged)")
    keys = list(cal)
    fams = {f: [k for k in keys if cal[k]["config"]["family"] == f] for f in
            ("D1_UNIFORM_TONIC", "D2_TONIC_INHIBITION", "D3_BOUNDED_TONIC_INHIBITION")}
    bars(CAP / "04-calibration-gates.svg",
         ["Gate1 pass (release)", "Gate2 pass (Rin overlaps excitation)", "Gate3 pass",
          "max median overlap", "max median Rin increase"],
         [sum(v["gate1"] for v in cal.values()) / len(cal), sum(v["gate2"] for v in cal.values()) / len(cal),
          sum(v.get("gate3", False) for v in cal.values()) / len(cal),
          max(v["median_overlap"] for v in cal.values()), max(v["median_rin_increase"] for v in cal.values())],
         "Exp21 calibration mechanism screen, 180 configurations (fractions / maxima)")
    fb = {f: max((cal[k]["cal_t4_mean_dsi"] for k in ks), default=0) for f, ks in fams.items()}
    sweep = {}
    for k, v in cal.items():
        if v["config"]["family"] == "D2_TONIC_INHIBITION":
            r0 = v["config"]["r0_inh"]; sweep[r0] = max(sweep.get(r0, -1), v["cal_t4_mean_dsi"])
    bars(CAP / "05-calibration-dsi-and-baseline-sweep.svg",
         [f"best {f}" for f in fb] + [f"D2 best, r0_inh={r0:g}" for r0 in sorted(sweep)],
         list(fb.values()) + [sweep[r] for r in sorted(sweep)],
         "Exp21 calibration T4 mean DSI (descriptive: no configuration passed the gates)", ref=THRESH_T4)
    bars(CAP / "06-heldout-subtype-dsi.svg", list(SUBTYPES), [rows[s]["dsi"] for s in SUBTYPES],
         "Exp21 held-out DSI by subtype, frozen config (T4 primary, T5 descriptive)",
         ref=THRESH_T4, cis=[rows[s]["ci"] for s in SUBTYPES])
    comp = [("E20 C0 (D0)", held["d0_e20"]["C0_SINGLE"]["t4_mean"], held["d0_e20"]["C0_SINGLE"]["mean_dsi"]),
            ("E20 C2 (D0)", held["d0_e20"]["C2_TWOCOMP_NL"]["t4_mean"], held["d0_e20"]["C2_TWOCOMP_NL"]["mean_dsi"]),
            ("E21 frozen", hs["t4_mean"], hs["mean_dsi"])]
    comp += [(f"E21 best {f.split('_')[0]}", v["summary"]["t4_mean"], v["summary"]["mean_dsi"])
             for f, v in held["families"].items()]
    comp += [("E19 oracle", held["oracle"]["t4_mean"], held["oracle"]["summary"]["mean_dsi"])]
    bars(CAP / "07-e21-vs-e20-vs-oracle.svg", [f"{c[0]} T4" for c in comp] + [f"{c[0]} all-8" for c in comp],
         [c[1] for c in comp] + [c[2] for c in comp], "Exp21 vs E20 vs E19 oracle (held-out mean DSI)", ref=THRESH_T4)
    an = [k for k in abl if k.startswith("A") and k not in ("A6_geometry_shuffle", "A10_static")]
    bars(CAP / "08-ablations-t4.svg", ["frozen (observed)"] + an, [hs["t4_mean"]] + [abl[k]["summary"]["t4_mean"] for k in an],
         "Exp21 ablations, held-out T4 mean DSI")
    histogram(CAP / "09-geometry-shuffle.svg", geo["null_samples"], geo["observed"],
              "Exp21 geometry shuffle null, 200 permutations (T4 mean DSI, held-out speed axis)")
    bars(CAP / "10-on-off-specificity.svg", list(SUBTYPES), [onoff[s]["on"] - onoff[s]["off"] for s in SUBTYPES],
         "Exp21 ON minus OFF response (T4 should be > 0, T5 < 0)")
    dk = [k for k in diag]
    lab, val = [], []
    for k in dk:
        for tag in ("preferred", "null"):
            d = diag[k].get(tag, {})
            if "release_to_added_ratio" in d:
                lab.append(f"{k.split('|')[0][:2]} {'|'.join(k.split('|')[1:3])} {tag}")
                val.append(d["release_to_added_ratio"]["median"])
    if lab:
        bars(CAP / "11-gate2-release-vs-added-conductance.svg", lab, val,
             "Exp21 Gate 2 diagnostic: Mi9 conductance removed / (Mi1 + Mi4) added, in excitation window",
             ref=1.0)

    print(f"OVERALL {verdict['overall']}  category {category}  T4 {verdict['t4_verdict']}  T5 {t5_verdict}")
    print(f"  selection status: {sel['selection_status']}; gate outcomes {sel['gate_outcomes']}")
    print(f"  held-out T4 {hs['t4_mean']:+.4f} ({hs['t4_positive']}/4), all-8 {hs['mean_dsi']:+.4f} ({hs['n_positive']}/8)")
    for s in SUBTYPES:
        r = rows[s]
        print(f"   {s}: DSI {r['dsi']:+.4f} CI[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] signResp {r['signResp']:.2f} pHolm {r['p_holm']:.2g}")
    print(f"  endpoint: {endpoint}")
    print(f"  geometry: obs {geo['observed']:+.4f} null {geo['null_mean']:+.4f}±{geo['null_sd']:.4f} "
          f"p {geo['p_empirical']:.4f} residual {geo['residual_fraction']}")
    print(f"  ON/OFF: T4 on {t4_on}, T5 off {t5_off}")


if __name__ == "__main__":
    main()
