"""Experiment 29 figures (no new simulation): 14 SVGs in captures/experiment-29/."""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from analyze_mb_joint_operating_point import scatter
from analyze_tonic_disinhibition_experiment import _svg, bars

ROOT = Path(__file__).resolve().parent
R = ROOT / "results" / "experiment-29-broader-mb-circuit-audit"
C = ROOT / "captures" / "experiment-29"
load = lambda n: json.loads((R / n).read_text()) if (R / n).exists() else None


def matrix(path, rows, cols, m, title, fmt="{:.0f}", log=False, outline=None, cw=34, ch=22):
    """Generic labelled heatmap (rows x cols); outline[i][j] draws a green border."""
    x0, y0 = 120, 150
    W, H = max(900, x0 + cw * len(cols) + 20), y0 + ch * len(rows) + 30
    p = _svg(W, H, title)
    v = np.log10(1 + np.abs(m)) if log else m
    fin = v[np.isfinite(v)]; lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (0, 1)
    a = max(abs(lo), abs(hi), 1e-12); div = lo < 0 < hi
    for j, c in enumerate(cols):
        p.append(f'<text x="{x0 + cw * j + cw / 2}" y="{y0 - 6}" font-size="10" transform="rotate(-60 {x0 + cw * j + cw / 2} {y0 - 6})">{escape(c)}</text>')
    for i, r in enumerate(rows):
        p.append(f'<text x="{x0 - 6}" y="{y0 + ch * i + ch / 2 + 4}" text-anchor="end" font-size="11">{escape(r)}</text>')
        for j in range(len(cols)):
            x = v[i, j]
            if not np.isfinite(x):
                col = "#eee"
            elif div:
                t = x / a
                col = f"rgb({int(255 - 120 * max(0, -t))},{int(255 - 150 * abs(t))},{int(255 - 120 * max(0, t))})"
            else:
                t = (x - lo) / max(hi - lo, 1e-12)
                col = f"rgb({int(245 - 200 * t)},{int(248 - 140 * t)},{int(255 - 90 * t)})"
            ok = outline is not None and outline[i][j]
            p.append(f'<rect x="{x0 + cw * j}" y="{y0 + ch * i}" width="{cw - 1}" height="{ch - 1}" fill="{col}" '
                     f'stroke="{"#1e8449" if ok else "none"}" stroke-width="{3 if ok else 0}"/>')
            if np.isfinite(m[i, j]) and m[i, j] != 0:
                p.append(f'<text x="{x0 + cw * j + cw / 2}" y="{y0 + ch * i + ch / 2 + 3}" text-anchor="middle" font-size="8">'
                         f'{escape(fmt.format(m[i, j]))}</text>')
    path.write_text("\n".join(p + ["</svg>"]))


def main():
    C.mkdir(parents=True, exist_ok=True)
    for f in C.glob("*.svg"):
        f.unlink()
    kc = load("kc-population-audit.json"); vis = load("kc-visual-response.json"); conn = load("kc-mbon-connectivity.json")
    fz = load("candidate-ranking-freeze.json"); lev = load("dynamic-leverage-results.json"); dan = load("dan-compartment-audit.json")
    cc = load("candidate-circuits.json"); conf = load("confirmation-results.json"); sens = load("operating-regime-sensitivity.json")
    types = list(kc["types"])
    bars(C / "01-kc-type-cell-counts.svg", types, [kc["types"][t]["n"] for t in types],
         f"E29 KC types (n = {kc['total_kc']} cells, {kc['n_types']} types)")
    bars(C / "02-visual-input-fraction-by-kc-class.svg", types,
         [100 * kc["types"][t]["static_visual_input"]["vp_fraction_of_input"] for t in types],
         "E29 direct visual-projection share of KC-class input |W| (%); >= 1 % = substantial", ref=1.0)
    stims = vis["stimuli"][2:]
    m = np.array([[100 * (vis["classes"][t]["stimuli"][s]["no_visual"]["rel_diff"] or np.nan) for s in stims] for t in types])
    outl = [[vis["classes"][t]["stimuli"][s]["passes_gate"] for s in stims] for t in types]
    matrix(C / "03-kc-visual-response-heatmap.svg", [f"{t} ({vis['classes'][t]['cells']})" for t in types], stims, m,
           "E29 KC-class response vs no-visual (%), 60 ms + APL x8, seeds 3001-3006; green = passes frozen visual gate",
           fmt="{:+.0f}", outline=outl, cw=70, ch=24)
    bars(C / "04-fraction-modulated-cells.svg", types,
         [vis["classes"][t]["stimuli"][vis["classes"][t]["best_stimulus"]]["fraction_cells_modulated"] for t in types],
         "E29 fraction of cells modulated (best stimulus; per-cell sign 5/6 and |d| >= 0.2 max(base,1))")
    mbons = conn["mbon_types"]
    get = lambda t, mb, k: conn["pairs"].get(f"{t}->{mb}", {}).get(k, 0)
    matrix(C / "05-kc-mbon-edge-count-matrix.svg", types, mbons, np.array([[get(t, mb, "edges") for mb in mbons] for t in types], float),
           "E29 KC type -> MBON type edge counts (FlyBrain W; colour log scale)", log=True)
    matrix(C / "06-kc-mbon-normalised-weight-matrix.svg", types, mbons,
           np.array([[get(t, mb, "abs_weight_sum") for mb in mbons] for t in types], float),
           "E29 KC type -> MBON type summed |normalised W| (each MBON's total input |W| ~ number of cells)", fmt="{:.2f}")
    elig = [r for r in fz["full_ranking"] if r["visual_gate"] and r["dan_match"]][:30]
    refs = [r for r in fz["full_ranking"] if r["key"] in ("KCg-d->MBON01", "KCg-d->MBON11")]
    bars(C / "07-static-mbon-input-fraction.svg", [f"#{r['rank']} {r['key']} ({r['dan_type']})" for r in elig + refs],
         [100 * r["static_input_fraction"] for r in elig + refs],
         "E29 static MBON input fraction (%) of eligible candidates (top 30) + E26 references")
    res = lev["results"]; keys = list(dict.fromkeys(f"{r['kc_type']}->{r['mbon_type']}" for r in res))
    lv = {(f"{r['kc_type']}->{r['mbon_type']}", r["regime"]): r for r in res}
    bars(C / "08-dynamic-leverage-by-candidate.svg", [f"{k} {g}" for k in keys for g in ("60/8", "80/3")],
         [100 * (lv[(k, g)]["rel_change"] or 0) for k in keys for g in ("60/8", "80/3")],
         "E29 MBON change when candidate KC->MBON edges are zeroed (%), seeds 3011-3016; pass <= -10 %", ref=-10.0)
    vrel = lambda k: 100 * abs(vis["classes"][k.split("->")[0]]["stimuli"][vis["classes"][k.split("->")[0]]["best_stimulus"]]["no_visual"]["rel_diff"] or 0)
    scatter(C / "09-visual-vs-dynamic-leverage.svg", [vrel(k) for k in keys], [-100 * (lv[(k, "60/8")]["rel_change"] or 0) for k in keys],
            keys, "E29 KC-class |visual response| vs dynamic leverage at 60/8 (green = visual gate and leverage pass)",
            "KC class best |visual response| vs no-visual (%)", "MBON drop when candidate edges zeroed (%)",
            xref=5.0, yref=10.0, hi=[vis["classes"][k.split("->")[0]]["passes_gate"] and lv[(k, "60/8")]["passes"] for k in keys])
    lab, val = [], []
    for c in cc["circuits"]:
        row = next((d for d in dan["pairs"][c["key"]]["dan_rows"] if d["dan_type"] == c["dan_type"]), None)
        if row:
            for f in ("dan_to_mbon", "dan_to_kc", "kc_to_dan", "mbon_to_dan"):
                lab.append(f"{c['key']} {c['dan_type']} {f}"); val.append(row[f])
    bars(C / "10-dan-connectivity-candidates.svg", lab, val, "E29 raw synapses between each candidate's matched DAN and its KC class / MBON")
    top = [c for c in cc["circuits"] if c["role"] == "candidate"][:3]
    rr = [c for c in cc["circuits"] if c["role"] == "reference"]
    stat = {r["key"]: r["static_input_fraction"] for r in fz["full_ranking"]}
    lab, val = [], []
    for c in rr + top:
        for nm, v in (("static %", 100 * stat[c["key"]]), ("lev 60/8 %", 100 * (c["leverage"]["60/8"]["rel"] or 0)),
                      ("lev 80/3 %", 100 * (c["leverage"]["80/3"]["rel"] or 0))):
            lab.append(f"{c['key']} {nm}"); val.append(v)
    bars(C / "11-reference-418-vs-top-candidates.svg", lab, val, "E29 original E26 circuit (references) vs top-ranked candidates")
    if conf:
        lab, val = [], []
        for k in conf["chosen"]:
            for rg in ("60/8", "80/3"):
                x = conf["per_regime"][rg]["rows"][k]; vb = x["visual"]["stimuli"][x["visual"]["best_stimulus"]]
                for nm, v in (("visual %", 100 * (vb["no_visual"]["rel_diff"] or 0)), ("leverage %", 100 * (x["leverage"]["rel_change"] or 0)),
                              ("DAN pp", x["dan"]["delta_pp"] if x["dan"] else 0)):
                    lab.append(f"{k} {rg} {nm}"); val.append(v)
        bars(C / "12-confirmation-results.svg", [l for l in lab if " 60/8 " in l or " 80/3 leverage" in l],
             [v for l, v in zip(lab, val) if " 60/8 " in l or " 80/3 leverage" in l],
             f"E29 confirmation, seeds 3021-3030 ({'descriptive' if conf['descriptive_confirmation_only'] else 'preregistered'})")
        bars(C / "13-sensitivity-60-8-vs-80-3.svg", lab, val, "E29 two-regime sensitivity: visual %, leverage %, DAN pp at 60/8 and 80/3")
    labs, vals = [], []
    for c in cc["circuits"]:
        s_ok = [bool(c["visual_gate"]), bool(c["dan_match"]), bool(c["leverage"]["60/8"]["passes"]), bool(c["leverage"]["80/3"]["passes"])]
        cf = next((x for x in (conf or {}).get("confirmation", []) if x["key"] == c["key"]), None)
        s_ok.append(bool(cf and cf["confirmed"]))
        labs.append(f"{c['key']} [vis dan lev60/8 lev80/3 conf]={''.join('1' if x else '0' for x in s_ok)}"); vals.append(sum(s_ok))
    bars(C / "14-final-candidate-gate-summary.svg", labs, vals, "E29 gates passed per tested unit (of 5)")
    print(sorted(p.name for p in C.glob("*.svg")))


if __name__ == "__main__":
    main()
