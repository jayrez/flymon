"""Experiment 28 analysis: response surfaces, descriptive trade-offs, 12 figures (no new simulation)."""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from scipy import stats

import run_mb_joint_operating_point as J
from analyze_tonic_disinhibition_experiment import _svg, bars

ROOT = Path(__file__).resolve().parent
R = J.OUT
C = ROOT / "captures" / "experiment-28"
SUBS = ("R1", "R2", "R3", "R4_PAM", "R4_PPL1", "R5_MBON01", "R5_MBON11", "R6")


def load(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else None


def surface(rows, f):
    """5 x 5 array indexed [refractory row, APL column]."""
    m = np.full((len(J.REFRACTORY_MS), len(J.APL_GAINS)), np.nan)
    for r in rows:
        i = J.REFRACTORY_MS.index(int(r["refractory_ms"])); j = J.APL_GAINS.index(r["apl_gain"])
        m[i, j] = f(r)
    return m


def heatmap(path, m, title, fmt="{:+.1f}", passes=None, diverging=False):
    """Refractory (rows) x APL gain (columns) response surface; outlined cells pass the criterion."""
    cw, ch, x0, y0 = 110, 56, 150, 70
    W, H = max(960, x0 + cw * m.shape[1] + 20), y0 + ch * m.shape[0] + 50
    p = _svg(W, H, title)
    finite = m[np.isfinite(m)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if diverging:
        a = max(abs(lo), abs(hi), 1e-12); lo, hi = -a, a
    sp = max(hi - lo, 1e-12)
    for j, g in enumerate(J.APL_GAINS):
        p.append(f'<text x="{x0 + cw * j + cw / 2}" y="{y0 - 8}" text-anchor="middle" font-size="12">APL x{g:g}</text>')
    for i, r in enumerate(J.REFRACTORY_MS):
        p.append(f'<text x="{x0 - 8}" y="{y0 + ch * i + ch / 2 + 4}" text-anchor="end" font-size="12">'
                 f'{r} ms ({J.refractory_steps(r)} st)</text>')
        for j in range(m.shape[1]):
            v = m[i, j]; t = (v - lo) / sp if np.isfinite(v) else 0.5
            if diverging:
                col = f"rgb({int(255 - 120 * max(0, t - .5) * 2)},{int(255 - 150 * abs(t - .5) * 2)},{int(255 - 120 * max(0, .5 - t) * 2)})"
            else:
                col = f"rgb({int(245 - 200 * t)},{int(248 - 140 * t)},{int(255 - 90 * t)})"
            ok = passes is not None and passes[i, j]
            p += [f'<rect x="{x0 + cw * j}" y="{y0 + ch * i}" width="{cw - 2}" height="{ch - 2}" fill="{col}" '
                  f'stroke="{"#1e8449" if ok else "#fff"}" stroke-width="{4 if ok else 1}"/>',
                  f'<text x="{x0 + cw * j + cw / 2}" y="{y0 + ch * i + ch / 2 + 4}" text-anchor="middle" font-size="13">'
                  f'{escape(fmt.format(v)) if np.isfinite(v) else "-"}</text>']
    p.append(f'<text x="{x0}" y="{H - 16}" font-size="11">green outline = criterion passes at this point; '
             f'rows: requested refractory (effective steps); columns: APL output gain</text>')
    path.write_text("\n".join(p + ["</svg>"]))


def scatter(path, xs, ys, labels, title, xl, yl, xref=None, yref=None, hi=None):
    W, H, L, T, Wd, Hd = 900, 620, 80, 50, 760, 500
    lo_x, hi_x = min(list(xs) + [xref or 0, 0]), max(list(xs) + [xref or 0, 0])
    lo_y, hi_y = min(list(ys) + [yref or 0, 0]), max(list(ys) + [yref or 0, 0])
    X = lambda v: L + Wd * (v - lo_x) / max(hi_x - lo_x, 1e-12)
    Y = lambda v: T + Hd - Hd * (v - lo_y) / max(hi_y - lo_y, 1e-12)
    p = _svg(W, H, title) + [f'<rect x="{L}" y="{T}" width="{Wd}" height="{Hd}" fill="none" stroke="#999"/>']
    if xref is not None:
        p.append(f'<line x1="{X(xref):.1f}" y1="{T}" x2="{X(xref):.1f}" y2="{T + Hd}" stroke="#c0392b" stroke-dasharray="4"/>')
    if yref is not None:
        p.append(f'<line x1="{L}" y1="{Y(yref):.1f}" x2="{L + Wd}" y2="{Y(yref):.1f}" stroke="#c0392b" stroke-dasharray="4"/>')
    for k, (x, y, lb) in enumerate(zip(xs, ys, labels)):
        col = "#1e8449" if hi and hi[k] else "#275e85"
        p += [f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="5" fill="{col}"/>',
              f'<text x="{X(x) + 7:.1f}" y="{Y(y) - 5:.1f}" font-size="10">{escape(lb)}</text>']
    p += [f'<text x="{L + Wd / 2}" y="{H - 12}" text-anchor="middle" font-size="12">{escape(xl)}</text>',
          f'<text x="18" y="{T + Hd / 2}" font-size="12" transform="rotate(-90 18 {T + Hd / 2})" text-anchor="middle">{escape(yl)}</text>',
          f'<text x="{L}" y="{T + Hd + 16}" font-size="10">{lo_x:+.2f}</text>',
          f'<text x="{L + Wd}" y="{T + Hd + 16}" font-size="10" text-anchor="end">{hi_x:+.2f}</text>',
          f'<text x="{L - 4}" y="{T + Hd}" font-size="10" text-anchor="end">{lo_y:+.2f}</text>',
          f'<text x="{L - 4}" y="{T + 10}" font-size="10" text-anchor="end">{hi_y:+.2f}</text>']
    path.write_text("\n".join(p + ["</svg>"]))


def tradeoffs(rows):
    """Descriptive Spearman correlations across the 25 grid points (no inference about causation)."""
    v = dict(refractory_steps=[r["refractory_steps"] for r in rows], apl_gain=[r["apl_gain"] for r in rows],
             kcgd_rate=[r["kcgd_rate"] for r in rows], kcgd_norm_duty=[r["kcgd_norm_duty"] for r in rows],
             visual_abs_rel=[abs(r["visual_rel"]) for r in rows], visual_consistent=[r["visual_consistent"] for r in rows],
             pam_pp=[r["pam_pp"] for r in rows], ppl_pp=[r["ppl_pp"] for r in rows],
             mbon01_leverage=[-r["mbon01_lev"] for r in rows], mbon11_leverage=[-r["mbon11_lev"] for r in rows],
             min_leverage=[-max(r["mbon01_lev"], r["mbon11_lev"]) for r in rows])
    keys = list(v); corr = {}
    for a in keys:
        for b in keys:
            if a < b:
                rho, p = stats.spearmanr(v[a], v[b])
                corr[f"{a}~{b}"] = dict(rho=None if np.isnan(rho) else float(rho), p_descriptive=None if np.isnan(p) else float(p))
    sets = {k: sorted(r["key"] for r in rows if r["criteria"][k]) for k in SUBS + ("R4", "R5", "all_pass")}
    overlap = dict(R3_and_R5=sorted(set(sets["R3"]) & set(sets["R5"])), R3_and_R4=sorted(set(sets["R3"]) & set(sets["R4"])),
                   R4_and_R5=sorted(set(sets["R4"]) & set(sets["R5"])),
                   R3_R4_R5=sorted(set(sets["R3"]) & set(sets["R4"]) & set(sets["R5"])))
    return dict(note="descriptive Spearman correlations across the 25 calibration grid points; p-values are "
                     "descriptive only (no multiple-comparison correction, points are not independent samples)",
                variables={k: dict(values=v[k]) for k in keys}, spearman=corr, passing_sets=sets, overlaps=overlap,
                requested={f"{a} vs {b}": corr.get(f"{min(a, b)}~{max(a, b)}", {}).get("rho") for a, b in (
                    ("visual_abs_rel", "mbon01_leverage"), ("visual_abs_rel", "mbon11_leverage"), ("visual_abs_rel", "kcgd_rate"),
                    ("visual_abs_rel", "apl_gain"), ("mbon01_leverage", "refractory_steps"), ("mbon11_leverage", "refractory_steps"),
                    ("pam_pp", "refractory_steps"), ("ppl_pp", "refractory_steps"), ("pam_pp", "apl_gain"), ("ppl_pp", "apl_gain"))},
                headline=dict(visual_vs_min_leverage=corr.get("min_leverage~visual_abs_rel"),
                              kcgd_rate_vs_visual=corr.get("kcgd_rate~visual_abs_rel"),
                              kcgd_rate_vs_min_leverage=corr.get("kcgd_rate~min_leverage")))


def main():
    C.mkdir(parents=True, exist_ok=True)
    for f in C.glob("*.svg"):
        f.unlink()
    g = load("grid-results.json"); rows = [J.summary_row(r) for r in g["results"]]
    tr = tradeoffs(rows); J.jdump("tradeoff-analysis.json", tr)
    crit = lambda k: surface(rows, lambda r: bool(r["criteria"][k])).astype(bool)
    lbl = [f"{int(r['refractory_ms'])}/{r['apl_gain']:g}" for r in rows]
    # ---- the 12 required figures
    heatmap(C / "01-visual-modulation-surface.svg", surface(rows, lambda r: 100 * r["visual_rel"]),
            "E28 best KCg-d visual modulation vs no-visual (%); R3 = |rel|>=5% vs no-visual & gray, >=5/6 seeds",
            "{:+.2f}%", crit("R3"), diverging=True)
    heatmap(C / "02-pam01-headroom-surface.svg", surface(rows, lambda r: r["pam_pp"]),
            "E28 PAM01 response to fixed DAN amplitude 0.3 (pp duty); R4_PAM needs >= 10", "{:+.1f}", crit("R4_PAM"))
    heatmap(C / "03-ppl101-headroom-surface.svg", surface(rows, lambda r: r["ppl_pp"]),
            "E28 PPL101 response to fixed DAN amplitude 0.3 (pp duty); R4_PPL1 needs >= 10", "{:+.1f}", crit("R4_PPL1"))
    heatmap(C / "04-mbon01-leverage-surface.svg", surface(rows, lambda r: 100 * r["mbon01_lev"]),
            "E28 MBON01 change when all 418 plastic edges are zeroed (%); R5_MBON01 needs <= -10%", "{:+.2f}%",
            crit("R5_MBON01"), diverging=True)
    heatmap(C / "05-mbon11-leverage-surface.svg", surface(rows, lambda r: 100 * r["mbon11_lev"]),
            "E28 MBON11 change when all 418 plastic edges are zeroed (%); R5_MBON11 needs <= -10%", "{:+.2f}%",
            crit("R5_MBON11"), diverging=True)
    heatmap(C / "06-criteria-count-surface.svg", surface(rows, lambda r: sum(bool(r["criteria"][k]) for k in SUBS)),
            "E28 sub-criteria passed (of 8: R1 R2 R3 R4_PAM R4_PPL1 R5_MBON01 R5_MBON11 R6); green = all pass",
            "{:.0f}/8", crit("all_pass"))
    for n, m in (("07", "mbon01"), ("08", "mbon11")):
        scatter(C / f"{n}-visual-vs-{m}-leverage.svg", [100 * abs(r["visual_rel"]) for r in rows],
                [-100 * r[f"{m}_lev"] for r in rows], lbl,
                f"E28 KCg-d |visual modulation| vs {m.upper()} leverage (labels ms/APL; green = R3 and R5_{m.upper()})",
                "best |visual modulation| vs no-visual (%)", f"{m.upper()} drop when 418 edges zeroed (%)",
                xref=5.0, yref=10.0, hi=[r["criteria"]["R3"] and r["criteria"][f"R5_{m.upper()}"] for r in rows])
    order = sorted(rows, key=lambda r: (r["refractory_steps"], r["apl_gain"]))
    bars(C / "09-dan-headroom-by-point.svg", [f"{x} {d}" for x in [f"{int(r['refractory_ms'])}ms APLx{r['apl_gain']:g}" for r in order]
                                               for d in ("PAM01", "PPL101")],
         [v for r in order for v in (r["pam_pp"], r["ppl_pp"])],
         "E28 DAN response at fixed amplitude 0.3 (pp duty) across refractory x APL; R4 needs >= 10 per DAN", ref=10.0)
    an = load("anchor-reproduction.json")["anchors"]
    stock = J.summary_row(an["ref0ms|apl1|kcg1|sens1"]["result"])
    e27 = next(r for r in rows if r["key"] == "ref20ms|apl5|kcg1|sens1")
    best = load("best-nonpassing-candidate.json") or load("frozen-operating-point.json")
    e28 = next(r for r in rows if r["key"] == best["key"])
    mets = (("KCg-d Hz", lambda x: x["kcgd_rate"]), ("vis %", lambda x: 100 * x["visual_rel"]), ("PAM pp", lambda x: x["pam_pp"]),
            ("PPL pp", lambda x: x["ppl_pp"]), ("MBON01 zero %", lambda x: 100 * x["mbon01_lev"]),
            ("MBON11 zero %", lambda x: 100 * x["mbon11_lev"]), ("criteria /8", lambda x: sum(bool(x["criteria"][k]) for k in SUBS)))
    bars(C / "10-stock-vs-e27best-vs-e28best.svg",
         [f"{n} {m}" for n in ("stock", "E27-best 20/5", f"E28-best {int(e28['refractory_ms'])}/{e28['apl_gain']:g}") for m, _ in mets],
         [f(x) for x in (stock, e27, e28) for _, f in mets],
         "E28: stock (E27 seeds 2801-2806) vs E27-best and E28-best (E28 calibration seeds 2901-2906)")
    conf = load("confirmation-results.json")
    cr, st = J.summary_row(conf["result"]), J.summary_row(conf["stock_same_seeds"])
    tag = "descriptive best non-passing" if conf["descriptive_confirmation_only"] else "frozen"
    bars(C / "11-confirmation-metrics.svg", [f"{n} {m}" for n in ("stock", tag) for m, _ in mets],
         [f(x) for x in (st, cr) for _, f in mets], f"E28 confirmation seeds 2911-2920: stock vs {tag} point {cr['key']}")
    calc = next(r for r in rows if r["key"] == cr["key"])
    bars(C / "12-final-gate-summary.svg",
         [f"{k} {w}" for k in SUBS for w in ("grid points passing /25", "best cal", "best conf")],
         [v for k in SUBS for v in (len(tr["passing_sets"][k]), int(calc["criteria"][k]), int(cr["criteria"][k]))],
         f"E28 gate summary: points passing each criterion (of 25); {cr['key']} calibration/confirmation pass (1/0); all-pass points: "
         f"{len(tr['passing_sets']['all_pass'])}")
    # ---- supplementary
    heatmap(C / "13-kcgd-rate-surface.svg", surface(rows, lambda r: r["kcgd_rate"]), "E28 KCg-d baseline rate (Hz)", "{:.2f}",
            crit("R2"))
    heatmap(C / "14-kcgd-ceiling-duty-surface.svg", surface(rows, lambda r: r["kcgd_norm_duty"]),
            "E28 KCg-d ceiling-normalised duty (R1 needs < 0.80 for all targets)", "{:.3f}", crit("R1"))
    heatmap(C / "15-visual-consistency-surface.svg", surface(rows, lambda r: r["visual_consistent"]),
            "E28 seeds (of 6) with consistent sign for the best visual contrast vs no-visual", "{:.0f}/6", crit("R3"))
    lab, val = [], []
    for k, a in an.items():
        c = a["comparison"]
        for nm, pr, sc in (("KCg-d Hz", c["kcgd_rate"], 1), ("vis %", c["visual_best_rel"], 100), ("zero-edge %", c["target_zeroed"], 100)):
            lab += [f"{k.split('|kcg')[0]} {nm} E27", f"{k.split('|kcg')[0]} {nm} E28"]; val += [sc * pr[1], sc * pr[0]]
    bars(C / "16-anchor-reproduction.svg", lab, val, "E28 anchor reproduction of E27 (E27 calibration seeds 2801-2806)")
    print(json.dumps(dict(overlaps=tr["overlaps"], requested=tr["requested"],
                          passing={k: len(v) for k, v in tr["passing_sets"].items()}), indent=1))


if __name__ == "__main__":
    main()
