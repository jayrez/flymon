"""Experiment 17 analysis: gain sweep, temporal audit, and the frozen held-out
evaluation of the selected optic-lobe dynamics configuration.

Preferred directions come from calibration seeds only (selected-config.json).
Held-out seeds 1221-1240 are evaluated once. CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "experiment-17-optic-lobe-dynamics"
CAPTURES = ROOT / "captures" / "experiment-17"
RNG = np.random.default_rng(17062026)
NPERM = 10000
NBOOT = 10000
E16_MAX_DSI = 0.012

EXC_T4 = ("Mi1", "Tm3")
EXC_T5 = ("Tm1", "Tm2", "Tm4", "Tm9")
INH = ("Mi9", "Mi4", "CT1")


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d); obs = float(d.mean())
    flips = RNG.choice([-1.0, 1.0], size=(NPERM, n))
    null = (flips * d).mean(axis=1)
    p = float((1 + np.sum(np.abs(null) >= abs(obs))) / (NPERM + 1))
    idx = RNG.integers(0, n, size=(NBOOT, n))
    boot = d[idx].mean(axis=1)
    sign = float(np.mean(np.sign(d) == np.sign(obs))) if obs != 0 else 0.0
    return dict(mean=obs, median=float(np.median(d)), n=n, permutation_p=p,
                sign_consistency=sign,
                boot95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))])


def bar_svg(path, labels, values, title, ref=0.0):
    from xml.sax.saxutils import escape
    width = 820; height = 70 + len(labels) * 28
    lo = min(min(values), ref, 0.0); hi = max(max(values), ref, 0.0)
    span = max(hi - lo, 1e-12); x0 = 250; xw = 500
    X = lambda v: x0 + xw * (v - lo) / span
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="15" y="26" font-size="15">{escape(title)}</text>',
         f'<line x1="{X(0.0):.1f}" y1="40" x2="{X(0.0):.1f}" y2="{height-12}" stroke="#999"/>']
    if ref:
        p.append(f'<line x1="{X(ref):.1f}" y1="40" x2="{X(ref):.1f}" y2="{height-12}" '
                 f'stroke="#c0392b" stroke-dasharray="4"/>')
    for i, (lb, v) in enumerate(zip(labels, values)):
        y = 46 + i * 28
        xa, xb = sorted([X(0.0), X(v)])
        p += [f'<text x="240" y="{y+14}" text-anchor="end" font-size="11">{escape(str(lb))}</text>',
              f'<rect x="{xa:.1f}" y="{y}" width="{max(xb-xa,0.5):.1f}" height="18" fill="#275e85"/>',
              f'<text x="{width-8}" y="{y+14}" font-size="11" text-anchor="end">{v:.4f}</text>']
    path.write_text("\n".join(p + ["</svg>"]))


def line_svg(path, series, title, ylabel=""):
    from xml.sax.saxutils import escape
    W, H, L, T = 820, 320, 60, 40
    colors = ["#275e85", "#c0392b", "#27ae60", "#8e44ad", "#e67e22", "#16a085"]
    n = max(len(v) for _, v in series)
    ymax = max(float(np.max(v)) for _, v in series) or 1.0
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="15" y="24" font-size="15">{escape(title)}</text>',
         f'<line x1="{L}" y1="{H-40}" x2="{W-20}" y2="{H-40}" stroke="#333"/>',
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-40}" stroke="#333"/>',
         f'<text x="8" y="{T+10}" font-size="10">{ymax:.2f}</text>',
         f'<text x="8" y="{H-42}" font-size="10">0</text>',
         f'<text x="{L}" y="{H-24}" font-size="10">step 0</text>',
         f'<text x="{W-60}" y="{H-24}" font-size="10">step {n}</text>']
    for k, (name, v) in enumerate(series):
        pts = " ".join(f"{L+(W-20-L)*i/max(n-1,1):.1f},{(H-40)-(H-40-T)*float(x)/ymax:.1f}"
                       for i, x in enumerate(v))
        p.append(f'<polyline fill="none" stroke="{colors[k%len(colors)]}" stroke-width="1.6" points="{pts}"/>')
        p.append(f'<text x="{W-200}" y="{T+14+k*15}" font-size="11" fill="{colors[k%len(colors)]}">{escape(name)}</text>')
    path.write_text("\n".join(p + ["</svg>"]))


AXIS = {"a": ("left", "right"), "b": ("left", "right"), "c": ("up", "down"), "d": ("up", "down")}


def subtype_pair(sub):
    pol = "on" if sub.startswith("T4") else "off"
    a, b = AXIS[sub[2]]
    return f"{pol}_{a}", f"{pol}_{b}"


def main():
    CAPTURES.mkdir(parents=True, exist_ok=True)
    sel = json.loads((RESULTS / "selected-config.json").read_text())
    preferred = sel["preferred_directions"]
    npz = np.load(RESULTS / "heldout-responses.npz", allow_pickle=False)
    stim = npz["stim_names"].astype(str).tolist()
    seeds = npz["seeds"].tolist()
    names = npz["group_names"].astype(str).tolist()
    si = {s: i for i, s in enumerate(stim)}; gi = {g: i for i, g in enumerate(names)}
    cfg_names = [k[5:] for k in npz.files if k.startswith("resp_")]
    resp = {c: npz[f"resp_{c}"] for c in cfg_names}
    T4S = [g for g in names if g.startswith("T4")]
    T5S = [g for g in names if g.startswith("T5")]

    def R(cfg, s, g):
        return resp[cfg][si[s], :, gi[g]]

    out = {}
    for cfg in cfg_names:
        dsi = {}
        for sub in T4S + T5S:
            a, b = subtype_pair(sub)
            if sub not in preferred:
                continue
            pref = preferred[sub]; null = b if pref == a else a
            rp, rn = R(cfg, pref, sub), R(cfg, null, sub)
            st = paired(rp, rn)
            dsi[sub] = dict(st, preferred=pref, null=null,
                            dsi=float(np.mean((rp - rn) / (rp + rn + 1e-6))),
                            pref_rate=float(rp.mean()), null_rate=float(rn.mean()))
        # PRIMARY: T5 horizontal aggregate
        t5h = [s for s in T5S if s[2] in ("a", "b") and s in dsi]
        per_seed = np.mean([R(cfg, dsi[s]["preferred"], s) - R(cfg, dsi[s]["null"], s) for s in t5h], axis=0)
        primary = paired(per_seed, np.zeros_like(per_seed))
        primary["dsi"] = float(np.mean([dsi[s]["dsi"] for s in t5h]))
        primary["subtypes"] = t5h
        # controls
        ctrl = {}
        for g in ["T5a_L", "T5b_L", "T4a_L", "R1_6", "L1"]:
            if g not in gi:
                continue
            ctrl[f"dynamic_vs_frozen|{g}"] = paired(R(cfg, "off_left", g), R(cfg, "off_left__frozen_first", g))
            ctrl[f"ordered_vs_shuffled|{g}"] = paired(R(cfg, "off_left", g), R(cfg, "off_left__shuffle", g))
            ctrl[f"forward_vs_reversed|{g}"] = paired(R(cfg, "off_left", g), R(cfg, "off_left__reverse", g))
        # ON/OFF distinction: T4 to ON motion vs T5 to OFF motion, each vs gray
        onoff = {}
        for g in T4S:
            on = np.mean([R(cfg, f"on_{d}", g) for d in ("left", "right", "up", "down")], axis=0)
            onoff[f"{g}|on_vs_gray"] = paired(on, R(cfg, "gray", g))
        for g in T5S:
            off = np.mean([R(cfg, f"off_{d}", g) for d in ("left", "right", "up", "down")], axis=0)
            onoff[f"{g}|off_vs_gray"] = paired(off, R(cfg, "gray", g))
        rates = {g: float(resp[cfg][:, :, gi[g]].mean()) for g in names}
        t45 = np.array([rates[g] for g in T4S + T5S])
        out[cfg] = dict(dsi=dsi, primary=primary, controls=ctrl, on_off=onoff, rates=rates,
                        max_abs_t4_dsi=float(max(abs(dsi[g]["dsi"]) for g in T4S if g in dsi)),
                        max_abs_t5_dsi=float(max(abs(dsi[g]["dsi"]) for g in T5S if g in dsi)),
                        firing_regime=dict(t4t5_mean=float(t45.mean()), t4t5_max=float(t45.max()),
                                           saturated=bool(t45.max() > 100), silent=bool(t45.mean() < 0.2)))

    # ---- temporal audit ----
    temporal = {}
    tp = RESULTS / "temporal-traces.npz"
    if tp.exists():
        tz = np.load(tp, allow_pickle=False)
        tnames = tz["group_names"].astype(str).tolist()
        tgi = {g: i for i, g in enumerate(tnames)}
        tsz = tz["sizes"]
        for key in [k for k in tz.files if "|" in k]:
            tr = tz[key] / tsz[None, :]
            def arm(fams):
                cols = [tgi[f] for f in fams if f in tgi]
                return tr[:, cols].mean(axis=1) if cols else np.zeros(tr.shape[0])
            e4, e5, inh = arm(EXC_T4), arm(EXC_T5), arm(INH)
            def peak(v):
                v = v - v.mean()
                return int(np.argmax(v)) if np.any(v) else -1
            def lag(x, y, maxlag=20):
                x = x - x.mean(); y = y - y.mean()
                if not np.any(x) or not np.any(y):
                    return 0
                cc = [float(np.dot(x[max(0, l):len(x)+min(0, l)], y[max(0, -l):len(y)+min(0, -l)]))
                      for l in range(-maxlag, maxlag + 1)]
                return int(np.argmax(cc) - maxlag)
            temporal[key] = dict(peak_exc_t4=peak(e4), peak_exc_t5=peak(e5), peak_inh=peak(inh),
                                 lag_exc_t4_vs_inh=lag(e4, inh), lag_exc_t5_vs_inh=lag(e5, inh),
                                 exc_t5_mean=float(e5.mean()), inh_mean=float(inh.mean()))
        (RESULTS / "temporal-dynamics.json").write_text(json.dumps(dict(
            note="per-step traces, mean over 5 calibration seeds; lag in brain steps, "
                 "positive = excitatory arm leads the inhibitory arm",
            families=dict(exc_t4=list(EXC_T4), exc_t5=list(EXC_T5), inhibitory=list(INH)),
            traces=temporal), indent=2) + "\n")

    # ---- gate status / verdict ----
    ref = out.get("reference"); rep = out.get(sel["selected"])
    prim = rep["primary"]
    passed = (prim["permutation_p"] < 0.05 and prim["sign_consistency"] >= 0.7
              and prim["mean"] > 0 and abs(prim["dsi"]) > E16_MAX_DSI * 3
              and not rep["firing_regime"]["saturated"] and not rep["firing_regime"]["silent"])
    gain = json.loads((RESULTS / "gain-sweep.json").read_text())["results"]
    gain_restored = max(abs(v["max_abs_t5_dsi"]) for v in gain.values()) > E16_MAX_DSI * 3
    pc = json.loads((RESULTS / "positive-control.json").read_text())
    stim_ok = abs(pc["off_horizontal_dsi"]) > 0.5 and pc["sign_flips_with_direction"]["off"]
    category = ("B" if gain_restored else "C" if passed else
                "D" if (prim["permutation_p"] < 0.05 and abs(prim["dsi"]) > E16_MAX_DSI) else
                ("E" if stim_ok else "A"))
    verdict = "PASS" if (passed or gain_restored) else ("PARTIAL" if category == "D" else "FAIL")
    gate = dict(verdict=verdict, category=category,
                stimulus_adequacy_supported=bool(stim_ok),
                gain_restored_selectivity=bool(gain_restored),
                dynamics_repair_passed=bool(passed),
                selected_config=sel["selected"],
                primary=dict(mean=prim["mean"], dsi=prim["dsi"], permutation_p=prim["permutation_p"],
                             sign_consistency=prim["sign_consistency"], boot95=prim["boot95"]),
                reference_primary=dict(mean=ref["primary"]["mean"], dsi=ref["primary"]["dsi"],
                                       permutation_p=ref["primary"]["permutation_p"]),
                experiment16_max_abs_dsi=E16_MAX_DSI,
                repaired_max_abs_t5_dsi=rep["max_abs_t5_dsi"],
                repaired_max_abs_t4_dsi=rep["max_abs_t4_dsi"],
                proceed_to_vp_check=bool(passed))
    (RESULTS / "gate-status.json").write_text(json.dumps(gate, indent=2) + "\n")
    (RESULTS / "heldout-results.json").write_text(json.dumps(dict(
        seeds=seeds, configs=cfg_names, selected=sel["selected"], results=out), indent=2) + "\n")
    (RESULTS / "statistics.json").write_text(json.dumps(dict(
        primary_endpoint=prim, per_subtype_dsi={c: {k: v["dsi"] for k, v in out[c]["dsi"].items()} for c in cfg_names},
        controls={c: out[c]["controls"] for c in cfg_names},
        on_off={c: out[c]["on_off"] for c in cfg_names},
        firing_regime={c: out[c]["firing_regime"] for c in cfg_names}), indent=2) + "\n")

    # ---- figures ----
    gl = sorted(gain, key=float)
    bar_svg(CAPTURES / "gain-vs-t5-rate.svg", [f"{g}x" for g in gl],
            [gain[g]["t5_rate"] for g in gl], "Exp17 Gate 1: T5 firing rate vs retinal gain")
    bar_svg(CAPTURES / "gain-vs-dsi.svg", [f"{g}x" for g in gl],
            [gain[g]["max_abs_t5_dsi"] for g in gl],
            "Exp17 Gate 1: max |T5 DSI| vs retinal gain (E16 = 0.012)", ref=E16_MAX_DSI)
    subs = [g for g in T5S if g in rep["dsi"]]
    bar_svg(CAPTURES / "dsi-by-subtype.svg", subs, [rep["dsi"][g]["dsi"] for g in subs],
            f"Exp17 held-out T5 DSI by subtype ({sel['selected']})")
    subs4 = [g for g in T4S if g in rep["dsi"]]
    bar_svg(CAPTURES / "t4-on-direction.svg", subs4, [rep["dsi"][g]["dsi"] for g in subs4],
            f"Exp17 held-out T4 DSI by subtype ({sel['selected']})")
    bar_svg(CAPTURES / "reference-vs-repaired.svg",
            ["E16 reference max|DSI|", "E17 reference max|T5 DSI|", "E17 repaired max|T5 DSI|"],
            [E16_MAX_DSI, ref["max_abs_t5_dsi"], rep["max_abs_t5_dsi"]],
            "Exp17: reference vs repaired direction selectivity")
    bar_svg(CAPTURES / "pref-null-comparison.svg",
            [f"{g} pref-null" for g in subs], [rep["dsi"][g]["mean"] for g in subs],
            f"Exp17 held-out preferred-minus-null spikes ({sel['selected']})")
    pcl = ["off_left", "off_right", "on_left", "on_right", "gray", "frozen", "shuffle", "reverse"]
    pcv = [pc["hr_response"][k] if k in pc["hr_response"] else pc["controls"][k] for k in
           ["off_left", "off_right", "on_left", "on_right", "gray", "frozen", "shuffle", "reverse"]]
    bar_svg(CAPTURES / "positive-control-hr.svg", pcl, pcv,
            "Exp17 Gate 0: Hassenstein-Reichardt control on the same retina samples")
    if tp.exists():
        tz = np.load(tp, allow_pickle=False)
        tnames = tz["group_names"].astype(str).tolist(); tgi = {g: i for i, g in enumerate(tnames)}
        tsz = tz["sizes"]
        for cfgn, fname in ((sel["selected"], "t5-temporal-traces.svg"), ("reference", "t4-temporal-traces.svg")):
            key = f"{cfgn}|off_left"
            if key in tz.files:
                tr = tz[key] / tsz[None, :]
                series = [(g, tr[:, tgi[g]]) for g in ("R1_6", "L1", "Tm2", "Mi9", "T5a_L", "T4a_L") if g in tgi]
                line_svg(CAPTURES / fname, series, f"Exp17 per-step traces ({cfgn}, off_left)")

    print(f"VERDICT {verdict}  category {category}")
    print(f"  stimulus adequacy supported (HR control): {stim_ok}")
    print(f"  gain restored selectivity: {gain_restored}")
    print(f"  selected config: {sel['selected']}")
    print(f"  PRIMARY held-out T5 OFF pref-null: mean {prim['mean']:+.4f} spike, DSI {prim['dsi']:+.5f}, "
          f"p={prim['permutation_p']:.4f}, sign {prim['sign_consistency']:.2f}, boot95 {[round(x,4) for x in prim['boot95']]}")
    print(f"  reference held-out primary: mean {ref['primary']['mean']:+.4f}, DSI {ref['primary']['dsi']:+.5f}, p={ref['primary']['permutation_p']:.4f}")
    print(f"  max|T5 DSI| reference {ref['max_abs_t5_dsi']:.4f} -> repaired {rep['max_abs_t5_dsi']:.4f} (E16 {E16_MAX_DSI})")
    print(f"  firing regime repaired: {rep['firing_regime']}")


if __name__ == "__main__":
    main()
