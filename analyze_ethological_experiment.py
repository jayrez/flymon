"""Experiment 16 analysis: Gate A (T4/T5 motion structure) then, only if it passes,
VP and DN feature transfer. Matched-seed statistics on held-out seeds (1121-1140);
preferred directions fixed on calibration seeds (1101-1120).

Loads results/experiment-16-ethological-visual-motor/responses.npz. CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "experiment-16-ethological-visual-motor"
CAPTURES = ROOT / "captures" / "experiment-16"
RNG = np.random.default_rng(16062026)
NPERM = 10000
NBOOT = 10000


def paired(a, b):
    """Matched-seed contrast d = a - b over seeds (same seeds in a and b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    obs = float(d.mean())
    flips = RNG.choice([-1.0, 1.0], size=(NPERM, n))
    null = (flips * d).mean(axis=1)
    p = float((1 + np.sum(np.abs(null) >= abs(obs))) / (NPERM + 1))
    idx = RNG.integers(0, n, size=(NBOOT, n))
    boot = d[idx].mean(axis=1)
    sign = float(np.mean(np.sign(d) == np.sign(obs))) if obs != 0 else 0.0
    return dict(mean=obs, median=float(np.median(d)), n=n, permutation_p=p,
                sign_consistency=sign, boot95=[float(np.quantile(boot, .025)),
                                               float(np.quantile(boot, .975))])


def bar_svg(path, labels, values, title, ref=0.0, errs=None):
    from xml.sax.saxutils import escape
    width = 780; height = 70 + len(labels) * 30
    lo = min(min(values), ref, 0.0); hi = max(max(values), ref, 0.0)
    span = max(hi - lo, 1e-9); x0 = 250; xw = 480
    def X(v): return x0 + xw * (v - lo) / span
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>', f'<text x="15" y="26" font-size="15">{escape(title)}</text>']
    parts.append(f'<line x1="{X(ref):.1f}" y1="40" x2="{X(ref):.1f}" y2="{height-15}" stroke="#c0392b" stroke-dasharray="4"/>')
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 48 + i * 30
        xa, xb = sorted([X(0.0), X(value)])
        parts += [f'<text x="240" y="{y+15}" text-anchor="end" font-size="12">{escape(str(label))}</text>',
                  f'<rect x="{xa:.1f}" y="{y}" width="{max(xb-xa,0.5):.1f}" height="20" fill="#275e85"/>',
                  f'<text x="{width-10}" y="{y+15}" font-size="12" text-anchor="end">{value:.3f}</text>']
    path.write_text('\n'.join(parts + ['</svg>']))


def main():
    CAPTURES.mkdir(parents=True, exist_ok=True)
    npz = np.load(RESULTS / "responses.npz", allow_pickle=False)
    stim = npz["stim_names"].astype(str).tolist()
    seeds = npz["seeds"].tolist()
    cal = set(npz["cal_seeds"].tolist()); held = npz["heldout_seeds"].tolist()
    names = npz["group_names"].astype(str).tolist()
    sizes = npz["group_sizes"].astype(float)
    resp = npz["responses"].astype(float)              # (stim, seed, group)
    rate = resp / sizes[None, None, :]                 # per-neuron mean count
    si = {s: i for i, s in enumerate(stim)}
    gi = {g: i for i, g in enumerate(names)}
    cal_mask = np.array([s in cal for s in seeds])
    held_mask = np.array([s in set(held) for s in seeds])

    def R(stimname, group, mask):
        return rate[si[stimname], mask, gi[group]]

    def has(group): return group in gi

    # ---------- Gate A: motion-vs-frozen (change selectivity) ----------
    motion_vs_frozen = {}
    for base, pol in [("on_left", "ON"), ("off_left", "OFF"), ("off_right", "OFF"), ("loom_dark", "LOOM"), ("flow_expand", "FLOW")]:
        frozen = f"{base}__frozen_first"
        for g in names:
            if g in ("R1_6", "lamina") or g.startswith("T4") or g.startswith("T5"):
                st = paired(R(base, g, held_mask), R(frozen, g, held_mask))
                baseline = float(R(frozen, g, held_mask).mean())
                st["relative_to_baseline"] = abs(st["mean"]) / (baseline + 1e-6)
                st["polarity"] = pol
                motion_vs_frozen[f"{base}|{g}"] = st

    # ---------- Gate A: direction selectivity (preferred fixed on calibration) ----------
    horiz = ("left", "right"); vert = ("up", "down")
    def subtype_dirs(sub):
        pol = "on" if sub.startswith("T4") else "off"
        axis = horiz if sub[2] in ("a", "b") else vert
        return pol, axis
    direction = {}
    preferred = {}
    for sub in [g for g in names if g.startswith(("T4", "T5"))]:
        pol, axis = subtype_dirs(sub)
        a, b = f"{pol}_{axis[0]}", f"{pol}_{axis[1]}"
        # fix preferred direction on calibration seeds
        ma, mb = R(a, sub, cal_mask).mean(), R(b, sub, cal_mask).mean()
        pref, null = (a, b) if ma >= mb else (b, a)
        preferred[sub] = pref
        st = paired(R(pref, sub, held_mask), R(null, sub, held_mask))
        denom = R(pref, sub, held_mask) + R(null, sub, held_mask) + 1e-6
        dsi = float(np.mean((R(pref, sub, held_mask) - R(null, sub, held_mask)) / denom))
        direction[sub] = dict(st, preferred=pref, null=null, dsi=dsi)

    # ---------- PRIMARY endpoint: T5 horizontal OFF direction selectivity ----------
    t5h = [g for g in names if g.startswith("T5") and g[2] in ("a", "b")]
    per_seed_pref_minus_null = np.zeros(held_mask.sum())
    for sub in t5h:
        pref, null = preferred[sub], direction[sub]["null"]
        per_seed_pref_minus_null += (R(pref, sub, held_mask) - R(null, sub, held_mask))
    per_seed_pref_minus_null /= max(len(t5h), 1)
    primary = paired(per_seed_pref_minus_null, np.zeros_like(per_seed_pref_minus_null))
    primary["subtypes"] = t5h
    primary["preferred"] = {s: preferred[s] for s in t5h}

    # ---------- ON/OFF gross separation (T4 ON, T5 OFF) vs matched static ----------
    on_off = {}
    for g in [x for x in names if x.startswith("T4")]:
        on = np.mean([R(f"on_{d}", g, held_mask) for d in ("left", "right", "up", "down")], axis=0)
        on_off[f"T4|{g}|on_vs_gray"] = paired(on, R("gray", g, held_mask))
    for g in [x for x in names if x.startswith("T5")]:
        off = np.mean([R(f"off_{d}", g, held_mask) for d in ("left", "right", "up", "down")], axis=0)
        on_off[f"T5|{g}|off_vs_gray"] = paired(off, R("gray", g, held_mask))

    # ---------- Gate A decision ----------
    # "Biological motion structure" = direction selectivity, or a *non-negligible*
    # change response (>=25% of that population's frozen baseline), in T4/T5 only.
    # (R1-6 / lamina retina responses are recorded but do not decide the optic-lobe gate.)
    REL_MIN = 0.25
    def sig(d): return d["permutation_p"] < 0.05 and d["sign_consistency"] >= 0.7 and abs(d["mean"]) > 0.05
    t45_motion_any_sig = any(sig(v) for k, v in motion_vs_frozen.items()
                             if ("|T4" in k or "|T5" in k))
    t45_motion_meaningful = any(sig(v) and v["relative_to_baseline"] >= REL_MIN
                                for k, v in motion_vs_frozen.items() if ("|T4" in k or "|T5" in k))
    t45_dir = any(sig(v) for k, v in direction.items())
    max_t45_rel = max((v["relative_to_baseline"] for k, v in motion_vs_frozen.items()
                       if "|T4" in k or "|T5" in k), default=0.0)
    max_abs_dsi = max((abs(v["dsi"]) for v in direction.values()), default=0.0)
    r16_motion_sig = any(sig(v) for k, v in motion_vs_frozen.items() if "|R1_6" in k)
    primary_pass = primary["permutation_p"] < 0.05 and primary["sign_consistency"] >= 0.7 and primary["mean"] > 0.05
    gate_a_pass = bool(t45_dir or t45_motion_meaningful)

    gate_status = dict(
        gate_a_pass=gate_a_pass, primary_endpoint_pass=bool(primary_pass),
        criterion=f"T4/T5 direction selectivity, or change response >= {REL_MIN} of frozen baseline",
        t4t5_direction_selective_significant=bool(t45_dir),
        t4t5_motion_change_any_significant=bool(t45_motion_any_sig),
        t4t5_motion_change_meaningful=bool(t45_motion_meaningful),
        max_t4t5_motion_relative_effect=float(max_t45_rel),
        max_t4t5_abs_dsi=float(max_abs_dsi),
        retina_r16_motion_significant=bool(r16_motion_sig),
        primary=dict(name="T5 horizontal OFF direction selectivity (held-out)",
                     mean=primary["mean"], permutation_p=primary["permutation_p"],
                     sign_consistency=primary["sign_consistency"], boot95=primary["boot95"]),
        decision="proceed_to_vp_dn" if gate_a_pass else "stop_at_gate_a",
        category="A" if not gate_a_pass else "pending_stage_B_C")

    # ---------- Exploratory (recorded regardless; NOT validated if Gate A fails) ----------
    exploratory = {}
    if has("LPLC2_R"):
        for g in ["LPLC2_L", "LPLC2_R", "LPLC1", "VS"]:
            if has(g):
                exploratory[f"loom_vs_recede|{g}"] = paired(R("loom_dark", g, held_mask), R("recede_dark", g, held_mask))
    for g in ["DNp01_L", "DNp01_R", "DNp03_R", "DNp04_R", "DNp11_R"]:
        if has(g):
            exploratory[f"loom_vs_recede|{g}"] = paired(R("loom_dark", g, held_mask), R("recede_dark", g, held_mask))
            exploratory[f"loom_vs_static|{g}"] = paired(R("loom_dark", g, held_mask), R("dark_disc", g, held_mask))
    # DNg13 / DNa02 horizontal-flow lateral asymmetry (R side pref flow_right - flow_left)
    for g in ["DNg13_L", "DNg13_R", "DNa02_L", "DNa02_R"]:
        if has(g):
            exploratory[f"flowR_vs_flowL|{g}"] = paired(R("flow_right", g, held_mask), R("flow_left", g, held_mask))
    for g in ["DNg100", "MDN"]:
        if has(g):
            exploratory[f"flow_vs_gray|{g}"] = paired(
                np.mean([R(f, g, held_mask) for f in ("flow_left", "flow_right", "flow_expand", "flow_contract")], axis=0),
                R("gray", g, held_mask))
    # dynamic vs frozen and ordered vs shuffle at DNp01 (exploratory confound check)
    for g in ["DNp01_R"]:
        exploratory[f"loom_dynamic_vs_frozen|{g}"] = paired(R("loom_dark", g, held_mask), R("loom_dark__frozen_first", g, held_mask))
        exploratory[f"loom_ordered_vs_shuffle|{g}"] = paired(R("loom_dark", g, held_mask), R("loom_dark__shuffle", g, held_mask))

    # ---------- flyvis benchmark (skipped) ----------
    try:
        import flyvis  # noqa: F401
        flyvis_status = dict(run=False, note="flyvis import succeeded but benchmark not implemented in this run")
    except Exception as e:
        flyvis_status = dict(run=False, reason=f"flyvis not installed ({type(e).__name__}); dependency deliberately not forced")
    (RESULTS / "flyvis-benchmark.json").write_text(json.dumps(flyvis_status, indent=2) + "\n")

    stats = dict(seeds_calibration=list(sorted(cal)), seeds_heldout=list(held),
                 primary=primary, direction_selectivity=direction, motion_vs_frozen=motion_vs_frozen,
                 on_off_separation=on_off, exploratory=exploratory, preferred_directions=preferred)
    (RESULTS / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    (RESULTS / "gate-status.json").write_text(json.dumps(gate_status, indent=2) + "\n")

    # rate table for the report
    key_stims = ["gray", "dark", "on_left", "off_left", "off_right", "loom_dark", "recede_dark", "flow_expand"]
    key_pops = ["R1_6", "lamina", "T4a_L", "T4b_L", "T5a_L", "T5b_L", "LPLC2_R", "VS", "DNp01_R", "DNg13_R"]
    table = {p: {s: float(R(s, p, held_mask).mean()) for s in key_stims} for p in key_pops if has(p)}
    (RESULTS / "controlled-results.json").write_text(json.dumps(dict(
        held_out_mean_rate=table, gate_status=gate_status), indent=2) + "\n")

    # ---------- figures ----------
    t4_labels = [g for g in names if g.startswith("T4")]
    bar_svg(CAPTURES / "t4-on-direction.svg", t4_labels, [direction[g]["dsi"] for g in t4_labels],
            "Exp16 T4 ON direction-selectivity index (held-out)", ref=0.0)
    t5_labels = [g for g in names if g.startswith("T5")]
    bar_svg(CAPTURES / "t5-off-direction.svg", t5_labels, [direction[g]["dsi"] for g in t5_labels],
            "Exp16 T5 OFF direction-selectivity index (held-out)", ref=0.0)
    mvf_labels = [k.split("|")[1] for k in motion_vs_frozen if k.startswith("off_left|") and (k.split("|")[1].startswith(("T4", "T5")))]
    bar_svg(CAPTURES / "t5-direction-selectivity.svg",
            [g for g in names if g.startswith("T5")],
            [direction[g]["mean"] for g in names if g.startswith("T5")],
            "Exp16 T5 preferred-null OFF response (held-out, matched-seed)", ref=0.0)
    loom_labels = [k.split("|")[1] for k in exploratory if k.startswith("loom_vs_recede|")]
    bar_svg(CAPTURES / "loom-transfer.svg", loom_labels,
            [exploratory[f"loom_vs_recede|{g}"]["mean"] for g in loom_labels],
            "Exp16 loom-vs-recede response (held-out, EXPLORATORY)", ref=0.0)
    flow_labels = [g for g in ["DNg13_L", "DNg13_R", "DNa02_L", "DNa02_R"] if has(g)]
    bar_svg(CAPTURES / "dng13-dna02-flow.svg", flow_labels,
            [exploratory[f"flowR_vs_flowL|{g}"]["mean"] for g in flow_labels],
            "Exp16 DNg13/DNa02 flow(right-left) asymmetry (held-out, EXPLORATORY)", ref=0.0)
    dvf = [("dynamic_vs_frozen", exploratory["loom_dynamic_vs_frozen|DNp01_R"]["mean"]),
           ("ordered_vs_shuffle", exploratory["loom_ordered_vs_shuffle|DNp01_R"]["mean"])]
    bar_svg(CAPTURES / "dynamic-vs-frozen.svg", [a for a, _ in dvf], [b for _, b in dvf],
            "Exp16 DNp01 loom dynamic/ordered contrasts (held-out, EXPLORATORY)", ref=0.0)
    bar_svg(CAPTURES / "ordered-vs-shuffled.svg", [a for a, _ in dvf], [b for _, b in dvf],
            "Exp16 DNp01 loom ordered/dynamic contrasts (held-out, EXPLORATORY)", ref=0.0)

    print("=== Gate A ===", flush=True)
    print(f"  retina R1-6 motion-vs-frozen significant: {r16_motion_sig}", flush=True)
    print(f"  T4/T5 motion change any-significant: {t45_motion_any_sig} | meaningful(>= {REL_MIN}x base): {t45_motion_meaningful} | max rel {max_t45_rel:.3f}", flush=True)
    print(f"  T4/T5 direction-selective significant: {t45_dir} | max |DSI| {max_abs_dsi:.4f}", flush=True)
    print(f"  PRIMARY T5 OFF dir-selectivity: mean={primary['mean']:.4f} "
          f"p={primary['permutation_p']:.4f} sign={primary['sign_consistency']:.2f} "
          f"boot95={[round(x,4) for x in primary['boot95']]}", flush=True)
    print(f"  GATE A PASS: {gate_a_pass}  ->  category {gate_status['category']}", flush=True)
    print("  (exploratory) DNp01_R loom vs recede:",
          {k: round(v['mean'], 3) for k, v in exploratory.items() if k == 'loom_vs_recede|DNp01_R'}, flush=True)
    return gate_status


if __name__ == "__main__":
    main()
