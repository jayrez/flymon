"""Experiment 22 analysis: preregistered criteria, category, attribution, statistics, figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_tonic_disinhibition_experiment import _svg, bars, boot_ci, histogram, holm, panel_traces, sign_p
from flymon import adapted_reference as ar

ROOT = Path(__file__).resolve().parent
R = ROOT / "results" / "experiment-22-fixed-background-adaptation"
CAP = ROOT / "captures" / "experiment-22"
SUBTYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
T4, T5 = SUBTYPES[:4], SUBTYPES[4:]
THRESH_T4, SECONDARY_T4 = 0.15, 0.08
GEOM_MAX, GEOM_STRONG, TEMPORAL_MIN, STABLE_MAX = 0.40, 0.30, 0.40, 0.05
REFS = ("R0", "R0g", "R1", "R2", "R3")


def load(name):
    return json.loads((R / name).read_text())


def drop(obs, flat):
    return (obs - flat) / obs if obs > 0 else None


def criteria(held):
    h = held["heldout"]; s = h["summary"]; cp = h["correct_polarity"]
    abl = held["ablations"]; init = held["initial_condition"]
    geo = abl["B5_geometry_shuffle"]["residual_fraction"]
    tdrop = drop(s["t4_mean"], abl["B4_temporal_flat"]["t4_mean"])
    ad, cold = init["adapted_40"], init["cold_filters"]
    c = dict(
        t4_on_4of4=bool(sum(cp[x] for x in T4) == 4),
        t4_mean_ge_015=bool(s["t4_mean"] >= THRESH_T4),
        t4_positive_4of4=bool(s["t4_positive"] == 4),
        geometry_residual_lt_040=bool(geo is not None and geo < GEOM_MAX),
        temporal_drop_ge_040=bool(tdrop is not None and tdrop >= TEMPORAL_MIN),
        stable_under_preadaptation=bool(abs(ad["t4_mean"] - cold["t4_mean"]) <= STABLE_MAX
                                        and ad["t4_on_correct"] == 4 and cold["t4_on_correct"] == 4))
    vals = dict(t4_mean=s["t4_mean"], t4_positive=s["t4_positive"],
                t4_on_correct=int(sum(cp[x] for x in T4)), geometry_residual=geo,
                temporal_drop=tdrop, adapted_minus_cold=ad["t4_mean"] - cold["t4_mean"])
    return c, vals


def category(held, c, v):
    refs = held["reference_conditions"]["E21_params"]
    s = held["heldout"]["summary"]
    if all(c.values()):
        return "A", "PASS"
    if s["t4_mean"] >= THRESH_T4 and s["t4_positive"] == 4 and not c["t4_on_4of4"]:
        return "C", "PARTIAL"
    if c["t4_on_4of4"] and s["t4_positive"] == 4 and s["t4_mean"] >= 0.05:
        return "B", "PARTIAL"
    collapse = lambda r: r["t4_mean"] < 0.05 or r["t4_positive"] < 3
    if collapse(refs["R1"]) and collapse(refs["R2"]) and (s["t4_mean"] < 0.05 or s["t4_positive"] < 3):
        return "D", "FAIL"
    return "E", "FAIL"


def attribution(held):
    refs = held["reference_conditions"]["E21_params"]
    r0, r2 = refs["R0"]["t4_mean"], refs["R2"]["t4_mean"]
    ret = r2 / r0 if r0 > 0 else None
    on_ok = refs["R2"]["t4_on_correct"] == 4
    if ret is None:
        label = "undefined"
    elif ret >= 0.67 and on_ok:
        label = "mostly real"
    elif ret >= 0.33:
        label = "partly real"
    else:
        label = "mostly artefact"
    return dict(retention_R2_over_R0=ret, R2_t4_on_correct=refs["R2"]["t4_on_correct"], label=label)


def readiness(cat, c):
    if cat == "A":
        return "GO"
    if cat == "B" and c["geometry_residual_lt_040"] and c["temporal_drop_ge_040"]:
        return "CONDITIONAL GO"
    return "NO-GO"


def subtype_stats():
    z = np.load(R / "heldout-per-neuron.npz")
    rows, ps = {}, []
    for s in SUBTYPES:
        d, e = z[f"e22_{s}_dsi"], z[f"e22_{s}_effect"]
        rows[s] = dict(n=int(len(d)), dsi=float(d.mean()), ci=boot_ci(d), p_sign=sign_p(e))
        ps.append(rows[s]["p_sign"])
    for s, a in zip(SUBTYPES, holm(ps)):
        rows[s]["p_holm"] = float(a); rows[s]["significant"] = bool(a < 0.05 and rows[s]["dsi"] > 0)
    return rows


# ---------------------------------------------------------------- figures
def schematic(path, cfg, L0):
    p = _svg(900, 300, "Exp22: fixed adapting background, pre-adaptation, tonic single compartment")
    p += ['<rect x="20" y="60" width="250" height="110" rx="10" fill="#f4f4f4" stroke="#555"/>',
          '<text x="145" y="85" font-size="12" text-anchor="middle">stimulus on background L0</text>',
          f'<text x="145" y="105" font-size="11" text-anchor="middle">L0 = {L0:.4f} (linear light)</text>',
          '<text x="145" y="125" font-size="11" text-anchor="middle">bars at Weber +/- c</text>',
          f'<text x="145" y="145" font-size="11" text-anchor="middle">{cfg["adapt"]} neutral frames first (unscored)</text>',
          '<rect x="320" y="60" width="250" height="110" rx="10" fill="#eef4f8" stroke="#275e85"/>',
          '<text x="445" y="85" font-size="12" text-anchor="middle">s = polarity x (L - L0), low-pass</text>',
          '<text x="445" y="105" font-size="11" text-anchor="middle">R0 control: s = L - mean_t L</text>',
          f'<text x="445" y="125" font-size="11" text-anchor="middle">r = max(0, r0 + beta s / sigma)</text>',
          f'<text x="445" y="145" font-size="11" text-anchor="middle">r0_exc={cfg["r0_exc"]}, Mi9={cfg["r0_mi9"]}, Mi4={cfg["r0_mi4"]}</text>',
          '<rect x="620" y="60" width="250" height="110" rx="10" fill="#fdf2e9" stroke="#e67e22"/>',
          '<text x="745" y="85" font-size="12" text-anchor="middle">passive compartment (E21)</text>',
          f'<text x="745" y="105" font-size="11" text-anchor="middle">G={cfg["G"]}, E_inh={cfg["E_inh"]}</text>',
          '<text x="745" y="125" font-size="11" text-anchor="middle">output = max(V - V_rest, 0)</text>',
          '<text x="745" y="145" font-size="11" text-anchor="middle">V starts at adapted rest</text>',
          '<line x1="270" y1="115" x2="320" y2="115" stroke="#333" stroke-width="2"/>',
          '<line x1="570" y1="115" x2="620" y2="115" stroke="#333" stroke-width="2"/>',
          '<text x="20" y="220" font-size="11" fill="#555">Frozen configuration shown. Class-level parameters only; no opponent product; no per-subtype parameters.</text>']
    path.write_text("\n".join(p + ["</svg>"]))


def adaptation_timeline(path):
    L0 = 0.5
    lum = np.concatenate([np.full((20, 1), L0), np.full((40, 1), 0.75)]).astype(np.float32)
    adapted = ar.cell_signals(lum, 4.0, 12.0, "fixed", L0, 40)
    cold = ar.cell_signals(lum, 4.0, 12.0, "fixed", L0, 0)
    seq = ar.cell_signals(lum, 4.0, 12.0, "sequence_mean")
    pad = np.full(40, np.nan)
    panel_traces(path, [
        ("luminance (40 adaptation + 60 stimulus frames)", [("L", np.concatenate([np.full(40, L0), lum[:, 0]]), "#333333")]),
        ("Mi1 signal: fixed ref, adapted (R2)", [("R2", adapted["Mi1"][:, 0], "#275e85")]),
        ("Mi1 signal: sequence mean (R0)", [("R0", np.concatenate([np.zeros(40), seq["Mi1"][:, 0]]), "#c0392b")]),
        ("Mi9 (slow) signal: fixed ref, adapted", [("R2", adapted["Mi9"][:, 0], "#275e85")]),
        ("Mi9 (slow) signal: sequence mean", [("R0", np.concatenate([np.zeros(40), seq["Mi9"][:, 0]]), "#c0392b")]),
        ("Mi1 signal: fixed ref, cold filters (R1)", [("R1", np.concatenate([np.zeros(40), cold["Mi1"][:, 0]]), "#e67e22")]),
    ], "Exp22 adaptation timeline: a full-field increment at stimulus frame 20 (frames 0-39 = neutral adaptation, unscored)")
    del pad


def main():
    CAP.mkdir(parents=True, exist_ok=True)
    sel = load("selected-config.json"); held = load("heldout-results.json")
    cal = load("calibration-grid.json"); gates = load("gate-outcomes.json"); audit = load("polarity-audit.json")
    abl = held["ablations"]; refs = held["reference_conditions"]
    c, v = criteria(held)
    cat, verdict = category(held, c, v)
    attr = attribution(held)
    ready = readiness(cat, c)
    rows = subtype_stats()
    secondary = bool(c["t4_on_4of4"] and c["t4_positive_4of4"] and SECONDARY_T4 <= v["t4_mean"] < THRESH_T4
                     and c["geometry_residual_lt_040"] and c["temporal_drop_ge_040"])
    stats = dict(frozen_model=sel["selected"], selection_status=sel["selection_status"],
                 subtypes=rows, criteria=c, criterion_values=v, secondary_mechanistic_lead=secondary,
                 unit="neuron (deterministic front end)", bootstrap=10000, correction="Holm over 8 subtypes")
    (R / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    ver = dict(scientific_verdict=verdict, mechanistic_category=cat, gameplay_readiness=ready,
               e21_lead_attribution=attr, criteria=c, criterion_values=v,
               polarity_audit_passed=all(a["passed"] for k, a in audit.items() if k.startswith(("R1", "R2"))),
               r0_reproduced=bool(sel["r0_reproduction"]["matches"]
                                  and refs["E21_params"]["R0"].get("matches_e21_saved_heldout")),
               gate_counts=gates["counts"])
    (R / "verdict.json").write_text(json.dumps(ver, indent=2) + "\n")

    # ---------------- figures ----------------
    cfg = sel["config"]
    schematic(CAP / "01-schematic.svg", cfg, sel["L0_measured"])
    adaptation_timeline(CAP / "02-adaptation-timeline.svg")
    tz = np.load(R / "traces.npz")
    panel_traces(CAP / "03-step-responses.svg",
                 [(f"{cls} signal", [("ON step", tz[f"signal_{cls}_step_ON"], "#275e85"),
                                     ("OFF step", tz[f"signal_{cls}_step_OFF"], "#c0392b")])
                  for cls in ("Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9")]
                 + [(f"{s} output", [("ON step", tz[f"{s}_step_ON"], "#275e85"),
                                     ("OFF step", tz[f"{s}_step_OFF"], "#c0392b")]) for s in ("T4a", "T5a")],
                 "Exp22 step-polarity audit: full-field increments/decrements at frame 20 (R2, frozen model)")
    labs, vals = [], []
    for pset in ("E21_params", "frozen_params"):
        for r in REFS:
            labs.append(f"{pset} {r}"); vals.append(refs[pset][r]["t4_mean"])
    bars(CAP / "04-reference-conditions-dsi.svg", labs, vals,
         "Exp22 held-out T4 mean DSI by contrast reference (R0 seq-mean E19 bank; R0g seq-mean gray bank; R1 fixed; R2 fixed+adapted; R3 R2 without tonic excitation)",
         ref=THRESH_T4)
    labs, vals = [], []
    for r in ("R0", "R2"):
        for s in T4:
            o = refs["E21_params"][r]["on_off"][s]; labs.append(f"E21 params {r} {s}"); vals.append(o["on"] - o["off"])
    bars(CAP / "05-t4-on-off.svg", labs, vals, "Exp22 T4 ON minus OFF mean response (should be > 0)")
    labs, vals = [], []
    for r in ("R0", "R2"):
        for s in T5:
            o = refs["E21_params"][r]["on_off"][s]; labs.append(f"E21 params {r} {s}"); vals.append(o["off"] - o["on"])
    bars(CAP / "06-t5-on-off.svg", labs, vals, "Exp22 T5 OFF minus ON mean response (should be > 0; descriptive)")
    init = held["initial_condition"]
    bars(CAP / "07-cold-vs-adapted.svg",
         [f"frozen {k}" for k in init] + ["E21 params R1 (cold filters)", "E21 params R2 (adapted)"],
         [init[k]["t4_mean"] for k in init] + [refs["E21_params"]["R1"]["t4_mean"], refs["E21_params"]["R2"]["t4_mean"]],
         "Exp22 initial condition: held-out T4 mean DSI")
    g = abl["B5_geometry_shuffle"]
    histogram(CAP / "08-geometry-null.svg", g["null_samples"], g["observed"],
              "Exp22 geometry shuffle null, 200 permutations (frozen model, held-out speed axis, T4)")
    labs, vals = [], []
    for pset in ("E21_params", "frozen_params"):
        for r in REFS:
            labs += [f"{pset} {r}", f"{pset} {r} flat"]
            vals += [refs[pset][r]["t4_mean"], refs[pset][r]["temporal_flat_t4"]]
    bars(CAP / "09-temporal-flat.svg", labs, vals, "Exp22 temporal flattening (tau_slow = tau_fast), held-out T4 mean DSI")
    dec = held["inhibition_decomposition"]
    bars(CAP / "10-inhibition-decomposition.svg", list(dec), [dec[k]["t4_mean"] for k in dec],
         "Exp22 inhibition decomposition around the frozen model, held-out T4 mean DSI")
    oc = held["oracle_comparison"]
    comp = [("E19 oracle (E19 bank)", oc["oracle_e19_bank_saved"]["t4_mean"]),
            ("E19 oracle (gray bank)", oc["oracle_gray_bank"]["t4_mean"]),
            ("E20 C2 frozen", oc["e20_c2_saved"]["summary"]["t4_mean"]),
            ("E21 frozen", oc["e21_frozen_saved"]["t4_mean"]),
            ("E21 exploratory A1 (R0)", oc["e21_exploratory_a1_saved"]["t4_mean"]),
            ("E22 E21-params R2", refs["E21_params"]["R2"]["t4_mean"]),
            ("E22 frozen", held["heldout"]["summary"]["t4_mean"])]
    bars(CAP / "11-e21-lead-vs-e22-vs-oracle.svg", [a for a, _ in comp], [b for _, b in comp],
         "Exp22 held-out T4 mean DSI: prior models, E22 and the E19 oracle (benchmark only)", ref=THRESH_T4)
    panel_traces(CAP / "12-pref-null-traces.svg",
                 [(s, [("pref", tz[f"{s}_pref"], "#275e85"), ("null", tz[f"{s}_null"], "#c0392b")])
                  for s in SUBTYPES if f"{s}_pref" in tz.files],
                 "Exp22 frozen model, held-out bars at 1.5 px/frame (T4: ON, T5: OFF), neuron-averaged output")
    st = abl["B10_static_steps"]; so = abl["B9_controls"]["static_on_off"]
    bars(CAP / "13-static-step-controls.svg",
         [f"{s} step ON-OFF" for s in SUBTYPES] + [f"{s} static bar ON-OFF" for s in SUBTYPES]
         + ["static frozen-mid T4 DSI"],
         [st[s]["ON"] - st[s]["OFF"] for s in SUBTYPES] + [so[s]["on"] - so[s]["off"] for s in SUBTYPES]
         + [abl["B9_controls"]["static_frozen_mid_t4_dsi"]],
         "Exp22 static controls (frozen model): full-field steps, frozen-mid bars, static DSI")

    print(json.dumps(ver, indent=2))
    for s in SUBTYPES:
        r = rows[s]
        print(f"{s} DSI {r['dsi']:+.4f} CI [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p_holm {r['p_holm']:.3g}")


if __name__ == "__main__":
    main()
