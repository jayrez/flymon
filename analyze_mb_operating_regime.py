"""Experiment 27 analysis: candidate ranking, freeze decision, figures (stock always included)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analyze_tonic_disinhibition_experiment import bars

ROOT = Path(__file__).resolve().parent
R = ROOT / "results" / "experiment-27-mb-operating-regime"
C = ROOT / "captures" / "experiment-27"
CRIT = ("R1", "R2", "R3", "R4", "R5", "R6")


def load(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else None


def row(r):
    P = r["baseline"]["populations"]; v = r["visual"]; best = v["best_condition"] if v else None
    return dict(key=r["key"], op=r["operating_point"], criteria=r["criteria"], passes=sum(bool(r["criteria"][c]) for c in CRIT),
                kcgd_rate=P["KCg-d"]["rate_hz"], kcgd_norm_duty=P["KCg-d"]["ceiling_normalised_duty"],
                mbon01_rate=P["MBON01"]["rate_hz"], mbon11_rate=P["MBON11"]["rate_hz"],
                pam01_rate=P["PAM01"]["rate_hz"], ppl101_rate=P["PPL101"]["rate_hz"],
                visual_best=best, visual_rel=(v["contrasts"][best]["no_visual"]["rel_diff"] if v else None),
                visual_consistent=(v["contrasts"][best]["no_visual"]["consistent_seeds"] if v else None),
                visual_dz=(v["contrasts"][best]["no_visual"]["dz"] if v else None),
                visual_cells_modulated=(v["contrasts"][best]["fraction_cells_modulated"] if v else None),
                pam_headroom_pp=r["dan"]["at_rule_amplitude"]["appetitive"]["delta_pp"],
                ppl_headroom_pp=r["dan"]["at_rule_amplitude"]["aversive"]["delta_pp"],
                dan_rule_amplitude=r["dan"]["rule_amplitude"],
                mbon01_zero=r["leverage"]["per_population"]["MBON01"]["zeroed_rel"],
                mbon11_zero=r["leverage"]["per_population"]["MBON11"]["zeroed_rel"],
                target_zero=r["leverage"]["target_zeroed_rel"], target_lower_seeds=r["leverage"]["target_lower_seeds"],
                whole_brain_at_ceiling=r["baseline"]["whole_brain_fraction_at_ceiling"],
                whole_brain_silent=r["baseline"]["whole_brain_fraction_silent"])


def rank_key(x):
    c = x["criteria"]
    from run_mb_operating_regime import OperatingPoint
    op = OperatingPoint(**x["op"])
    return (not (c["R1"] and c["R2"]), not c["R3"], not c["R4"], not c["R5"], not c["R6"], op.magnitude())


def main():
    C.mkdir(parents=True, exist_ok=True)
    stock = load("stock-reproduction.json")["result"]
    sweeps = {f: load(f"{f}-sweep.json") for f in ("refractory", "apl", "kcgain")}
    rows = {f: [row(r) for r in s["results"]] for f, s in sweeps.items() if s}
    everything = [x for f in rows for x in rows[f]]
    passing = [x for x in everything if all(x["criteria"][c] for c in CRIT)]
    ranked = sorted(everything, key=rank_key)
    decision = dict(stages_run=list(rows), stages_skipped=[f for f in ("refractory", "apl", "kcgain") if f not in rows],
                    passing=[x["key"] for x in passing], ranked=[x["key"] for x in ranked],
                    best=ranked[0]["key"], frozen=(sorted(passing, key=rank_key)[0]["key"] if passing else None))
    (R / "candidate-ranking.json").write_text(json.dumps(dict(stock=row(stock), families=rows, decision=decision), indent=1) + "\n")
    if "--freeze" in sys.argv:
        pick = sorted(passing, key=rank_key)[0] if passing else ranked[0]
        (R / "frozen-operating-point.json").write_text(json.dumps(dict(
            operating_point=pick["op"], key=pick["key"], passes_all=bool(passing),
            status=("least-invasive point passing R1-R6 (preregistered)" if passing else
                    "NO POINT PASSED: best-ranked point frozen for descriptive confirmation only"),
            calibration_criteria=pick["criteria"]), indent=1) + "\n")
    # ---------------- figures (stock always included) ----------------
    ref = rows.get("refractory", [])
    labels = [x["key"].split("|")[0] for x in ref]
    bars(C / "01-mb-duty-vs-refractory.svg", [f"{l} {p}" for l in labels for p in ("KCg-d", "MBON01", "PAM01", "PPL101")],
         [x[k] for x in ref for k in ("kcgd_rate", "mbon01_rate", "pam01_rate", "ppl101_rate")],
         "E27 baseline firing rate (Hz) vs refractory (0 ms = stock)")
    bars(C / "02-kcgd-visual-modulation-vs-refractory.svg", labels, [100 * (x["visual_rel"] or 0) for x in ref],
         "E27 best KCg-d visual modulation (% vs no visual) vs refractory; R3 needs >= 5%", ref=5.0)
    bars(C / "03-dan-headroom-vs-refractory.svg", [f"{l} {c}" for l in labels for c in ("PAM01", "PPL101")],
         [x[k] for x in ref for k in ("pam_headroom_pp", "ppl_headroom_pp")],
         "E27 DAN stimulation headroom (percentage points of duty); R4 needs >= 10", ref=10.0)
    bars(C / "04-mbon-zero-edge-effect-vs-refractory.svg", [f"{l} {c}" for l in labels for c in ("MBON01", "MBON11", "target")],
         [100 * (x[k] or 0) for x in ref for k in ("mbon01_zero", "mbon11_zero", "target_zero")],
         "E27 effect of zeroing all 418 plastic edges (%); R5 needs <= -10%", ref=-10.0)
    for f in ("apl", "kcgain"):
        if f in rows:
            fl = [x["key"] for x in rows[f]]
            bars(C / f"05-{f}-sweep.svg", [f"{k} {m}" for k in fl for m in ("vis%", "PAMpp", "PPLpp", "zero%")],
                 [v for x in rows[f] for v in (100 * (x["visual_rel"] or 0), x["pam_headroom_pp"], x["ppl_headroom_pp"],
                                               100 * (x["target_zero"] or 0))],
                 f"E27 {f} sweep (base refractory 20 ms): visual %, DAN headroom pp, zero-edge %")
    best = ranked[0]
    bars(C / "06-stock-vs-best.svg", [f"{n} {m}" for n in ("stock", "best") for m in ("KCg-d Hz", "vis%", "PAMpp", "zero%")],
         [v for x in (row(stock), best) for v in (x["kcgd_rate"], 100 * (x["visual_rel"] or 0), x["pam_headroom_pp"],
                                                  100 * (x["target_zero"] or 0))],
         f"E27 stock vs best-ranked point ({best['key']})")
    bars(C / "07-criteria-passed.svg", [x["key"] for x in [row(stock)] + everything],
         [x["passes"] for x in [row(stock)] + everything], "E27 number of criteria R1-R6 passed per operating point")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
