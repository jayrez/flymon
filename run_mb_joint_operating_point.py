"""Experiment 28: joint refractory x APL operating-point search (no learning).

Stages (results/experiment-28-mb-joint-operating-point/preregistration.md):
  anchors   reproduce E27 anchor points on E27 calibration seeds (stock, 20 ms+APL5, 60 ms)
  grid      evaluate every preregistered joint point on calibration seeds 2901-2906
  select    score R1-R6 per point, freeze the least-invasive all-pass point or an explicit null
  confirm   run once on confirmation seeds 2911-2920 (frozen point, or descriptive best non-passing)

Reuses the E27 harness and assays (baseline, visual, leverage) unchanged. R4 uses a fixed
DAN amplitude of 0.3 (dan_assay_fixed); E27's adaptive rule_amplitude is not used. R5 is
scored per MBON. The connectome, 418-edge plastic set, T4 path and stimuli are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import run_mb_operating_regime as E27
from flymon import mb_plasticity as mp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiment-28-mb-joint-operating-point"
CAL_SEEDS = tuple(range(2901, 2907))
CONF_SEEDS = tuple(range(2911, 2921))
DT_MS = 20.0
DAN_AMPLITUDE = 0.3
WARMUP = 200
BASELINE_WINDOW = 100
# requested 20,30,40,50,60 ms collapse to steps 1,2,2,2,3 (round(1.5)=round(2.5)=2); 30 and 50 were
# replaced prospectively by 80 and 100 ms -> unique effective steps 1..5
REFRACTORY_MS = (20, 40, 60, 80, 100)
APL_GAINS = (3.0, 4.0, 5.0, 6.0, 8.0)
R1_MAX, R3_MIN_REL, R4_MIN_PP, R5_MAX_REL, MIN_SEEDS = 0.80, 0.05, 10.0, -0.10, 5


def refractory_steps(ms):
    return int(round((ms / 1000.0) / (DT_MS / 1000.0)))


def grid():
    pts = [E27.OperatingPoint(refractory_ms=float(r), apl_gain=g) for r in REFRACTORY_MS for g in APL_GAINS]
    return pts


def grid_config():
    rows = [dict(key=p.key(), requested_refractory_ms=p.refractory_ms, refractory_steps=refractory_steps(p.refractory_ms),
                 effective_refractory_ms=refractory_steps(p.refractory_ms) * DT_MS,
                 ceiling_hz=1000.0 / DT_MS / (refractory_steps(p.refractory_ms) + 1), apl_gain=p.apl_gain)
            for p in grid()]
    return dict(dt_ms=DT_MS, rows=rows, n=len(rows),
                recommended_requested_ms=[20, 30, 40, 50, 60],
                recommended_mapping={r: refractory_steps(r) for r in (20, 30, 40, 50, 60)},
                resolution="30 and 50 ms map to 2 steps (same as 40 ms); replaced prospectively by 80 and 100 ms")


# ---------------------------------------------------------------- fixed-amplitude DAN assay (R4)
def dan_assay_fixed(h, seeds, amplitude=DAN_AMPLITUDE):
    """Warm-up, steady-state baseline, then DAN stimulation at ONE fixed amplitude per channel."""
    out = {}
    for ch, g in (("appetitive", "PAM01"), ("aversive", "PPL101")):
        base, stim = [], []
        for s in seeds:
            r = h.run_schedule(s, [dict(n=WARMUP, tag="warmup"), dict(n=BASELINE_WINDOW, tag="baseline"),
                                   dict(n=E27.E26.DAN_WINDOW, dan=(ch, amplitude, 0, E27.E26.DAN_WINDOW), tag="dan")],
                               mp.PlasticState.zeros(h.edges))
            n = len(h.dan[ch])
            base.append(r[1]["counts"][g] / n / BASELINE_WINDOW)
            stim.append(r[2]["counts"][g] / n / E27.E26.DAN_WINDOW)
        base, stim = np.array(base), np.array(stim)
        d = stim - base
        out[g] = dict(amplitude=amplitude, baseline_duty=float(base.mean()), stimulated_duty=float(stim.mean()),
                      delta_pp=float(100 * d.mean()), rel_delta=float(d.mean() / base.mean()) if base.mean() > 0 else None,
                      per_seed_delta_pp=(100 * d).tolist(), positive_seeds=int((d > 0).sum()),
                      passes=bool(100 * d.mean() >= R4_MIN_PP and (d > 0).sum() >= MIN_SEEDS))
    return out


# ---------------------------------------------------------------- criteria (per population)
def score(base, visual, dan, lev):
    P = base["populations"]
    r1 = all(P[p]["ceiling_normalised_duty"] < R1_MAX for p in E27.TARGETS)
    rates = [P[p]["rate_hz"] for p in E27.TARGETS]
    r2 = bool(np.median(rates) > 1.0 and min(rates) > 0.0)
    r3 = bool(visual["R3"])
    r5 = {}
    for m in ("MBON01", "MBON11"):
        pp = lev["per_population"][m]
        b, z = np.array(pp["baseline"]), np.array(pp["zeroed"])
        rel = float((z.mean() - b.mean()) / b.mean()) if b.mean() else None
        r5[m] = dict(zeroed_rel=rel, lower_seeds=int((z < b).sum()),
                     passes=bool(rel is not None and rel <= R5_MAX_REL and (z < b).sum() >= MIN_SEEDS))
    r6 = bool(base["finite"] and base["whole_brain_fraction_at_ceiling"] < 0.01 and base["whole_brain_fraction_silent"] < 0.5)
    crit = dict(R1=bool(r1), R2=r2, R3=r3, R4_PAM=dan["PAM01"]["passes"], R4_PPL1=dan["PPL101"]["passes"],
                R5_MBON01=r5["MBON01"]["passes"], R5_MBON11=r5["MBON11"]["passes"], R6=r6)
    crit["R4"] = bool(crit["R4_PAM"] and crit["R4_PPL1"])
    crit["R5"] = bool(crit["R5_MBON01"] and crit["R5_MBON11"])
    crit["all_pass"] = bool(all(crit[k] for k in ("R1", "R2", "R3", "R4", "R5", "R6")))
    return crit, r5


def provenance(h):
    base = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return dict(edge_hash=h.edges.sha256, t4_sha256=h.t4.sha256(), base_w_sha256=h.w_hash, base_commit=base)


def evaluate_point(op, seeds):
    t0 = time.perf_counter()
    h = E27.make_harness(op)
    base = E27.baseline_assay(h, seeds)
    vis = E27.visual_assay(h, seeds)
    dan = dan_assay_fixed(h, seeds)
    lev = E27.leverage_assay(h, seeds)
    crit, r5 = score(base, vis, dan, lev)
    out = dict(operating_point=asdict(op), key=op.key(), refractory_steps=refractory_steps(op.refractory_ms),
               seeds=list(seeds), baseline=base, visual=vis, dan_fixed=dan, leverage=lev, r5=r5, criteria=crit,
               provenance=provenance(h), seconds=time.perf_counter() - t0)
    del h
    return out


# ---------------------------------------------------------------- selection
def selection_key(row):
    """All-pass first; then fewest refractory steps; then lowest APL gain; then requested magnitude."""
    op = row["operating_point"]
    return (not row["criteria"]["all_pass"], row["refractory_steps"], op["apl_gain"], op["refractory_ms"])


def select(rows):
    passing = [r for r in rows if r["criteria"]["all_pass"]]
    ranked = sorted(rows, key=selection_key)
    return passing, ranked


def nonpassing_key(row):
    c = row["criteria"]
    npass = sum(bool(c[k]) for k in ("R1", "R2", "R3", "R4_PAM", "R4_PPL1", "R5_MBON01", "R5_MBON11", "R6"))
    return (-npass, not c["R3"], not c["R5"], not c["R4"], row["refractory_steps"], row["operating_point"]["apl_gain"])


def summary_row(r):
    P = r["baseline"]["populations"]; v = r["visual"]; b = v["best_condition"]
    return dict(key=r["key"], refractory_ms=r["operating_point"]["refractory_ms"], refractory_steps=r["refractory_steps"],
                apl_gain=r["operating_point"]["apl_gain"], kcgd_rate=P["KCg-d"]["rate_hz"],
                kcgd_norm_duty=P["KCg-d"]["ceiling_normalised_duty"], visual_best=b,
                visual_rel=v["contrasts"][b]["no_visual"]["rel_diff"], visual_consistent=v["contrasts"][b]["no_visual"]["consistent_seeds"],
                visual_dz=v["contrasts"][b]["no_visual"]["dz"], cells_modulated=v["contrasts"][b]["fraction_cells_modulated"],
                pam_pp=r["dan_fixed"]["PAM01"]["delta_pp"], ppl_pp=r["dan_fixed"]["PPL101"]["delta_pp"],
                mbon01_lev=r["r5"]["MBON01"]["zeroed_rel"], mbon11_lev=r["r5"]["MBON11"]["zeroed_rel"],
                criteria=r["criteria"])


def jdump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def brief(r):
    s = summary_row(r)
    c = s["criteria"]
    flags = "".join("1" if c[k] else "0" for k in ("R1", "R2", "R3", "R4_PAM", "R4_PPL1", "R5_MBON01", "R5_MBON11", "R6"))
    return (f"KCg-d {s['kcgd_rate']:.2f}Hz vis {100 * (s['visual_rel'] or 0):+.2f}% ({s['visual_best']},{s['visual_consistent']}) "
            f"PAM {s['pam_pp']:+.1f}pp PPL {s['ppl_pp']:+.1f}pp M01 {100 * s['mbon01_lev']:+.2f}% "
            f"M11 {100 * s['mbon11_lev']:+.2f}% [{flags}] all={c['all_pass']}")


# ---------------------------------------------------------------- stages
def stage_anchors():
    e27 = {f: json.loads((E27.OUT / f"{f}-sweep.json").read_text()) for f in ("refractory", "apl")}
    old = {r["key"]: r for f in e27.values() for r in f["results"]}
    rows = {}
    for op in (E27.OperatingPoint(), E27.OperatingPoint(refractory_ms=20, apl_gain=5.0), E27.OperatingPoint(refractory_ms=60)):
        r = evaluate_point(op, E27.CAL_SEEDS)
        o = old[op.key()]
        cmp = dict(kcgd_rate=(r["baseline"]["populations"]["KCg-d"]["rate_hz"], o["baseline"]["populations"]["KCg-d"]["rate_hz"]),
                   visual_best_rel=(r["visual"]["contrasts"][o["visual"]["best_condition"]]["no_visual"]["rel_diff"],
                                    o["visual"]["contrasts"][o["visual"]["best_condition"]]["no_visual"]["rel_diff"]),
                   target_zeroed=(r["leverage"]["target_zeroed_rel"], o["leverage"]["target_zeroed_rel"]),
                   old_dan_grid_0p3=(o["dan"]["grid"].get("appetitive|0.3", {}).get("delta_pp"),
                                     o["dan"]["grid"].get("aversive|0.3", {}).get("delta_pp")),
                   new_fixed_0p3=(r["dan_fixed"]["PAM01"]["delta_pp"], r["dan_fixed"]["PPL101"]["delta_pp"]))
        cmp["identical_core"] = bool(cmp["kcgd_rate"][0] == cmp["kcgd_rate"][1] and cmp["target_zeroed"][0] == cmp["target_zeroed"][1]
                                     and cmp["visual_best_rel"][0] == cmp["visual_best_rel"][1])
        rows[op.key()] = dict(comparison=cmp, result=r)
        print(f"[anchor] {op.key():<26} {brief(r)}\n   {json.dumps({k: v for k, v in cmp.items()}, default=float)}", flush=True)
    jdump("anchor-reproduction.json", dict(seeds=list(E27.CAL_SEEDS), anchors=rows))


def stage_grid():
    fz = json.loads((OUT / "seed-freeze.json").read_text())
    assert tuple(fz["calibration"]) == CAL_SEEDS and not set(fz["calibration"]) & set(fz["confirmation"])
    path = OUT / "grid-results.json"
    done = json.loads(path.read_text())["results"] if path.exists() else []
    have = {r["key"] for r in done}
    for op in grid():
        if op.key() in have:
            continue
        r = evaluate_point(op, CAL_SEEDS)
        done.append(r)
        jdump("grid-results.json", dict(seeds=list(CAL_SEEDS), expected_keys=[p.key() for p in grid()], results=done))
        print(f"[grid] {op.key():<26} steps {r['refractory_steps']} {brief(r)}", flush=True)


def stage_select():
    g = json.loads((OUT / "grid-results.json").read_text())
    rows = g["results"]
    assert sorted(r["key"] for r in rows) == sorted(p.key() for p in grid())
    passing, ranked = select(rows)
    ranking = dict(rule="all-pass first; then fewest refractory steps; then lowest APL gain; then requested ms",
                   ranked=[summary_row(r) for r in ranked], passing=[r["key"] for r in passing])
    jdump("candidate-ranking.json", ranking)
    if passing:
        w = ranked[0]
        jdump("frozen-operating-point.json", dict(status="FROZEN: least-invasive all-pass calibration point",
                                                  operating_point=w["operating_point"], key=w["key"],
                                                  refractory_steps=w["refractory_steps"], criteria=w["criteria"],
                                                  provenance=w["provenance"]))
        (OUT / "best-nonpassing-candidate.json").unlink(missing_ok=True)
    else:
        jdump("frozen-operating-point.json", dict(status="NO PASSING OPERATING POINT", operating_point=None))
        best = sorted(rows, key=nonpassing_key)[0]
        jdump("best-nonpassing-candidate.json", dict(
            note="descriptive only; NOT frozen; must not be used as an operating point",
            rule="most criteria passed; then R3, R5, R4; then fewest refractory steps; then lowest APL gain",
            operating_point=best["operating_point"], key=best["key"], criteria=best["criteria"], summary=summary_row(best)))
    print("passing:", [r["key"] for r in passing])


def stage_confirm():
    fz = json.loads((OUT / "frozen-operating-point.json").read_text())
    if fz["operating_point"] is not None:
        op = E27.OperatingPoint(**fz["operating_point"]); descriptive = False
    else:
        b = json.loads((OUT / "best-nonpassing-candidate.json").read_text())
        op = E27.OperatingPoint(**b["operating_point"]); descriptive = True
    r = evaluate_point(op, CONF_SEEDS)
    stock = evaluate_point(E27.OperatingPoint(), CONF_SEEDS)
    jdump("confirmation-results.json", dict(descriptive_confirmation_only=descriptive, operating_point=asdict(op),
                                            result=r, stock_same_seeds=stock,
                                            robust=bool(not descriptive and r["criteria"]["all_pass"])))
    print("[confirm]", op.key(), brief(r)); print("[confirm stock]", brief(stock))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["anchors", "grid", "select", "confirm"])
    a = ap.parse_args()
    {"anchors": stage_anchors, "grid": stage_grid, "select": stage_select, "confirm": stage_confirm}[a.stage]()


if __name__ == "__main__":
    main()
