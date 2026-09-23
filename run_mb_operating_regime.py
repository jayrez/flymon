"""Experiment 27: restore a responsive mushroom-body operating regime (no learning).

Stages (results/experiment-27-mb-operating-regime/preregistration.md):
  stock        reproduce the E26 stock failure on the E26 seeds (2701-2706)
  sweep        evaluate an intervention family on calibration seeds (2801-2806):
                 --family refractory | apl | kcgain
  diagnostic   non-selectable mechanistic diagnostics (sensory_input=False, voltages)
  confirm      frozen operating point, once, on confirmation seeds (2811-2820)

Reuses the E26 harness, stimuli, bridge and maximum-effect protocol unchanged. The frozen
connectome, the 418-edge plastic set, the T4 injection and the visual stimuli are untouched;
no learning, no Pokemon objective.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import run_mb_plasticity as E26
from flymon import mb_plasticity as mp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiment-27-mb-operating-regime"
E26_SEEDS = tuple(range(2701, 2707))
CAL_SEEDS = tuple(range(2801, 2807))
CONF_SEEDS = tuple(range(2811, 2821))
REFRACTORY_GRID_MS = (0, 20, 40, 60, 80, 100)
APL_GAIN_GRID = (1.5, 2.0, 3.0, 5.0)
KC_GAIN_GRID = (0.75, 0.5, 0.35, 0.25)
TARGETS = ("KCg-d", "MBON01", "MBON11", "PAM01", "PPL101")
REPORT_POPS = ("all_KC", "KCg-d", "MBON01", "MBON11", "PAM01", "PPL101", "APL", "DPM", "controller_DN")
BASELINE_STEPS = 600
WARMUP = 200                      # steps discarded after reset before DAN baselines (post-transient)


@dataclass(frozen=True)
class OperatingPoint:
    refractory_ms: float = 0.0        # FlyBrain's own global refractory (0 = stock)
    apl_gain: float = 1.0             # scale of existing APL output (1 = stock)
    kc_gain: float = 1.0              # shared input gain onto all KCs (1 = stock)
    sensory_input: bool = True        # FlyBrain flag; False only in the non-selectable diagnostic

    def key(self):
        return f"ref{self.refractory_ms:g}ms|apl{self.apl_gain:g}|kcg{self.kc_gain:g}|sens{int(self.sensory_input)}"

    def magnitude(self):
        """Ordering for 'least invasive': family rank, then distance from stock."""
        fam = 0 if (self.apl_gain == 1.0 and self.kc_gain == 1.0) else (1 if self.kc_gain == 1.0 else 2)
        return (fam, abs(self.refractory_ms), abs(self.apl_gain - 1.0), abs(1.0 - self.kc_gain))


class OperatingPointBrain(mp.PlasticBrain):
    """E26 PlasticBrain + existing-edge scaling for APL output or KC input (identity at 1.0)."""

    def __init__(self, brain, edges, op: OperatingPoint, cell_type, W):
        super().__init__(brain, edges)
        xp = brain.xp
        self.op = op
        self.apl = np.flatnonzero(cell_type == "APL")
        self.kc_all = np.flatnonzero(np.char.startswith(cell_type, "KC"))
        if op.apl_gain != 1.0:
            cols = W[:, self.apl].tocsc()                        # existing APL outgoing edges only
            self._apl_cols = [xp.asarray(cols[:, [k]].toarray().ravel().astype(np.float32)) for k in range(len(self.apl))]
        if op.kc_gain != 1.0:
            self._kc_dev = xp.asarray(self.kc_all)

    def step(self, prepared, delta):
        b, xp = self.b, self.b.xp
        op = self.op
        if op.apl_gain == 1.0 and op.kc_gain == 1.0:
            return super().step(prepared, delta)
        current = b.synaptic_input(b.fired) * b.gain
        if delta is not None and np.any(delta):
            self._mask[:] = 0
            if b.fired.size:
                self._mask[b.fired] = 1.0
            contrib = xp.asarray(delta, xp.float32) * self._mask[self.pre]
            per_post = xp.bincount(self.inv, weights=contrib, minlength=len(self.post_u)).astype(xp.float32)
            current[self.post_u_dev, 0] += per_post * xp.float32(b.gain)
        if op.apl_gain != 1.0 and b.fired.size:
            fired_host = b.fired.get() if xp is not np else b.fired
            for k, a in enumerate(self.apl):
                if np.any(fired_host == a):
                    current[:, 0] += xp.float32((op.apl_gain - 1.0) * b.gain) * self._apl_cols[k]
        if op.kc_gain != 1.0:
            current[self._kc_dev, 0] *= xp.float32(op.kc_gain)
        b.v *= b.decay
        b.v += current + b.tonic
        b.v += (b.rng.random((b.n, 1)) < b.noise_hz * b.dt) * np.float32(b.noise_amp)
        if prepared is not None:
            idx, vals = prepared
            b.v[idx, 0] += vals
        if b.refractory_steps:
            b.v[(b.steps - b.last_spike) <= b.refractory_steps] = 0.0
        fired = xp.flatnonzero(b.v >= 1.0)
        b.v.ravel()[fired] = 0.0
        if b.refractory_steps:
            b.last_spike.ravel()[fired] = b.steps
        b.fired = fired
        b.steps += 1
        return fired if xp is np else fired.get()


def make_harness(op: OperatingPoint):
    from scipy import sparse
    h = E26.Harness(refractory=op.refractory_ms / 1000.0, sensory_input=op.sensory_input)
    W = sparse.load_npz(ROOT / "redfly-benchmark" / "data" / "weights.npz").tocsr()
    h.pb = OperatingPointBrain(h.brain, h.edges, op, h.ct, W)
    h.op = op
    h.ceiling = 1.0 / (h.brain.refractory_steps + 1)
    h.all_pops = {"all_KC": np.flatnonzero(np.char.startswith(h.ct, "KC")), "KCg-d": h.pop["KC"],
                  "MBON01": h.pop["MBON01"], "MBON11": h.pop["MBON11"], "PAM01": h.pop["PAM01"],
                  "PPL101": h.pop["PPL101"], "APL": mp.population(h.ct, "APL"), "DPM": mp.population(h.ct, "DPM"),
                  "controller_DN": h.pop["DN"]}
    return h


# ---------------------------------------------------------------- assays
def baseline_assay(h, seeds):
    per = {p: [] for p in REPORT_POPS}; onset = {p: [] for p in REPORT_POPS}; vmean = {p: [] for p in REPORT_POPS}
    whole_hi, whole_silent, finite = [], [], True
    for s in seeds:
        h.brain.reset(seed=s)
        counts = np.zeros(h.brain.n, np.int32); frac_t = {p: [] for p in REPORT_POPS}
        for t in range(BASELINE_STEPS):
            f = h.pb.step(None, None)
            counts[f] += 1
            m = np.zeros(h.brain.n, bool); m[f] = True
            for p in REPORT_POPS:
                frac_t[p].append(m[h.all_pops[p]].mean())
        v = h.brain.v.get()[:, 0] if hasattr(h.brain.v, "get") else h.brain.v[:, 0]
        finite &= bool(np.isfinite(v).all())
        duty = counts / BASELINE_STEPS
        for p in REPORT_POPS:
            ix = h.all_pops[p]
            per[p].append(float(duty[ix].mean()))
            ft = np.array(frac_t[p]); thr = 0.9 * h.ceiling
            onset[p].append(int(np.argmax(ft >= thr)) if (ft >= thr).any() else None)
            vmean[p].append(float(v[ix].mean()))
        whole_hi.append(float(np.mean(duty >= 0.9 * h.ceiling))); whole_silent.append(float(np.mean(duty == 0)))
    dt = float(h.brain.dt)
    pops = {p: dict(duty=float(np.mean(per[p])), duty_sd=float(np.std(per[p])), rate_hz=float(np.mean(per[p]) / dt),
                    ceiling_normalised_duty=float(np.mean(per[p]) / h.ceiling), step_to_90pct_of_ceiling=onset[p],
                    mean_voltage_end=float(np.mean(vmean[p])), per_seed_duty=per[p]) for p in REPORT_POPS}
    return dict(populations=pops, ceiling_duty=h.ceiling, whole_brain_fraction_at_ceiling=float(np.mean(whole_hi)),
                whole_brain_fraction_silent=float(np.mean(whole_silent)), finite=finite)


STIM_LABELS = {"pokemon_house1f": "pokemon_bedroom_after_walk"}    # E26 label corrected (map stayed 38)


def visual_assay(h, seeds):
    stim, meta = E26.make_stimuli()
    stim = {STIM_LABELS.get(k, k): v for k, v in stim.items()}
    res = {}
    for name in ["no_visual"] + list(stim):
        fn = stim.get(name, stim["gray"])
        res[name] = [h.visual_trial(s, fn, inject=name != "no_visual") for s in seeds]
    tot = {k: np.array([float(r["kc_per_cell"].sum()) for r in v]) for k, v in res.items()}
    cells = {k: np.stack([r["kc_per_cell"] for r in v]) for k, v in res.items()}
    rows = {}
    for c in stim:
        if c == "gray":
            continue
        row = {}
        for ref in ("no_visual", "gray"):
            d = tot[c] - tot[ref]
            sd = d.std(ddof=1)
            row[ref] = dict(mean_diff=float(d.mean()),
                            rel_diff=float(d.mean() / tot[ref].mean()) if tot[ref].mean() > 0 else None,
                            consistent_seeds=int(max((d > 0).sum(), (d < 0).sum())),
                            dz=float(d.mean() / sd) if sd > 0 else (float(np.sign(d.mean())) * float("inf") if d.mean() else 0.0))
        dc = cells[c] - cells["no_visual"]; base = cells["no_visual"].mean(axis=0)
        sig = np.sign(dc); cons = np.maximum((sig > 0).sum(0), (sig < 0).sum(0)) >= 5
        big = np.abs(dc.mean(0)) >= 0.2 * np.maximum(base, 1.0)
        row["fraction_cells_modulated"] = float(np.mean(cons & big))
        row["passes_R3"] = bool(all(row[r]["rel_diff"] is not None and abs(row[r]["rel_diff"]) >= 0.05
                                    and row[r]["consistent_seeds"] >= 5 and np.sign(row[r]["mean_diff"]) ==
                                    np.sign(row["no_visual"]["mean_diff"]) for r in ("no_visual", "gray")))
        rows[c] = row
    summary = {k: dict(kcgd_rate_hz=float(tot[k].mean() / len(h.pop["KC"]) / 12.0), total_mean=float(tot[k].mean()),
                       total_sd=float(tot[k].std(ddof=1)),
                       cv=float(tot[k].std(ddof=1) / tot[k].mean()) if tot[k].mean() > 0 else None,
                       fraction_active=float(np.mean(cells[k] > 0)), per_seed=tot[k].tolist()) for k in res}
    best = max(rows, key=lambda c: abs(rows[c]["no_visual"]["rel_diff"] or 0))
    return dict(summary=summary, contrasts=rows, best_condition=best, stimuli_meta=meta,
                R3=bool(any(r["passes_R3"] for r in rows.values())))


def dan_assay(h, seeds):
    out = {}
    for ch, g in (("appetitive", "PAM01"), ("aversive", "PPL101")):
        for a in E26.AMP_GRID:
            base, stim = [], []
            for s in seeds:
                # warm-up past the post-reset transient, then a steady-state baseline window, then stimulation
                r = h.run_schedule(s, [dict(n=WARMUP, tag="warmup"), dict(n=100, tag="baseline"),
                                       dict(n=E26.DAN_WINDOW, dan=(ch, a, 0, E26.DAN_WINDOW), tag="dan")],
                                   mp.PlasticState.zeros(h.edges))
                n = len(h.dan[ch])
                base.append(r[1]["counts"][g] / n / 100); stim.append(r[2]["counts"][g] / n / E26.DAN_WINDOW)
            base, stim = np.array(base), np.array(stim)
            d = stim - base
            out[f"{ch}|{a}"] = dict(channel=ch, amplitude=a, baseline_duty=float(base.mean()), stimulated_duty=float(stim.mean()),
                                    delta_pp=float(100 * d.mean()), rel_delta=float(d.mean() / base.mean()) if base.mean() > 0 else None,
                                    positive_seeds=int((d > 0).sum()))
    # preregistered: E26 bridge amplitude rule re-applied at this operating point (smallest amplitude with
    # >= 50% of DANs spiking at least once in the window, both channels); R4 evaluated there
    once = {}
    for a in E26.AMP_GRID:
        ok = True
        for ch in ("appetitive", "aversive"):
            fr = []
            for s in seeds:
                h.brain.reset(seed=s); tally = np.zeros(h.brain.n, bool)
                for _ in range(WARMUP + 100):
                    h.pb.step(None, None)
                for _ in range(E26.DAN_WINDOW):
                    f = h.pb.step(h.pb.prepare([(h.dan[ch], a)]), None); tally[f] = True
                fr.append(tally[h.dan[ch]].mean())
            once[f"{ch}|{a}"] = float(np.mean(fr)); ok &= np.mean(fr) >= 0.5
        if ok and "rule_amplitude" not in once:
            once["rule_amplitude"] = a
    amp = once.get("rule_amplitude", max(E26.AMP_GRID))
    r4 = {ch: out[f"{ch}|{amp}"] for ch in ("appetitive", "aversive")}
    passes = all(v["delta_pp"] >= 10.0 and v["positive_seeds"] >= 5 for v in r4.values())
    return dict(grid=out, spiking_once=once, rule_amplitude=amp, at_rule_amplitude=r4, R4=bool(passes))


def leverage_assay(h, seeds):
    conds = {"baseline": np.zeros(h.edges.n), "zeroed": -h.edges.w0.copy(), "double_diagnostic": h.edges.w0.copy()}
    counts = {}
    for name, delta in conds.items():
        st = mp.PlasticState(h.edges.sha256, delta)
        counts[name] = [h.run_schedule(s, [dict(n=300, cs="all", tag="kc_drive")], st)[0]["counts"] for s in seeds]
    res = {}
    for g in ("MBON01", "MBON11", "KC"):
        b = np.array([c[g] for c in counts["baseline"]]); z = np.array([c[g] for c in counts["zeroed"]])
        dd = np.array([c[g] for c in counts["double_diagnostic"]])
        res[g] = dict(baseline=b.tolist(), zeroed=z.tolist(), double=dd.tolist(),
                      zeroed_rel=float((z.mean() - b.mean()) / b.mean()) if b.mean() else None,
                      double_rel=float((dd.mean() - b.mean()) / b.mean()) if b.mean() else None)
    tb = np.array([c["MBON01"] + c["MBON11"] for c in counts["baseline"]])
    tz = np.array([c["MBON01"] + c["MBON11"] for c in counts["zeroed"]])
    rel = float((tz.mean() - tb.mean()) / tb.mean()) if tb.mean() else None
    return dict(per_population=res, target_zeroed_rel=rel, target_lower_seeds=int((tz < tb).sum()),
                R5=bool(rel is not None and rel <= -0.10 and (tz < tb).sum() >= 5))


def evaluate(op: OperatingPoint, seeds, include_visual=True):
    t0 = time.perf_counter()
    h = make_harness(op)
    base = baseline_assay(h, seeds)
    P = base["populations"]
    r1 = all(P[p]["ceiling_normalised_duty"] < 0.80 for p in TARGETS)
    rates = [P[p]["rate_hz"] for p in TARGETS]
    r2 = bool(np.median(rates) > 1.0 and min(rates) > 0.0)
    r6 = bool(base["finite"] and base["whole_brain_fraction_at_ceiling"] < 0.01 and base["whole_brain_fraction_silent"] < 0.5)
    vis = visual_assay(h, seeds) if include_visual else None
    dan = dan_assay(h, seeds)
    lev = leverage_assay(h, seeds)
    crit = dict(R1=bool(r1), R2=r2, R3=vis["R3"] if vis else None, R4=dan["R4"], R5=lev["R5"], R6=r6)
    out = dict(operating_point=asdict(op), key=op.key(), seeds=list(seeds), baseline=base, visual=vis, dan=dan,
               leverage=lev, criteria=crit, all_pass=bool(all(v for v in crit.values())),
               edge_hash=h.edges.sha256, frozen_w_sha256=h.w_hash, t4_sha256=h.t4.sha256(),
               seconds=time.perf_counter() - t0)
    del h
    return out


def brief(r):
    P = r["baseline"]["populations"]
    v = r["visual"]
    return (f"KCg-d {P['KCg-d']['rate_hz']:.1f}Hz (norm {P['KCg-d']['ceiling_normalised_duty']:.2f}) "
            f"MBON01 {P['MBON01']['rate_hz']:.1f} PAM01 {P['PAM01']['rate_hz']:.1f} PPL101 {P['PPL101']['rate_hz']:.1f} | "
            f"vis best {v['best_condition'] if v else '-'} "
            f"{(v['contrasts'][v['best_condition']]['no_visual']['rel_diff'] or 0) * 100 if v else 0:+.2f}% | "
            f"DAN dpp app {r['dan']['at_rule_amplitude']['appetitive']['delta_pp']:+.1f} "
            f"av {r['dan']['at_rule_amplitude']['aversive']['delta_pp']:+.1f} | "
            f"zero-edge {r['leverage']['target_zeroed_rel'] * 100 if r['leverage']['target_zeroed_rel'] is not None else float('nan'):+.2f}% | "
            f"{r['criteria']}")


def jdump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float) + "\n")


# ---------------------------------------------------------------- stages
def stage_stock():
    t0 = time.perf_counter()
    r = evaluate(OperatingPoint(), E26_SEEDS)
    e26v = json.loads((ROOT / "results/experiment-26/visual-drive-results.json").read_text())
    e26m = json.loads((ROOT / "results/experiment-26/max-effect-results.json").read_text())
    e26s = json.loads((ROOT / "results/experiment-26/mb-saturation-diagnostic.json").read_text())
    comp = dict(
        kcgd_visual_totals_identical=all(r["visual"]["summary"][STIM_LABELS.get(k, k)]["per_seed"] == v["per_seed"]
                                         for k, v in e26v["summary"].items()),
        max_effect_target_rel=(r["leverage"]["target_zeroed_rel"], e26m["target_rel_change_zeroed"]),
        saturation_duty={p: (r["baseline"]["populations"][p]["duty"], e26s["populations"][p]["spike_fraction_of_steps"])
                         for p in ("KCg-d", "MBON01", "MBON11", "PAM01", "PPL101", "APL", "DPM", "all_KC", "controller_DN")})
    jdump("stock-reproduction.json", dict(result=r, comparison_to_e26=comp, seconds=time.perf_counter() - t0))
    print(brief(r)); print(json.dumps(comp, default=float)[:1500])


def stage_sweep(family):
    grid = {"refractory": [OperatingPoint(refractory_ms=m) for m in REFRACTORY_GRID_MS]}
    if family == "refractory":
        ops = grid["refractory"]
    else:
        base_ref = json.loads((OUT / "refractory-sweep.json").read_text())["apl_base_refractory_ms"]
        if family == "apl":
            ops = [OperatingPoint(refractory_ms=base_ref, apl_gain=g) for g in APL_GAIN_GRID]
        else:
            prev = json.loads((OUT / "apl-sweep.json").read_text()) if (OUT / "apl-sweep.json").exists() else {}
            ops = [OperatingPoint(refractory_ms=base_ref, kc_gain=g) for g in KC_GAIN_GRID]
    rows = []
    for op in ops:
        r = evaluate(op, CAL_SEEDS)
        rows.append(r)
        print(f"[{family}] {op.key():<28} {brief(r)}", flush=True)
    doc = dict(family=family, seeds=list(CAL_SEEDS), results=rows)
    if family == "refractory":
        # preregistered: base for a later APL/gain stage = smallest refractory with R1 & R6 (gross saturation fixed)
        fixed = [r for r in rows if r["criteria"]["R1"] and r["criteria"]["R6"]]
        doc["apl_base_refractory_ms"] = (min(r["operating_point"]["refractory_ms"] for r in fixed) if fixed else 0.0)
    jdump(f"{family}-sweep.json", doc)


def stage_diagnostic():
    """Non-selectable: identifies the saturation source (FlyBrain's documented sensory_input flag)."""
    r = evaluate(OperatingPoint(sensory_input=False), CAL_SEEDS, include_visual=True)
    print("[diagnostic sensory_input=False]", brief(r))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stock", "sweep", "diagnostic", "confirm"])
    ap.add_argument("--family", default="refractory", choices=["refractory", "apl", "kcgain"])
    a = ap.parse_args()
    if a.stage == "stock":
        stage_stock()
    elif a.stage == "sweep":
        stage_sweep(a.family)
    elif a.stage == "diagnostic":
        jdump("diagnostic-sensory-input-off.json", stage_diagnostic())
    elif a.stage == "confirm":
        fz = json.loads((OUT / "frozen-operating-point.json").read_text())
        op = OperatingPoint(**fz["operating_point"])
        r = evaluate(op, CONF_SEEDS)
        stock = evaluate(OperatingPoint(), CONF_SEEDS)
        jdump("confirmation-results.json", dict(frozen=fz, result=r, stock_same_seeds=stock))
        print("[confirm]", brief(r)); print("[confirm stock]", brief(stock))


if __name__ == "__main__":
    main()
