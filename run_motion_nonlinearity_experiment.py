"""Experiment 19: which minimal mechanism reads out the Experiment-18 T4/T5 geometry.

Stages:
  calibrate - model families N0..N4 (+ controls) on calibration stimuli only
  heldout   - frozen selected mechanism + controls, ablations, sensitivity, once

Reuses the Experiment-18 geometry and stimulus machinery unchanged. No gameplay,
RAM, action labels, reward or RL.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from flymon import column_motion as cm
from flymon import motion_nonlinearity as mn
from flymon.optic_columns import (SUBTYPES, audit_receptive_fields, load_column_coords,
                                  shuffle_columns)
from flymon.retina import load_mapping

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "redfly-benchmark" / "data"
ANNOT = DATA / "raw" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
EXP5 = ROOT / "results" / "experiment-05-retinotopic"
RESULTS = ROOT / "results" / "experiment-19-motion-nonlinearity"
FRAMES = 60
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}
CAL_SPEEDS = (0.5, 1.0, 2.0)
HELDOUT_SPEEDS = (0.25, 0.75, 1.5, 3.0, 4.0)
HELDOUT_WIDTHS = (12, 26)
HELDOUT_CONTRASTS = (0.25, 0.5, 0.75)
HELDOUT_OFFSETS = (28.0, -28.0)
HELDOUT_GRATING = ((24.0, np.pi / 4), (48.0, 3 * np.pi / 4))
GEOMETRY_PERMUTATIONS = 200
MOTION_WINDOW = (mn.STARTUP_FRAMES, FRAMES)
NBOOT = 10000

TAU_PAIRS_NONLINEAR = ((1.0, 3.0), (2.0, 5.0), (2.0, 8.0), (4.0, 12.0))


def build_grid():
    grid = [mn.Mechanism(family="N0_E18", tau_fast=2.0, tau_slow=8.0, label="E18 regression")]
    for tf in (1.0, 2.0, 4.0):
        for ts in (3.0, 5.0, 8.0, 12.0, 16.0, 24.0):
            if ts > tf:
                grid.append(mn.Mechanism(family="N1_LEXT", tau_fast=tf, tau_slow=ts))
    for tf, ts in TAU_PAIRS_NONLINEAR:
        for gate in ("identity", "saturating"):
            grid.append(mn.Mechanism(family="N2_COINCIDENCE", tau_fast=tf, tau_slow=ts, gate=gate))
        for a in (0.5, 1.0, 2.0):
            grid.append(mn.Mechanism(family="N3_SHUNT", tau_fast=tf, tau_slow=ts, alpha=a))
        for b in (0.5, 1.0):
            for g in (0.5, 1.0):
                grid.append(mn.Mechanism(family="N4_SUBUNIT", tau_fast=tf, tau_slow=ts,
                                         beta=b, gamma=g))
        grid.append(mn.Mechanism(family="N5_HR_CONTROL", tau_fast=tf, tau_slow=ts))
        grid.append(mn.Mechanism(family="P_ONLY", tau_fast=tf, tau_slow=ts))
        grid.append(mn.Mechanism(family="OPPONENT_PRODUCT", tau_fast=tf, tau_slow=ts))
    return grid


# ---------------------------------------------------------------- setup
def load_context():
    records, _ = load_mapping(EXP5 / "retina-mapping.json")
    meta = np.load(DATA / "brain.npz")
    hexes, cart = load_column_coords(DATA / "brain.npz", ANNOT)
    W = sparse.load_npz(DATA / "weights.npz").tocsr()
    audit = audit_receptive_fields(W, meta["cell_type"], meta["side"], cart, meta["ids"])
    colindex = cm.column_index(records)
    uv = np.asarray([(r.u, r.v) for r in records], np.float32)
    proj = cm.build_projections(W, meta["cell_type"], hexes, meta["side"], audit, colindex)
    col_uv = column_uv(records, colindex)
    rf_uv = {s: receptive_centres(proj[s], col_uv) for s in proj}
    resolved = {s: np.array([r.fast_weight + r.slow_weight for r in audit[s]["records"]])
                for s in audit if audit[s]["records"]}
    return dict(records=records, meta=meta, hexes=hexes, cart=cart, W=W, audit=audit,
                colindex=colindex, uv=uv, proj=proj, col_uv=col_uv, rf_uv=rf_uv,
                resolved=resolved)


def column_uv(records, colindex):
    n = len(colindex)
    s = np.zeros((n, 2)); c = np.zeros(n)
    for r in records:
        i = colindex[(r.eye_side, int(r.h1), int(r.h2))]
        s[i] += (r.u, r.v); c[i] += 1
    c[c == 0] = 1
    return s / c[:, None]


def receptive_centres(proj_sub, col_uv):
    """Weighted screen-coordinate centre of each neuron's resolved input field."""
    total = None; norm = None
    for m in proj_sub.values():
        a = m.copy(); a.data = np.abs(a.data)
        t = a @ col_uv
        w = np.asarray(a.sum(axis=1)).ravel()
        total = t if total is None else total + t
        norm = w if norm is None else norm + w
    norm = np.where(norm > 0, norm, 1.0)
    return total / norm[:, None]


def stimulus_bank(kind):
    bank = {}
    def add(name, seq, **meta):
        bank[name] = (seq, meta)
    speeds = CAL_SPEEDS if kind == "calibration" else HELDOUT_SPEEDS
    for pol in ("ON", "OFF"):
        for d in ("left", "right", "up", "down"):
            for sp in speeds:
                add(f"bar_{pol}_{d}_sp{sp}", cm.parametric_bar(d, pol, FRAMES, sp),
                    polarity=pol, direction=d, speed=sp, width=18, contrast=1.0,
                    start_offset=0.0, axis="speed")
            if kind == "heldout":
                for w in HELDOUT_WIDTHS:
                    add(f"bar_{pol}_{d}_w{w}", cm.parametric_bar(d, pol, FRAMES, 1.0, bar_px=w),
                        polarity=pol, direction=d, speed=1.0, width=w, contrast=1.0,
                        start_offset=0.0, axis="width")
                for c in HELDOUT_CONTRASTS:
                    add(f"bar_{pol}_{d}_c{c}", cm.parametric_bar(d, pol, FRAMES, 1.0, contrast=c),
                        polarity=pol, direction=d, speed=1.0, width=18, contrast=c,
                        start_offset=0.0, axis="contrast")
                for o in HELDOUT_OFFSETS:
                    add(f"bar_{pol}_{d}_o{o}",
                        cm.parametric_bar(d, pol, FRAMES, 1.0, start_offset=o),
                        polarity=pol, direction=d, speed=1.0, width=18, contrast=1.0,
                        start_offset=o, axis="position")
    if kind == "heldout":
        for d in ("left", "right"):
            for per, ph in HELDOUT_GRATING:
                add(f"grating_{d}_p{int(per)}_ph{ph:.2f}",
                    cm.parametric_grating(d, FRAMES, 1.0, per, ph),
                    polarity="GRATING", direction=d, speed=1.0, axis="grating")
    canonical = cm.parametric_bar("left", "OFF", FRAMES, 1.0)
    add("ctrl_frozen", cm.temporal_variant(canonical, "frozen_first"), control="frozen")
    add("ctrl_shuffle", cm.temporal_variant(canonical, "shuffle"), control="shuffle")
    add("ctrl_reverse", cm.temporal_variant(canonical, "reverse"), control="reverse")
    add("ctrl_gray", np.full((FRAMES, cm.H, cm.W), 0.5, np.float32), control="gray")
    return bank


def luminance_cache(bank, records, colindex, uv):
    out = {}
    for name, (seq, _m) in bank.items():
        out[name] = cm.column_luminance(cm.to_rgba(seq), records, colindex, uv)
    return out


# ---------------------------------------------------------------- evaluation
class Evaluator:
    def __init__(self, ctx, bank, lum):
        self.ctx = ctx; self.bank = bank; self.lum = lum
        self._sig = {}
        self._arms = {}

    def signals(self, stim, tf, ts):
        key = (stim, tf, ts)
        if key not in self._sig:
            self._sig[key] = mn.signals_for(self.lum[stim],
                                            mn.Mechanism(tau_fast=tf, tau_slow=ts))
        return self._sig[key]

    def arms(self, stim, sub, tf, ts, magnitude, proj=None):
        key = (stim, sub, tf, ts, magnitude, id(proj))
        if key not in self._arms:
            p = (proj or self.ctx["proj"])[sub]
            self._arms[key] = mn.arm_drives(p, self.signals(stim, tf, ts), sub[:2],
                                            magnitude=magnitude)
        return self._arms[key]

    def responses(self, mech, sub, stim, proj=None):
        magnitude = mech.family not in mn.SIGNED_FAMILIES
        F, S = self.arms(stim, sub, mech.tau_fast, mech.tau_slow, magnitude, proj)
        return mn.apply_mechanism(F, S, mech)

    def scalars(self, mech, sub, stim, proj=None):
        """Metrics A (whole mean), C (motion window), D (anatomy event window)."""
        Y = self.responses(mech, sub, stim, proj)
        meta = self.bank[stim][1]
        a = mn.metric_traces(Y)
        c = mn.metric_traces(Y, MOTION_WINDOW)
        d = a
        if meta.get("direction") and meta.get("speed"):
            uvc = self.ctx["rf_uv"][sub]
            cross = mn.predicted_crossing(uvc[:, 0], uvc[:, 1], meta["direction"], FRAMES,
                                          meta["speed"], meta.get("width", 18),
                                          meta.get("start_offset", 0.0))
            d = mn.event_window_means(Y, cross)
        return a, c, d, Y


def direction_metrics(ev, mech, metric_idx, proj=None, audit=None):
    """Per-subtype preferred-vs-null using the ANATOMY-predicted direction per neuron."""
    audit = audit or ev.ctx["audit"]
    out = {}
    for sub in SUBTYPES:
        recs = audit[sub]["records"]
        if not recs:
            continue
        pol = "ON" if sub.startswith("T4") else "OFF"
        pref_dirs = np.array([r.predicted_direction for r in recs])
        pref = np.zeros(len(recs)); null = np.zeros(len(recs)); ok = np.zeros(len(recs), bool)
        for d in ("left", "right", "up", "down"):
            names = [n for n, (_s, m) in ev.bank.items()
                     if m.get("direction") == d and m.get("polarity") == pol]
            names_o = [n for n, (_s, m) in ev.bank.items()
                       if m.get("direction") == OPPOSITE[d] and m.get("polarity") == pol]
            sel = pref_dirs == d
            if not names or not names_o or not sel.any():
                continue
            p = np.mean([ev.scalars(mech, sub, n, proj)[metric_idx] for n in names], axis=0)
            q = np.mean([ev.scalars(mech, sub, n, proj)[metric_idx] for n in names_o], axis=0)
            pref[sel] = p[sel]; null[sel] = q[sel]; ok[sel] = True
        if not ok.any():
            continue
        p, q = pref[ok], null[ok]
        d_ = mn.dsi(p, q)
        resp = (p + q) > 0
        out[sub] = dict(n=int(ok.sum()), pref_rate=float(p.mean()), null_rate=float(q.mean()),
                        effect=float((p - q).mean()), dsi_mean=float(d_.mean()),
                        dsi_median=float(np.median(d_)),
                        responsive_fraction=float(resp.mean()),
                        sign_consistency_responsive=(float(np.mean((p - q)[resp] > 0))
                                                     if resp.any() else 0.0),
                        per_neuron_dsi=d_, per_neuron_effect=(p - q))
    return out


def summarise(m):
    if not m:
        return dict(mean_abs_dsi=0.0, mean_dsi=0.0, mean_sign=0.0, n_coherent=0, min_dsi=0.0)
    dsis = [v["dsi_mean"] for v in m.values()]
    signs = [v["sign_consistency_responsive"] for v in m.values()]
    return dict(mean_abs_dsi=float(np.mean(np.abs(dsis))), mean_dsi=float(np.mean(dsis)),
                mean_sign=float(np.mean(signs)),
                n_coherent=int(sum(1 for d, s in zip(dsis, signs) if d > 0 and s >= 0.70)),
                min_dsi=float(np.min(dsis)), max_abs_dsi=float(np.max(np.abs(dsis))),
                all_same_sign=bool(all(d > 0 for d in dsis) or all(d < 0 for d in dsis)))


def strip(m):
    return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("per_neuron")}
            for k, v in m.items()}


# ---------------------------------------------------------------- stages
def stage_calibrate():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ctx = load_context()
    bank = stimulus_bank("calibration")
    lum = luminance_cache(bank, ctx["records"], ctx["colindex"], ctx["uv"])
    ev = Evaluator(ctx, bank, lum)
    grid = build_grid()
    table = {}
    for mech in grid:
        m = direction_metrics(ev, mech, 0)
        s = summarise(m)
        table[mech.key()] = dict(mechanism=mech.to_json(), metric="A_whole_mean", **s,
                                 per_subtype={k: dict(dsi_mean=v["dsi_mean"],
                                                      sign=v["sign_consistency_responsive"],
                                                      effect=v["effect"])
                                              for k, v in m.items()})
        print(f"[cal] {mech.key():<34} meanDSI {s['mean_dsi']:+.4f} |DSI| {s['mean_abs_dsi']:.4f} "
              f"sign {s['mean_sign']:.2f} coherent {s['n_coherent']}/8", flush=True)
    (RESULTS / "calibration-results.json").write_text(json.dumps(table, indent=2) + "\n")
    (RESULTS / "model-configs.json").write_text(json.dumps([m.to_json() for m in grid], indent=2) + "\n")

    THRESH_DSI, THRESH_SIGN, MIN_COHERENT = 0.15, 0.70, 6
    eligible = [(mn.SIMPLICITY[v["mechanism"]["family"]], k) for k, v in table.items()
                if v["mechanism"]["family"] in mn.SELECTABLE
                and v["mean_abs_dsi"] >= THRESH_DSI and v["n_coherent"] >= MIN_COHERENT]
    if eligible:
        rank = min(e[0] for e in eligible)
        cands = [k for r, k in eligible if r == rank]
        selected = max(cands, key=lambda k: table[k]["mean_abs_dsi"])
        met = True
    else:
        bio = {k: v for k, v in table.items() if v["mechanism"]["family"] in mn.SELECTABLE}
        selected = max(bio, key=lambda k: bio[k]["mean_abs_dsi"])
        met = False
    (RESULTS / "selected-model.json").write_text(json.dumps(dict(
        selected=selected, mechanism=table[selected]["mechanism"],
        met_preregistered_threshold=met,
        thresholds=dict(mean_abs_dsi=THRESH_DSI, sign=THRESH_SIGN, min_coherent=MIN_COHERENT),
        calibration=dict(mean_abs_dsi=table[selected]["mean_abs_dsi"],
                         n_coherent=table[selected]["n_coherent"]),
        selection_rule="simplest family (N0<N1<N2<N3<N4) meeting mean|DSI|>=0.15 and >=6/8 "
                       "coherent subtypes on calibration only; controls never selectable",
        seconds=time.time() - t0), indent=2) + "\n")
    print(f"SELECTED {selected} (met threshold: {met})  {time.time()-t0:.1f}s", flush=True)


def stage_heldout():
    t0 = time.time()
    ctx = load_context()
    sel = json.loads((RESULTS / "selected-model.json").read_text())
    mech = mn.Mechanism(**sel["mechanism"])
    bank = stimulus_bank("heldout")
    lum = luminance_cache(bank, ctx["records"], ctx["colindex"], ctx["uv"])
    ev = Evaluator(ctx, bank, lum)

    out = {"selected": sel["selected"], "mechanism": sel["mechanism"]}
    metrics = {}
    for idx, name in ((0, "A_whole_mean"), (1, "C_motion_window"), (2, "D_event_window")):
        m = direction_metrics(ev, mech, idx)
        metrics[name] = m
        out[name] = dict(summary=summarise(m), subtypes=strip(m))
        print(f"[heldout {name}] meanDSI {summarise(m)['mean_dsi']:+.4f} "
              f"coherent {summarise(m)['n_coherent']}/8", flush=True)

    # control mechanisms on the same held-out data (reported, never selected)
    controls = {}
    for fam in ("N5_HR_CONTROL", "P_ONLY", "OPPONENT_PRODUCT",
                "P_ONLY_SIGNED", "OPPONENT_SIGNED", "N1_LEXT"):
        cm_mech = mn.Mechanism(family=fam, tau_fast=mech.tau_fast, tau_slow=mech.tau_slow)
        cmet = direction_metrics(ev, cm_mech, 0)
        controls[fam] = dict(summary=summarise(cmet), subtypes=strip(cmet))
        print(f"[control {fam:<18}] meanDSI {summarise(cmet)['mean_dsi']:+.4f} "
              f"coherent {summarise(cmet)['n_coherent']}/8", flush=True)
    out["control_mechanisms"] = controls

    # ablations
    abl = {}
    flat = mn.temporal_flat(mech)
    abl["temporal_flat"] = dict(summary=summarise(direction_metrics(ev, flat, 0)))
    inter = mn.interaction_ablation(mech)
    abl["interaction_ablation"] = dict(mechanism=inter.to_json(),
                                       summary=summarise(direction_metrics(ev, inter, 0)))
    # Geometry ablation: permute which optic column each presynaptic partner of a given
    # cell type occupies, preserving the marginal column distribution and every synaptic
    # weight, while keeping the ORIGINAL anatomy-predicted direction. This asks directly
    # whether the true column geometry is what produces the measured directionality.
    # Implemented as a column permutation of the projection matrices, which is exactly
    # equivalent to relabelling partner columns within a cell type, and is cheap.
    rng = np.random.default_rng(19062026)
    ncol = len(ctx["colindex"])
    nulls = []
    for i in range(GEOMETRY_PERMUTATIONS):
        perm = {t: rng.permutation(ncol) for t in cm.ALL_PARTNERS}
        proj_s = {sub: {t: m[:, perm[t]] for t, m in mats.items()}
                  for sub, mats in ctx["proj"].items()}
        ms = direction_metrics(ev, mech, 0, proj=proj_s)
        nulls.append(summarise(ms)["mean_dsi"])
        ev._arms.clear()
        if (i + 1) % 25 == 0:
            print(f"   geometry permutation {i+1}/{GEOMETRY_PERMUTATIONS}", flush=True)
    nulls = np.array(nulls)
    obs = out["A_whole_mean"]["summary"]["mean_dsi"]
    abl["geometry_shuffle"] = dict(permutations=int(len(nulls)), null_mean=float(nulls.mean()),
                                   null_sd=float(nulls.std()),
                                   null_95=[float(np.quantile(nulls, .025)), float(np.quantile(nulls, .975))],
                                   observed=float(obs),
                                   p_empirical=float((1 + np.sum(np.abs(nulls) >= abs(obs))) / (len(nulls) + 1)))
    out["ablations"] = abl

    # controls: static / shuffle / reverse rates
    ctrl_rates = {}
    for c in ("frozen", "shuffle", "reverse", "gray"):
        ctrl_rates[c] = {s: float(ev.scalars(mech, s, f"ctrl_{c}")[0].mean()) for s in SUBTYPES
                         if s in ctx["proj"]}
    dyn = {s: float(np.mean([ev.scalars(mech, s, n)[0].mean() for n, (_q, m) in bank.items()
                             if m.get("axis") == "speed"])) for s in SUBTYPES if s in ctx["proj"]}
    out["controls"] = dict(rates=ctrl_rates, dynamic_reference=dyn)

    # ON/OFF specificity
    onoff = {}
    for s in SUBTYPES:
        if s not in ctx["proj"]:
            continue
        on = [ev.scalars(mech, s, n)[0].mean() for n, (_q, m) in bank.items() if m.get("polarity") == "ON"]
        off = [ev.scalars(mech, s, n)[0].mean() for n, (_q, m) in bank.items() if m.get("polarity") == "OFF"]
        if on and off:
            onoff[s] = dict(on=float(np.mean(on)), off=float(np.mean(off)))
    out["on_off"] = onoff

    # generalisation by axis
    by_axis = {}
    for axis in ("speed", "width", "contrast", "position"):
        sub_bank = {n: v for n, v in bank.items() if v[1].get("axis") == axis}
        if not sub_bank:
            continue
        ev2 = Evaluator(ctx, sub_bank, {k: lum[k] for k in sub_bank})
        by_axis[axis] = dict(summary=summarise(direction_metrics(ev2, mech, 0)))
    out["by_axis"] = by_axis

    # resolved-weight sensitivity (descriptive)
    sens = {}
    for thr in (0.25, 0.40, 0.60):
        rows = {}
        for sub in SUBTYPES:
            recs = ctx["audit"][sub]["records"]
            if not recs or sub not in metrics["A_whole_mean"]:
                continue
            frac = np.array([(r.fast_weight + r.slow_weight) for r in recs])
            frac = frac / max(frac.max(), 1e-9)
            keep = frac >= thr
            d = metrics["A_whole_mean"][sub]["per_neuron_dsi"]
            k = keep[:len(d)]
            if k.sum() > 20:
                rows[sub] = dict(n=int(k.sum()), dsi_mean=float(d[k].mean()))
        if rows:
            sens[str(thr)] = dict(subtypes=rows,
                                  mean_dsi=float(np.mean([v["dsi_mean"] for v in rows.values()])))
    out["resolved_weight_sensitivity"] = sens
    out["seconds"] = time.time() - t0

    arrays = {}
    for name, m in metrics.items():
        for s, v in m.items():
            arrays[f"{name}_{s}_dsi"] = v["per_neuron_dsi"]
            arrays[f"{name}_{s}_effect"] = v["per_neuron_effect"]
    np.savez_compressed(RESULTS / "heldout-per-neuron.npz", **arrays)
    (RESULTS / "heldout-results.json").write_text(json.dumps(out, indent=2) + "\n")
    (RESULTS / "generalization-results.json").write_text(json.dumps(by_axis, indent=2) + "\n")
    (RESULTS / "ablation-results.json").write_text(json.dumps(abl, indent=2) + "\n")
    (RESULTS / "geometry-sensitivity.json").write_text(json.dumps(sens, indent=2) + "\n")
    save_time_traces(ev, mech, ctx, bank)
    print(f"heldout stage {time.time()-t0:.1f}s", flush=True)


def save_time_traces(ev, mech, ctx, bank):
    """Subtype-aggregate preferred/null time courses for the transient figure."""
    store = {}
    for sub in SUBTYPES:
        recs = ctx["audit"][sub]["records"]
        if not recs:
            continue
        pol = "ON" if sub.startswith("T4") else "OFF"
        pref_dirs = np.array([r.predicted_direction for r in recs])
        P = np.zeros(FRAMES); N = np.zeros(FRAMES); c = 0
        for d in ("left", "right", "up", "down"):
            names = [n for n, (_s, m) in bank.items()
                     if m.get("direction") == d and m.get("polarity") == pol and m.get("axis") == "speed"]
            names_o = [n for n, (_s, m) in bank.items()
                       if m.get("direction") == OPPOSITE[d] and m.get("polarity") == pol and m.get("axis") == "speed"]
            sel = pref_dirs == d
            if not names or not names_o or not sel.any():
                continue
            yp = np.mean([ev.responses(mech, sub, n)[sel].mean(axis=0) for n in names], axis=0)
            yn = np.mean([ev.responses(mech, sub, n)[sel].mean(axis=0) for n in names_o], axis=0)
            P += yp; N += yn; c += 1
        if c:
            store[f"{sub}_pref"] = P / c
            store[f"{sub}_null"] = N / c
    np.savez_compressed(RESULTS / "time-resolved-results.npz", **store)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "heldout"])
    a = ap.parse_args()
    {"calibrate": stage_calibrate, "heldout": stage_heldout}[a.stage]()


if __name__ == "__main__":
    main()
