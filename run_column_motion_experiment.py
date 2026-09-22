"""Experiment 18: column-resolved motion pathway.

Stages:
  geometry   - anatomical T4/T5 receptive-field audit (no stimulus, no labels)
  calibrate  - model hierarchy M0..M3 (+M_HR control) on calibration stimuli only
  heldout    - frozen selected model, evaluated once on held-out stimuli + ablations

Offline and opt-in; stock FlyBrain and the Experiment-17 adapter are untouched.
No gameplay, RAM, action labels, reward or RL.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import sparse

from flymon import column_motion as cm
from flymon import ethology
from flymon.optic_columns import (SUBTYPES, audit_receptive_fields, load_column_coords,
                                  shuffle_columns, subtype_summary)
from flymon.retina import load_mapping

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "redfly-benchmark" / "data"
ANNOT = DATA / "raw" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
EXP5 = ROOT / "results" / "experiment-05-retinotopic"
RESULTS = ROOT / "results" / "experiment-18-column-motion"
FRAMES = 60
CAL_SEEDS = tuple(range(1301, 1321))
HELDOUT_SEEDS = tuple(range(1321, 1341))
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}

CAL_SPEEDS = (0.5, 2.0)
HELDOUT_SPEEDS = (0.25, 1.0, 4.0)
HELDOUT_WIDTHS = (12, 26)
HELDOUT_CONTRASTS = (0.25, 0.5, 0.75)
HELDOUT_OFFSETS = (20.0, -20.0)
HELDOUT_GRATING = ((24.0, 0.0), (48.0, np.pi / 2))

GRID = [cm.ModelConfig(name="M0", temporal=False, rectify_inputs=False, output_exponent=1.0)]
for tf in (1.0, 2.0):
    for ts in (3.0, 5.0, 8.0):
        GRID.append(cm.ModelConfig(name="M1", tau_fast=tf, tau_slow=ts))
        GRID.append(cm.ModelConfig(name="M2", tau_fast=tf, tau_slow=ts, rectify_inputs=True))
        GRID.append(cm.ModelConfig(name="M3", tau_fast=tf, tau_slow=ts, rectify_inputs=True,
                                   output_exponent=2.0))
GRID.append(cm.ModelConfig(name="M_HR", tau_fast=1.0, tau_slow=5.0, correlator=True))


# ---------------------------------------------------------------- shared setup
def load_all():
    records, _ = load_mapping(EXP5 / "retina-mapping.json")
    meta = np.load(DATA / "brain.npz")
    hexes, cart = load_column_coords(DATA / "brain.npz", ANNOT)
    W = sparse.load_npz(DATA / "weights.npz").tocsr()
    return records, meta, hexes, cart, W


def stimulus_bank(kind):
    """name -> (float sequence, metadata). Luminance controlled; labels are not inputs."""
    bank = {}
    def add(name, seq, **meta):
        bank[name] = (seq, meta)
    if kind == "calibration":
        for pol in ("ON", "OFF"):
            for sp in CAL_SPEEDS:
                for d in ("left", "right", "up", "down"):
                    add(f"bar_{pol}_{d}_sp{sp}", cm.parametric_bar(d, pol, FRAMES, sp),
                        polarity=pol, direction=d, speed=sp, width=18, contrast=1.0)
        canonical = cm.parametric_bar("left", "OFF", FRAMES, 2.0)
        add("ctrl_frozen", cm.temporal_variant(canonical, "frozen_first"), control="frozen")
        add("ctrl_shuffle", cm.temporal_variant(canonical, "shuffle"), control="shuffle")
        add("ctrl_reverse", cm.temporal_variant(canonical, "reverse"), control="reverse")
        add("ctrl_gray", np.full((FRAMES, cm.H, cm.W), 0.5, np.float32), control="gray")
    else:
        for pol in ("ON", "OFF"):
            for d in ("left", "right", "up", "down"):
                for sp in HELDOUT_SPEEDS:
                    add(f"bar_{pol}_{d}_sp{sp}", cm.parametric_bar(d, pol, FRAMES, sp),
                        polarity=pol, direction=d, speed=sp, width=18, contrast=1.0, axis="speed")
                for w in HELDOUT_WIDTHS:
                    add(f"bar_{pol}_{d}_w{w}", cm.parametric_bar(d, pol, FRAMES, 1.0, bar_px=w),
                        polarity=pol, direction=d, speed=1.0, width=w, contrast=1.0, axis="width")
                for c in HELDOUT_CONTRASTS:
                    add(f"bar_{pol}_{d}_c{c}", cm.parametric_bar(d, pol, FRAMES, 1.0, contrast=c),
                        polarity=pol, direction=d, speed=1.0, width=18, contrast=c, axis="contrast")
                for o in HELDOUT_OFFSETS:
                    add(f"bar_{pol}_{d}_o{o}", cm.parametric_bar(d, pol, FRAMES, 1.0, start_offset=o),
                        polarity=pol, direction=d, speed=1.0, width=18, contrast=1.0, axis="position")
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


def e16_bank():
    """The exact unchanged Experiment-16 catalogue, reported separately."""
    return {k: (v.astype(np.float32)[..., 0] / 255.0, {}) for k, v in ethology.catalogue().items()}


def evaluate(proj, bank, cfg, records, colindex, uv):
    """{stimulus: {subtype: per-neuron mean response}}"""
    out = {}
    for name, (seq, _meta) in bank.items():
        rgba = cm.to_rgba(seq) if seq.ndim == 3 else seq
        lum = cm.column_luminance(rgba, records, colindex, uv)
        if cfg.correlator:
            out[name] = cm.hr_responses(proj, lum, cfg)
        else:
            out[name] = cm.responses(proj, cm.cell_signals(lum, cfg), cfg)
    return out


def direction_metrics(resp, bank, audit, polarity_for):
    """Per-subtype preferred-vs-null using the ANATOMY-predicted direction per neuron."""
    metrics = {}
    for sub, entry in audit.items():
        recs = entry["records"]
        if not recs:
            continue
        pol = polarity_for(sub)
        pref_dirs = np.array([r.predicted_direction for r in recs])
        pref = np.zeros(len(recs)); null = np.zeros(len(recs)); ok = np.zeros(len(recs), bool)
        for d in ("left", "right", "up", "down"):
            names = [n for n, (_s, m) in bank.items()
                     if m.get("direction") == d and m.get("polarity") == pol]
            names_o = [n for n, (_s, m) in bank.items()
                       if m.get("direction") == OPPOSITE[d] and m.get("polarity") == pol]
            if not names or not names_o:
                continue
            sel = pref_dirs == d
            if not sel.any():
                continue
            p = np.mean([resp[n][sub] for n in names if sub in resp[n]], axis=0)
            q = np.mean([resp[n][sub] for n in names_o if sub in resp[n]], axis=0)
            pref[sel] = p[sel]; null[sel] = q[sel]; ok[sel] = True
        if not ok.any():
            continue
        p, q = pref[ok], null[ok]
        d_ = cm.dsi(p, q)
        responsive = (p + q) > 0
        sign_resp = float(np.mean((p - q)[responsive] > 0)) if responsive.any() else 0.0
        metrics[sub] = dict(n=int(ok.sum()), pref_rate=float(p.mean()), null_rate=float(q.mean()),
                            effect=float((p - q).mean()), dsi_mean=float(d_.mean()),
                            dsi_median=float(np.median(d_)),
                            sign_consistency=float(np.mean((p - q) > 0)),
                            responsive_fraction=float(responsive.mean()),
                            sign_consistency_responsive=sign_resp,
                            per_neuron_dsi=d_, per_neuron_effect=(p - q))
    return metrics


# ---------------------------------------------------------------- stages
def stage_geometry():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    records, meta, hexes, cart, W = load_all()
    audit = audit_receptive_fields(W, meta["cell_type"], meta["side"], cart, meta["ids"])
    summary = subtype_summary(audit)
    for sub, s in summary.items():
        if s.get("n_eligible"):
            print(f"{sub}: n={s['n_eligible']}/{s['n_total']} ({100*s['eligible_fraction']:.1f}%) "
                  f"angle={s['mean_angle_deg']:6.1f} R={s['concentration']:.3f} "
                  f"|v|={s['mean_magnitude']:.3f} pred={s['predicted_direction']}", flush=True)
    (RESULTS / "geometry-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    arrays = {}
    for sub, entry in audit.items():
        recs = entry["records"]
        if not recs:
            continue
        arrays[f"{sub}_angle"] = np.array([r.angle for r in recs])
        arrays[f"{sub}_mag"] = np.array([r.magnitude for r in recs])
        arrays[f"{sub}_offset"] = np.array([r.offset for r in recs])
        arrays[f"{sub}_screen"] = np.array([r.screen_offset for r in recs])
        arrays[f"{sub}_side"] = np.array([r.side for r in recs])
        arrays[f"{sub}_pred"] = np.array([r.predicted_direction for r in recs])
        arrays[f"{sub}_id"] = np.array([r.flywire_id for r in recs])
    np.savez_compressed(RESULTS / "geometry-audit.npz", **arrays)
    (RESULTS / "geometry-runtime.json").write_text(json.dumps(
        dict(seconds=time.time() - t0, min_partners=3), indent=2) + "\n")
    print(f"geometry stage {time.time()-t0:.1f}s", flush=True)


def stage_calibrate():
    t0 = time.time()
    records, meta, hexes, cart, W = load_all()
    audit = audit_receptive_fields(W, meta["cell_type"], meta["side"], cart, meta["ids"])
    colindex = cm.column_index(records)
    uv = np.asarray([(r.u, r.v) for r in records], np.float32)
    proj = cm.build_projections(W, meta["cell_type"], hexes, meta["side"], audit, colindex)
    bank = stimulus_bank("calibration")
    polarity_for = lambda sub: "ON" if sub.startswith("T4") else "OFF"
    table = {}
    for cfg in GRID:
        resp = evaluate(proj, bank, cfg, records, colindex, uv)
        m = direction_metrics(resp, bank, audit, polarity_for)
        if not m:
            continue
        dsis = [abs(v["dsi_mean"]) for v in m.values()]
        signs = [v["sign_consistency"] for v in m.values()]
        # dynamic vs frozen on the canonical control
        dyn = np.mean([resp[f"bar_OFF_left_sp2.0"][s].mean() for s in m if f"bar_OFF_left_sp2.0" in resp and s in resp["bar_OFF_left_sp2.0"]])
        fro = np.mean([resp["ctrl_frozen"][s].mean() for s in m if s in resp["ctrl_frozen"]])
        key = f"{cfg.name}|tf{cfg.tau_fast}|ts{cfg.tau_slow}|p{cfg.output_exponent}"
        table[key] = dict(config=cfg.to_json(), mean_abs_dsi=float(np.mean(dsis)),
                          min_sign=float(np.min(signs)), mean_sign=float(np.mean(signs)),
                          dynamic=float(dyn), frozen=float(fro),
                          dynamic_vs_frozen=float(dyn - fro),
                          per_subtype={k: dict(dsi_mean=v["dsi_mean"], effect=v["effect"],
                                               sign_consistency=v["sign_consistency"])
                                       for k, v in m.items()})
        print(f"[cal] {key:<28} meanDSI {table[key]['mean_abs_dsi']:+.4f} "
              f"minSign {table[key]['min_sign']:.2f} dyn-froz {table[key]['dynamic_vs_frozen']:+.4f}", flush=True)
    (RESULTS / "calibration-results.json").write_text(json.dumps(table, indent=2) + "\n")
    (RESULTS / "model-configs.json").write_text(json.dumps([c.to_json() for c in GRID], indent=2) + "\n")

    # preregistered selection: simplest model meeting the minimum meaningful effect
    THRESH_DSI, THRESH_SIGN = 0.05, 0.6
    order = {"M0": 0, "M1": 1, "M2": 2, "M3": 3}
    eligible = [(order[v["config"]["name"]], k) for k, v in table.items()
                if v["config"]["name"] in order
                and v["mean_abs_dsi"] >= THRESH_DSI and v["min_sign"] >= THRESH_SIGN
                and v["dynamic_vs_frozen"] > 0]
    if eligible:
        best_rank = min(e[0] for e in eligible)
        cands = [k for r, k in eligible if r == best_rank]
        selected = max(cands, key=lambda k: table[k]["mean_abs_dsi"])
        null_set = False
    else:
        bio = {k: v for k, v in table.items() if v["config"]["name"] in order}
        selected = max(bio, key=lambda k: bio[k]["mean_abs_dsi"])
        null_set = True
    (RESULTS / "selected-config.json").write_text(json.dumps(dict(
        selected=selected, config=table[selected]["config"],
        calibration_mean_abs_dsi=table[selected]["mean_abs_dsi"],
        min_sign_consistency=table[selected]["min_sign"],
        thresholds=dict(mean_abs_dsi=THRESH_DSI, min_sign=THRESH_SIGN),
        met_preregistered_threshold=not null_set,
        selection_rule="simplest model (M0<M1<M2<M3) meeting mean|DSI|>=0.05, min sign>=0.6 "
                       "and dynamic>frozen on calibration stimuli only; M_HR never selected",
        calibration_seconds=time.time() - t0), indent=2) + "\n")
    print(f"SELECTED: {selected}  (met threshold: {not null_set})  {time.time()-t0:.1f}s", flush=True)


def stage_heldout():
    t0 = time.time()
    sel = json.loads((RESULTS / "selected-config.json").read_text())
    cfg = cm.ModelConfig(**sel["config"])
    records, meta, hexes, cart, W = load_all()
    ct = meta["cell_type"]
    audit = audit_receptive_fields(W, ct, meta["side"], cart, meta["ids"])
    colindex = cm.column_index(records)
    uv = np.asarray([(r.u, r.v) for r in records], np.float32)
    proj = cm.build_projections(W, ct, hexes, meta["side"], audit, colindex)
    polarity_for = lambda sub: "ON" if sub.startswith("T4") else "OFF"

    banks = {"heldout": stimulus_bank("heldout"), "e16_canonical": e16_bank()}
    store = {}
    for bname, bank in banks.items():
        resp = evaluate(proj, bank, cfg, records, colindex, uv)
        if bname == "heldout":
            m = direction_metrics(resp, bank, audit, polarity_for)
            store["heldout"] = m
            store["heldout_by_axis"] = {}
            for axis in ("speed", "width", "contrast", "position"):
                sub_bank = {n: v for n, v in bank.items() if v[1].get("axis") == axis}
                if sub_bank:
                    store["heldout_by_axis"][axis] = direction_metrics(
                        {k: resp[k] for k in sub_bank}, sub_bank, audit, polarity_for)
            store["controls"] = {c: {s: float(resp[f"ctrl_{c}"][s].mean())
                                     for s in resp[f"ctrl_{c}"]}
                                 for c in ("frozen", "shuffle", "reverse", "gray")}
            store["dynamic_reference"] = {s: float(np.mean([resp[n][s] for n, v in bank.items()
                                                            if v[1].get("axis") == "speed" and s in resp[n]], axis=0).mean())
                                          for s in SUBTYPES if any(s in resp[n] for n in bank)}
            # ON/OFF specificity
            onoff = {}
            for s in SUBTYPES:
                on = [resp[n][s] for n, v in bank.items() if v[1].get("polarity") == "ON" and s in resp[n]]
                off = [resp[n][s] for n, v in bank.items() if v[1].get("polarity") == "OFF" and s in resp[n]]
                if on and off:
                    onoff[s] = dict(on=float(np.mean(on)), off=float(np.mean(off)))
            store["on_off"] = onoff
        else:
            store["e16_canonical_rates"] = {n: {s: float(resp[n][s].mean()) for s in resp[n]}
                                            for n in resp}

    # ---- ablations (re-using the frozen selected config) ----
    ablations = {}
    rng = np.random.default_rng(HELDOUT_SEEDS[0])
    perm = shuffle_columns(ct, cm.ALL_PARTNERS, rng)
    audit_shuf = audit_receptive_fields(W, ct, meta["side"], cart, meta["ids"], column_permutation=perm)
    hex_shuf = hexes.copy()
    ctv = ct.astype(str)
    for t, order in perm.items():
        idx = np.flatnonzero(ctv == t)
        if len(idx) == len(order):
            hex_shuf[idx] = hexes[idx][order]
    proj_shuf = cm.build_projections(W, ct, hex_shuf, meta["side"], audit_shuf, colindex)
    resp_shuf = evaluate(proj_shuf, banks["heldout"], cfg, records, colindex, uv)
    ablations["geometry_shuffle"] = direction_metrics(resp_shuf, banks["heldout"], audit_shuf, polarity_for)

    flat = cm.ModelConfig(**{**sel["config"], "tau_slow": cfg.tau_fast})
    ablations["temporal_flat"] = direction_metrics(
        evaluate(proj, banks["heldout"], flat, records, colindex, uv), banks["heldout"], audit, polarity_for)

    if cfg.rectify_inputs:
        simpler = cm.ModelConfig(**{**sel["config"], "rectify_inputs": False, "output_exponent": 1.0})
        ablations["no_rectification"] = direction_metrics(
            evaluate(proj, banks["heldout"], simpler, records, colindex, uv), banks["heldout"], audit, polarity_for)

    hr = cm.ModelConfig(name="M_HR", tau_fast=cfg.tau_fast, tau_slow=cfg.tau_slow, correlator=True)
    ablations["explicit_correlator"] = direction_metrics(
        evaluate(proj, banks["heldout"], hr, records, colindex, uv), banks["heldout"], audit, polarity_for)

    def strip(m):
        return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("per_neuron")}
                for k, v in m.items()}
    arrays = {}
    for s, v in store["heldout"].items():
        arrays[f"selected_{s}_dsi"] = v["per_neuron_dsi"]
        arrays[f"selected_{s}_effect"] = v["per_neuron_effect"]
    for ab, m in ablations.items():
        for s, v in m.items():
            arrays[f"{ab}_{s}_dsi"] = v["per_neuron_dsi"]
            arrays[f"{ab}_{s}_effect"] = v["per_neuron_effect"]
    np.savez_compressed(RESULTS / "heldout-per-neuron.npz", **arrays)
    out = dict(selected=sel["selected"], config=sel["config"],
               heldout=strip(store["heldout"]),
               by_axis={a: strip(v) for a, v in store["heldout_by_axis"].items()},
               controls=store["controls"], dynamic_reference=store["dynamic_reference"],
               on_off=store["on_off"], e16_canonical_rates=store["e16_canonical_rates"],
               ablations={k: strip(v) for k, v in ablations.items()},
               seconds=time.time() - t0)
    (RESULTS / "heldout-results.json").write_text(json.dumps(out, indent=2) + "\n")
    for s, v in store["heldout"].items():
        print(f"[heldout] {s}: DSI {v['dsi_mean']:+.4f} effect {v['effect']:+.5f} "
              f"sign {v['sign_consistency']:.2f} n={v['n']}", flush=True)
    print(f"heldout stage {time.time()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["geometry", "calibrate", "heldout"])
    a = ap.parse_args()
    {"geometry": stage_geometry, "calibrate": stage_calibrate, "heldout": stage_heldout}[a.stage]()


if __name__ == "__main__":
    main()
