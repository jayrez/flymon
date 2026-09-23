"""Experiment 23: is T4 DS a Mi1-Mi4 spatially offset inhibitory veto?

Stages:
  calibrate - geometry metrics of the frozen transforms and calibration verification of
              every preregistered condition (implementation check only; nothing is tuned)
  heldout   - one run on a fresh held-out bank: native, reversed, co-located, geometry
              null (200), Mi4 sign / removal ablations, Mi1 removal, temporal tests,
              static controls, per-neuron reversal, oracle comparison

The model is the E22 fixed-reference adapted single compartment (flymon.adapted_reference)
with Mi1 + Mi4 inputs; geometry manipulations come from flymon.spatial_veto and act only on
MaleCNS anatomy. No gameplay, RAM, labels, reward or RL.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flymon import adapted_reference as ar
from flymon import column_motion as cm
from flymon import spatial_veto as sv
from flymon.optic_columns import circular_stats
import run_fixed_background_experiment as E22
import run_motion_nonlinearity_experiment as E19
import run_tonic_disinhibition_experiment as E21

RESULTS = Path(__file__).resolve().parent / "results" / "experiment-23-spatial-inhibitory-veto"
T4 = E21.T4
FRAMES = E19.FRAMES
HELDOUT_SPEEDS = (0.4, 1.25, 2.5, 3.5)
HELDOUT_WIDTHS = (10, 22)
HELDOUT_CONTRASTS = (0.35, 0.65)
HELDOUT_OFFSETS = (14.0, -14.0)
PERMUTATIONS, PERM_SEED = 200, 23092026
COMMON_TAUS = (2.0, 6.0, 12.0)


def candidate():
    """Frozen E23 candidate: E22 R2 at E21 parameters, Mi1 + Mi4 (Mi9 excluded)."""
    return ar.AdaptedConfig(level="E23", reference="fixed", background=0.5,
                            adapt=ar.ADAPT_FRAMES, G=16.0, beta=2.0, r0_exc=1.0,
                            r0_mi9=0.0, r0_mi4=0.0, E_inh=-0.2, tau_fast=4.0, tau_slow=12.0,
                            drop=("Mi9",))


# ---------------------------------------------------------------- stimuli
def heldout_bank(background=0.5):
    """Fresh held-out bank: speeds / widths / contrasts / offsets never used before."""
    bank = {}
    def add(name, seq, **meta):
        bank[name] = (seq, meta)
    for pol in ("ON", "OFF"):
        for d in ("left", "right", "up", "down"):
            for sp in HELDOUT_SPEEDS:
                add(f"bar_{pol}_{d}_sp{sp}", ar.gray_bar(d, pol, FRAMES, sp, background=background),
                    polarity=pol, direction=d, speed=sp, width=18, contrast=1.0, start_offset=0.0,
                    axis="speed")
            for w in HELDOUT_WIDTHS:
                add(f"bar_{pol}_{d}_w{w}", ar.gray_bar(d, pol, FRAMES, 1.0, bar_px=w, background=background),
                    polarity=pol, direction=d, speed=1.0, width=w, contrast=1.0, start_offset=0.0,
                    axis="width")
            for c in HELDOUT_CONTRASTS:
                add(f"bar_{pol}_{d}_c{c}", ar.gray_bar(d, pol, FRAMES, 1.0, contrast=c, background=background),
                    polarity=pol, direction=d, speed=1.0, width=18, contrast=c, start_offset=0.0,
                    axis="contrast")
            for o in HELDOUT_OFFSETS:
                add(f"bar_{pol}_{d}_o{o:g}", ar.gray_bar(d, pol, FRAMES, 1.0, start_offset=o, background=background),
                    polarity=pol, direction=d, speed=1.0, width=18, contrast=1.0, start_offset=o,
                    axis="position")
    canonical = ar.gray_bar("left", "ON", FRAMES, 1.25, background=background)
    add("ctrl_frozen", cm.temporal_variant(canonical, "frozen_first"), control="frozen")
    add("ctrl_shuffle", cm.temporal_variant(canonical, "shuffle"), control="shuffle")
    add("ctrl_reverse", cm.temporal_variant(canonical, "reverse"), control="reverse")
    add("ctrl_gray", ar.uniform(FRAMES, background), control="gray")
    for pol in ("ON", "OFF"):
        add(f"step_{pol}", ar.full_field_step(pol, FRAMES, 20, 0.5, background), control=f"step_{pol}")
    return bank


# ---------------------------------------------------------------- conditions
def geometry(ctx):
    xy = sv.column_pixels(ctx["col_uv"])
    sides = sv.column_sides(ctx["colindex"])
    rev, drev = sv.transform(ctx["proj"], xy, sides, "reversed")
    col, dcol = sv.transform(ctx["proj"], xy, sides, "colocated")
    return dict(xy=xy, sides=sides, native=ctx["proj"], reversed=rev, colocated=col,
                sign=sv.sign_converted(ctx["proj"]), diag=dict(reversed=drev, colocated=dcol))


def conditions(geo):
    c0 = candidate()
    return {
        "native": (c0, geo["native"]),
        "reversed": (c0, geo["reversed"]),
        "colocated": (c0, geo["colocated"]),
        "S1_mi4_excitatory": (c0, geo["sign"]),
        "S2_mi4_removed": (c0.replace(drop=("Mi9", "Mi4")), geo["native"]),
        "S3_mi1_removed": (c0.replace(drop=("Mi9", "Mi1")), geo["native"]),
        "T1_equal_filters": (c0.replace(tau_slow=c0.tau_fast), geo["native"]),
        "T2_no_filtering": (c0.replace(tau_fast=0.0, tau_slow=0.0), geo["native"]),
        **{f"T3_common_tau_{t:g}": (c0.replace(tau_fast=t, tau_slow=t), geo["native"]) for t in COMMON_TAUS},
        "T1R_reversed_equal_filters": (c0.replace(tau_slow=c0.tau_fast), geo["reversed"]),
        "E22_strong_with_Mi9_phasic": (c0.replace(drop=()), geo["native"]),
    }


def evaluate(sim, cfg, proj, bank=None):
    bank = bank or sim.bank
    on_bank = E22.moving(bank, "ON")
    m = E21.direction_metrics(sim, cfg, subtypes=T4, bank=on_bank, proj=proj)
    s = E21.summarise(m)
    oo = {}
    for sub in T4:
        on = [float(np.mean(sim.scalar(cfg, sub, n, proj))) for n in E22.moving(bank, "ON")]
        off = [float(np.mean(sim.scalar(cfg, sub, n, proj))) for n in E22.moving(bank, "OFF")]
        oo[sub] = dict(on=float(np.mean(on)), off=float(np.mean(off)))
    row = dict(t4_mean=s["t4_mean"], t4_positive=s["t4_positive"],
               t4_negative=int(sum(v["dsi_mean"] < 0 for v in m.values())),
               t4_on_correct=int(sum(v["on"] > v["off"] for v in oo.values())),
               subtypes={k: dict(dsi=v["dsi_mean"], sign_resp=v["sign_consistency_responsive"],
                                 pref=v["pref_rate"], null=v["null_rate"], n=v["n"],
                                 responsive_fraction=v["responsive_fraction"]) for k, v in m.items()},
               amplitude=float(np.mean([v["pref_rate"] + v["null_rate"] for v in m.values()])),
               on_off=oo)
    return row, m


def geometry_metrics(ctx, geo):
    out = {}
    sp = sv.column_spacing(geo["xy"], geo["sides"])
    for sub in T4:
        dirs = [r.predicted_direction for r in ctx["audit"][sub]["records"]]
        rows = {}
        for mode in ("native", "reversed", "colocated"):
            _c1, _c4, d, _w1, w4 = sv.offsets(geo[mode][sub], geo["xy"])
            ok = np.isfinite(d[:, 0])
            pr = sv.axis_projection(np.nan_to_num(d), dirs)[ok]
            mag = np.linalg.norm(d[ok], axis=1)
            ang = np.arctan2(d[ok, 1], d[ok, 0])
            cmean, R = circular_stats(ang)
            rows[mode] = dict(n_with_mi4=int(ok.sum()), n_total=len(dirs),
                              magnitude_median_px=float(np.median(mag)),
                              projection_median_px=float(np.median(pr)),
                              projection_positive_fraction=float(np.mean(pr > 0)),
                              screen_angle_circular_mean_deg=cmean, angle_concentration_R=R)
        errs = {m: geo["diag"][m][sub]["snap_error"] for m in ("reversed", "colocated")}
        rows["snap_error_px"] = {m: dict(median=float(np.median(e)), p90=float(np.quantile(e, .9)))
                                 for m, e in errs.items()}
        rows["weight_totals_preserved"] = bool(all(
            np.allclose(np.asarray(abs(geo[m][sub]["Mi4"]).sum(axis=1)).ravel(),
                        np.asarray(abs(geo["native"][sub]["Mi4"]).sum(axis=1)).ravel())
            for m in ("reversed", "colocated")))
        rows["input_counts_preserved"] = bool(all(
            np.array_equal(np.diff(geo[m][sub]["Mi4"].tocsr().indptr),
                           np.diff(geo["native"][sub]["Mi4"].tocsr().indptr))
            for m in ("reversed", "colocated")))
        out[sub] = rows
    return dict(column_spacing_px=sp, subtypes=out)


def make_sim(ctx, const, bank):
    return E22.make_sim(ctx, const, bank)


# ---------------------------------------------------------------- stages
def stage_calibrate():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ctx, const = E22.contexts()
    geo = geometry(ctx)
    gm = geometry_metrics(ctx, geo)
    (RESULTS / "geometry-metrics.json").write_text(json.dumps(gm, indent=2) + "\n")
    maps = {}
    for mode in ("reversed", "colocated"):
        maps[mode] = {s: dict(rule="p -> p - 2d" if mode == "reversed" else "p -> p - d",
                              offset_before_median_projection=gm["subtypes"][s]["native"]["projection_median_px"],
                              offset_after_median_projection=gm["subtypes"][s][mode]["projection_median_px"],
                              snap_error_px=gm["subtypes"][s]["snap_error_px"][mode],
                              changed_inputs_fraction=float(np.mean(
                                  geo[mode][s]["Mi4"].tocsr().indices != geo["native"][s]["Mi4"].tocsr().indices))
                              if geo[mode][s]["Mi4"].nnz == geo["native"][s]["Mi4"].nnz else None)
                      for s in T4}
    (RESULTS / "transform-mappings.json").write_text(json.dumps(maps, indent=2) + "\n")
    bank = E22.gray_bank("calibration", 0.5)
    sim = make_sim(ctx, const, bank)
    cal = {}
    for name, (cfg, proj) in conditions(geo).items():
        row, _m = evaluate(sim, cfg, proj)
        row["config"] = cfg.to_json()
        cal[name] = row
        print(f"[cal {name:<28}] T4 {row['t4_mean']:+.4f} (+{row['t4_positive']}/-{row['t4_negative']}) "
              f"ON>OFF {row['t4_on_correct']}/4 amp {row['amplitude']:.4f}", flush=True)
    (RESULTS / "calibration-verification.json").write_text(json.dumps(dict(
        note="implementation verification only; nothing was tuned on these values",
        candidate=candidate().to_json(), conditions=cal, seconds=time.time() - t0), indent=2) + "\n")
    print(f"calibration verification {time.time()-t0:.1f}s", flush=True)


def stage_heldout():
    t0 = time.time()
    timing = {}
    ctx, const = E22.contexts()
    geo = geometry(ctx)
    bank = heldout_bank()
    sim = make_sim(ctx, const, bank)
    out = dict(candidate=candidate().to_json(), bank=dict(
        speeds=HELDOUT_SPEEDS, widths=HELDOUT_WIDTHS, contrasts=HELDOUT_CONTRASTS, offsets=HELDOUT_OFFSETS,
        n_stimuli=len(bank)))

    t1 = time.time()
    conds, per_neuron = {}, {}
    for name, (cfg, proj) in conditions(geo).items():
        row, m = evaluate(sim, cfg, proj)
        row["config"] = cfg.to_json()
        conds[name] = row
        if name in ("native", "reversed", "colocated", "S1_mi4_excitatory", "T1R_reversed_equal_filters"):
            for s, v in m.items():
                per_neuron[f"{name}_{s}_dsi"] = v["per_neuron_dsi"]
                per_neuron[f"{name}_{s}_effect"] = v["per_neuron_effect"]
        print(f"[heldout {name:<28}] T4 {row['t4_mean']:+.4f} (+{row['t4_positive']}/-{row['t4_negative']}) "
              f"ON>OFF {row['t4_on_correct']}/4 amp {row['amplitude']:.4f} "
              f"{ {k: round(v['dsi'], 3) for k, v in row['subtypes'].items()} }", flush=True)
    out["conditions"] = conds
    timing["conditions"] = time.time() - t1

    # ---- static controls ----
    t1 = time.time()
    st = make_sim(ctx, const, E22.static_bank(bank))
    static = {}
    for name in ("native", "reversed"):
        cfg, proj = conditions(geo)[name]
        r, _ = evaluate(st, cfg, proj)
        static[name] = dict(static_t4_dsi=r["t4_mean"], static_on_off=r["on_off"],
                            static_t4_on_correct=r["t4_on_correct"])
        static[name]["controls"] = {c: {s: float(np.mean(sim.scalar(cfg, s, f"ctrl_{c}", proj))) for s in T4}
                                    for c in ("gray", "frozen", "shuffle", "reverse")}
        static[name]["steps"] = {s: {p: float(np.mean(sim.scalar(cfg, s, f"step_{p}", proj)))
                                     for p in ("ON", "OFF")} for s in T4}
        print(f"[static {name}] DSI {r['t4_mean']:+.4f} ON>OFF {r['t4_on_correct']}/4", flush=True)
    out["static"] = static
    timing["static"] = time.time() - t1

    # ---- geometry null (200) ----
    t1 = time.time()
    cfg = candidate()
    perms = E22.permuted_projections(ctx, PERMUTATIONS, PERM_SEED)
    speed_on = {n: v for n, v in bank.items() if v[1].get("axis") == "speed" and v[1]["polarity"] == "ON"}
    g = E22.geometry_null(sim, cfg, perms, speed_on, workers=E22.WORKERS)
    g["endpoint"] = "T4 mean DSI, held-out speed axis, ON bars, all partner classes permuted"
    out["geometry_null"] = {k: v for k, v in g.items() if k != "null_samples"}
    print(f"[geometry] observed {g['observed']:+.4f} null {g['null_mean']:+.4f}±{g['null_sd']:.4f} "
          f"residual {g['residual_fraction']}", flush=True)
    timing["geometry_null"] = time.time() - t1

    # ---- oracle (benchmark only): native and reversed Mi4 geometry ----
    t1 = time.time()
    from flymon import motion_nonlinearity as mn
    oracle = mn.Mechanism(family="OPPONENT_SIGNED", tau_fast=cfg.tau_fast, tau_slow=cfg.tau_slow)
    ev = E19.Evaluator(ctx, bank, sim.lum)
    orc = {}
    for name in ("native", "reversed"):
        om = E19.direction_metrics(ev, oracle, 0, proj=geo[name])
        orc[name] = dict(t4_mean=float(np.mean([om[s]["dsi_mean"] for s in T4])),
                         subtypes={s: dict(dsi=om[s]["dsi_mean"], sign_resp=om[s]["sign_consistency_responsive"])
                                   for s in T4})
        for s in T4:
            per_neuron[f"oracle_{name}_{s}_dsi"] = om[s]["per_neuron_dsi"]
    out["oracle"] = orc
    timing["oracle"] = time.time() - t1

    # ---- traces (speed 1.25) ----
    traces = {}
    for name in ("native", "reversed", "colocated"):
        c, proj = conditions(geo)[name]
        for sub in T4:
            pd = np.array([r.predicted_direction for r in ctx["audit"][sub]["records"]])
            for tag in ("pref", "null"):
                acc, cnt = 0, 0
                for d in ("left", "right", "up", "down"):
                    sel = pd == d
                    stim = f"bar_ON_{d if tag == 'pref' else E19.OPPOSITE[d]}_sp1.25"
                    if sel.any():
                        Y, _ = sim.run(c, sub, stim, proj)
                        acc = acc + Y[sel].mean(axis=0); cnt += 1
                traces[f"{name}_{sub}_{tag}"] = acc / cnt
    np.savez_compressed(RESULTS / "traces.npz", **traces)
    np.savez_compressed(RESULTS / "heldout-per-neuron.npz", **per_neuron,
                        **{f"null_samples": np.asarray(g["null_samples"])})
    timing["total"] = time.time() - t0
    out["timing_seconds"] = timing
    (RESULTS / "heldout-results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"heldout stage {time.time()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "heldout"])
    a = ap.parse_args()
    {"calibrate": stage_calibrate, "heldout": stage_heldout}[a.stage]()


if __name__ == "__main__":
    main()
