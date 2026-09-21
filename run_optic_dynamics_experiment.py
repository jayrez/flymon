"""Experiment 17: optic-lobe dynamics repair.

Stages (run separately so calibration is frozen before held-out evaluation):
  audit      - MaleCNS connectivity around the T4/T5 motion pathway (CPU, label-free)
  gain       - Gate 1 retinal gain sweep on calibration seeds only
  calibrate  - candidate dynamics configs on calibration seeds only
  traces     - per-step temporal traces for the reference and selected configs
  heldout    - single evaluation of the frozen selected config on held-out seeds

Uses the Experiment-16 stimuli unchanged. No gameplay, RAM, action labels,
reward, RL, or controller code.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flymon import ethology
from flymon.optic_dynamics import OpticDynamicsConfig, build_repaired, candidates
from flymon.retina import bilinear_sample, full_eye_drive, linear_luminance, load_mapping

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "redfly-benchmark" / "data"
EXP5 = ROOT / "results" / "experiment-05-retinotopic"
RESULTS = ROOT / "results" / "experiment-17-optic-lobe-dynamics"
CAL_SEEDS = tuple(range(1201, 1221))
HELDOUT_SEEDS = tuple(range(1221, 1241))
STEPS_PER_FRAME = 20
GAIN_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)
LAMINA_TONIC_GRID = (0.06, 0.09, 0.12)

# Stimulus subsets (Experiment-16 catalogue, unchanged)
DIRECTION_STIMULI = ("on_left", "on_right", "on_up", "on_down",
                     "off_left", "off_right", "off_up", "off_down")
CAL_STIMULI = DIRECTION_STIMULI + ("gray", "off_left__frozen_first", "on_left__frozen_first",
                                   "off_left__shuffle", "off_left__reverse",
                                   "loom_dark", "recede_dark", "flow_left", "flow_right")
GAIN_STIMULI = ("off_left", "off_right", "on_left", "on_right", "gray", "off_left__frozen_first")
TRACE_STIMULI = ("off_left", "off_right", "off_left__frozen_first")

MEDULLA = ("Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm3", "Tm4", "Tm9", "CT1")


def build_groups(cell_type, side):
    ct = cell_type.astype(str); sd = side.astype(str)
    g = {}
    g["R1_6"] = np.flatnonzero(ct == "R1-6")
    for lam in ("L1", "L2", "L3", "L5"):
        g[lam] = np.flatnonzero(ct == lam)
    for m in MEDULLA:
        g[m] = np.flatnonzero(ct == m)
    for sub in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"):
        for s in ("L", "R"):
            idx = np.flatnonzero((ct == sub) & (sd == s))
            if len(idx):
                g[f"{sub}_{s}"] = idx
    g["LPLC1"] = np.flatnonzero(ct == "LPLC1")
    g["LPLC2"] = np.flatnonzero(ct == "LPLC2")
    g["VS"] = np.flatnonzero(np.char.startswith(ct, "VS"))
    return {k: v for k, v in g.items() if len(v)}


def group_ids(n, groups):
    names = list(groups)
    gid = np.full(n, -1, np.int32)
    for k, name in enumerate(names):
        gid[groups[name]] = k
    return names, gid


def encode(frames_rgba, receptors, visual_count):
    uv = np.asarray([(r.u, r.v) for r in receptors], np.float32)
    previous = np.full(len(receptors), 0.9, np.float32)
    out = []
    for frame in frames_rgba:
        sample = bilinear_sample(linear_luminance(frame), uv)
        out.append(full_eye_drive(visual_count, receptors, sample, previous, temporal=True))
        previous = sample
    return out


def run_trial(brain, seed, drives, gid, G, trace=False):
    brain.reset(seed=seed)
    totals = np.zeros(G, np.int64)
    traces = [] if trace else None
    for fi in range(ethology.FRAMES):
        for _ in range(STEPS_PER_FRAME):
            fired = brain.step(eye_drive=drives[fi])
            g = gid[fired]; g = g[g >= 0]
            counts = np.bincount(g, minlength=G).astype(np.int64)
            totals += counts
            if trace:
                traces.append(counts.astype(np.int32))
    return (totals, np.asarray(traces)) if trace else (totals, None)


def rates(totals, sizes):
    return totals / sizes


def make_brain(cfg):
    from flybrain import FlyBrain
    return build_repaired(FlyBrain, cfg, data=DATA, device="cuda", batch=1)


def sweep(brain, stim_names, seeds, drives_by_stim, gid, G, sizes, label=""):
    out = np.zeros((len(stim_names), len(seeds), G))
    t0 = time.monotonic()
    for si, name in enumerate(stim_names):
        for sj, seed in enumerate(seeds):
            tot, _ = run_trial(brain, seed, drives_by_stim[name], gid, G)
            out[si, sj] = rates(tot, sizes)
        print(f"   {label} [{si+1:>2}/{len(stim_names)}] {name:<24} {time.monotonic()-t0:6.1f}s", flush=True)
    return out


# ---------------- direction-selectivity helpers ----------------
AXIS = {"a": ("left", "right"), "b": ("left", "right"), "c": ("up", "down"), "d": ("up", "down")}


def subtype_pair(sub):
    pol = "on" if sub.startswith("T4") else "off"
    a, b = AXIS[sub[2]]
    return f"{pol}_{a}", f"{pol}_{b}"


def dsi_table(resp, stim_names, names, subtypes, preferred=None):
    """Per-subtype DSI and preferred-null effect. preferred=None -> pick on this data."""
    si = {s: i for i, s in enumerate(stim_names)}
    gi = {g: i for i, g in enumerate(names)}
    out = {}
    pref_out = {}
    for sub in subtypes:
        a, b = subtype_pair(sub)
        if a not in si or b not in si or sub not in gi:
            continue
        ra, rb = resp[si[a], :, gi[sub]], resp[si[b], :, gi[sub]]
        if preferred is None:
            pref, null, rp, rn = (a, b, ra, rb) if ra.mean() >= rb.mean() else (b, a, rb, ra)
        else:
            pref = preferred[sub]; null = b if pref == a else a
            rp, rn = (ra, rb) if pref == a else (rb, ra)
        d = rp - rn
        out[sub] = dict(preferred=pref, null=null,
                        dsi=float(np.mean(d / (rp + rn + 1e-6))),
                        effect=float(d.mean()),
                        sign_consistency=float(np.mean(np.sign(d) == np.sign(d.mean())) if d.mean() != 0 else 0.0),
                        pref_rate=float(rp.mean()), null_rate=float(rn.mean()))
        pref_out[sub] = pref
    return out, pref_out


def pathology(resp, stim_names, names, sizes):
    """Firing-regime diagnostics for the T4/T5 populations."""
    gi = {g: i for i, g in enumerate(names)}
    subs = [g for g in names if g.startswith(("T4", "T5"))]
    r = np.array([resp[:, :, gi[s]].mean() for s in subs])
    return dict(t4t5_mean_rate=float(r.mean()), t4t5_max_rate=float(r.max()),
                saturated=bool(r.max() > 100.0),   # >50% duty cycle over 200 steps
                silent=bool(r.mean() < 0.2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["audit", "gain", "calibrate", "traces", "heldout"])
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.stage == "audit":
        return audit()

    receptors, _ = load_mapping(EXP5 / "retina-mapping.json")
    meta = np.load(DATA / "brain.npz")
    groups = build_groups(meta["cell_type"], meta["side"])
    names, gid = group_ids(len(meta["ids"]), groups)
    sizes = np.array([len(groups[n]) for n in names], float)
    G = len(names)
    cat = ethology.catalogue()
    T4S = [g for g in names if g.startswith("T4")]
    T5S = [g for g in names if g.startswith("T5")]

    if args.stage == "gain":
        brain = make_brain(OpticDynamicsConfig())
        drives = {n: encode(cat[n], receptors, len(brain.visual)) for n in GAIN_STIMULI}
        result = {}
        for gain in GAIN_GRID:
            b = make_brain(OpticDynamicsConfig(name=f"gain{gain}", eye_gain_scale=gain))
            resp = sweep(b, list(GAIN_STIMULI), CAL_SEEDS, drives, gid, G, sizes, f"gain {gain}x")
            d4, _ = dsi_table(resp, list(GAIN_STIMULI), names, T4S)
            d5, _ = dsi_table(resp, list(GAIN_STIMULI), names, T5S)
            gi = {g: i for i, g in enumerate(names)}
            result[str(gain)] = dict(
                r1_6_rate=float(resp[:, :, gi["R1_6"]].mean()),
                lamina_rate=float(np.mean([resp[:, :, gi[l]].mean() for l in ("L1", "L2", "L3") if l in gi])),
                t4_rate=float(np.mean([resp[:, :, gi[s]].mean() for s in T4S])),
                t5_rate=float(np.mean([resp[:, :, gi[s]].mean() for s in T5S])),
                max_abs_t4_dsi=float(max(abs(v["dsi"]) for v in d4.values())),
                max_abs_t5_dsi=float(max(abs(v["dsi"]) for v in d5.values())),
                t4_dsi={k: v["dsi"] for k, v in d4.items()},
                t5_dsi={k: v["dsi"] for k, v in d5.items()},
                **pathology(resp, list(GAIN_STIMULI), names, sizes))
            print(f"gain {gain}: R1-6 {result[str(gain)]['r1_6_rate']:.1f} "
                  f"T5 {result[str(gain)]['t5_rate']:.2f} maxT5|DSI| {result[str(gain)]['max_abs_t5_dsi']:.4f}", flush=True)
        (RESULTS / "gain-sweep.json").write_text(json.dumps(dict(
            grid=list(GAIN_GRID), seeds=list(CAL_SEEDS), stimuli=list(GAIN_STIMULI),
            note="calibration seeds only; objective is direction selectivity, not firing rate",
            results=result), indent=2) + "\n")
        print("saved gain-sweep.json", flush=True)
        return

    if args.stage == "calibrate":
        drives = {n: encode(cat[n], receptors, 6006) for n in CAL_STIMULI}
        # 1) pick lamina tonic level on calibration seeds (dynamic-range objective)
        tonic_scan = {}
        for tonic in LAMINA_TONIC_GRID:
            cfg = OpticDynamicsConfig(name=f"tonic{tonic}",
                                      tonic_add={l: tonic for l in ("L1", "L2", "L3", "L5")})
            b = make_brain(cfg)
            resp = sweep(b, list(DIRECTION_STIMULI), CAL_SEEDS[:8], drives, gid, G, sizes, f"tonic {tonic}")
            gi = {g: i for i, g in enumerate(names)}
            tonic_scan[str(tonic)] = dict(
                lamina_rate=float(np.mean([resp[:, :, gi[l]].mean() for l in ("L1", "L2", "L3")])),
                t5_rate=float(np.mean([resp[:, :, gi[s]].mean() for s in T5S])),
                t4_rate=float(np.mean([resp[:, :, gi[s]].mean() for s in T4S])),
                **pathology(resp, list(DIRECTION_STIMULI), names, sizes))
            print(f"tonic {tonic}: lamina {tonic_scan[str(tonic)]['lamina_rate']:.2f} "
                  f"T5 {tonic_scan[str(tonic)]['t5_rate']:.2f}", flush=True)
        # choose the tonic giving the largest non-pathological T5 dynamic range
        ok = {k: v for k, v in tonic_scan.items() if not v["saturated"]}
        chosen_tonic = float(max(ok, key=lambda k: ok[k]["t5_rate"]))
        print(f"chosen lamina tonic (calibration): {chosen_tonic}", flush=True)

        # 2) candidate configs
        cal = {}
        pref_by_cfg = {}
        for cfg in candidates(chosen_tonic):
            b = make_brain(cfg)
            resp = sweep(b, list(CAL_STIMULI), CAL_SEEDS, drives, gid, G, sizes, cfg.name)
            d4, p4 = dsi_table(resp, list(CAL_STIMULI), names, T4S)
            d5, p5 = dsi_table(resp, list(CAL_STIMULI), names, T5S)
            path = pathology(resp, list(CAL_STIMULI), names, sizes)
            t5h = [s for s in T5S if s[2] in ("a", "b")]
            cal[cfg.name] = dict(config=cfg.to_json(), t4_dsi=d4, t5_dsi=d5, pathology=path,
                                 objective_dsi=float(np.mean([abs(d5[s]["dsi"]) for s in t5h if s in d5])),
                                 objective_sign=float(np.mean([d5[s]["sign_consistency"] for s in t5h if s in d5])),
                                 max_abs_t5_dsi=float(max(abs(v["dsi"]) for v in d5.values())),
                                 max_abs_t4_dsi=float(max(abs(v["dsi"]) for v in d4.values())))
            pref_by_cfg[cfg.name] = {**p4, **p5}
            print(f"[cal] {cfg.name:<26} objDSI {cal[cfg.name]['objective_dsi']:.4f} "
                  f"maxT5|DSI| {cal[cfg.name]['max_abs_t5_dsi']:.4f} "
                  f"sign {cal[cfg.name]['objective_sign']:.2f} sat={path['saturated']}", flush=True)
        viable = {k: v for k, v in cal.items() if not v["pathology"]["saturated"] and not v["pathology"]["silent"]}
        selected = max(viable, key=lambda k: (viable[k]["objective_dsi"], viable[k]["objective_sign"]))
        (RESULTS / "calibration-results.json").write_text(json.dumps(dict(
            seeds=list(CAL_SEEDS), stimuli=list(CAL_STIMULI), tonic_scan=tonic_scan,
            chosen_lamina_tonic=chosen_tonic, candidates=cal), indent=2) + "\n")
        (RESULTS / "candidate-configs.json").write_text(json.dumps(
            [c.to_json() for c in candidates(chosen_tonic)], indent=2) + "\n")
        (RESULTS / "selected-config.json").write_text(json.dumps(dict(
            selected=selected, config=cal[selected]["config"],
            chosen_lamina_tonic=chosen_tonic,
            preferred_directions=pref_by_cfg[selected],
            selection_rule="max mean |DSI| over T5 horizontal subtypes on calibration seeds; "
                           "ties by sign consistency; pathological configs excluded",
            calibration_objective=cal[selected]["objective_dsi"]), indent=2) + "\n")
        print(f"SELECTED CONFIG: {selected}", flush=True)
        return

    if args.stage == "traces":
        sel = json.loads((RESULTS / "selected-config.json").read_text())
        cfgs = [OpticDynamicsConfig(), OpticDynamicsConfig(**sel["config"])]
        drives = {n: encode(cat[n], receptors, 6006) for n in TRACE_STIMULI}
        out = {}
        for cfg in cfgs:
            b = make_brain(cfg)
            for name in TRACE_STIMULI:
                acc = []
                for seed in CAL_SEEDS[:5]:
                    _, tr = run_trial(b, seed, drives[name], gid, G, trace=True)
                    acc.append(tr)
                out[f"{cfg.name}|{name}"] = np.mean(acc, axis=0)
                print(f"   traces {cfg.name} {name}", flush=True)
        np.savez_compressed(RESULTS / "temporal-traces.npz",
                            group_names=np.asarray(names), sizes=sizes,
                            **{k: v for k, v in out.items()})
        print("saved temporal-traces.npz", flush=True)
        return

    if args.stage == "heldout":
        sel = json.loads((RESULTS / "selected-config.json").read_text())
        stim = list(cat)
        drives = {n: encode(cat[n], receptors, 6006) for n in stim}
        store = {}
        for cfg in (OpticDynamicsConfig(), OpticDynamicsConfig(**sel["config"])):
            b = make_brain(cfg)
            resp = sweep(b, stim, HELDOUT_SEEDS, drives, gid, G, sizes, f"heldout {cfg.name}")
            store[cfg.name] = resp
        np.savez_compressed(RESULTS / "heldout-responses.npz",
                            stim_names=np.asarray(stim), seeds=np.asarray(HELDOUT_SEEDS),
                            group_names=np.asarray(names), sizes=sizes,
                            **{f"resp_{k}": v for k, v in store.items()})
        print("saved heldout-responses.npz", flush=True)
        return


def audit():
    """Label-free connectivity audit around the T4/T5 motion pathway."""
    import collections
    from scipy import sparse
    meta = np.load(DATA / "brain.npz")
    ct = meta["cell_type"].astype(str)
    W = sparse.load_npz(DATA / "weights.npz").tocsr()   # W[post, pre]
    out = {}
    # photoreceptor -> lamina sign structure
    R = np.flatnonzero(ct == "R1-6")
    lam_rows = {}
    for lam in ("L1", "L2", "L3", "L5"):
        post = np.flatnonzero(ct == lam)
        d = W[post][:, R].data
        if len(d):
            lam_rows[lam] = dict(edges=int(len(d)), mean_weight=float(d.mean()),
                                 negative_fraction=float((d < 0).mean()), sum_weight=float(d.sum()))
    out["photoreceptor_to_lamina"] = lam_rows
    # inputs onto each T4/T5 subtype
    inputs = {}
    for sub in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"):
        t = np.flatnonzero(ct == sub)
        coo = W[t].tocoo()
        agg = collections.defaultdict(lambda: [0, 0.0])
        for c, v in zip(coo.col, coo.data):
            k = ct[c]; agg[k][0] += 1; agg[k][1] += float(v)
        rows = sorted(agg.items(), key=lambda kv: -abs(kv[1][1]))[:12]
        inputs[sub] = [dict(cell_type=k, edges=int(n), sum_weight=float(s),
                            mean_weight=float(s / n),
                            polarity="excitatory-like" if s > 0 else "inhibitory-like")
                       for k, (n, s) in rows]
    out["t4_t5_inputs"] = inputs
    out["note"] = ("Signed MaleCNS weights, W[post,pre]. Connectivity only: no stimulus, "
                   "no neural activity, no labels.")
    (RESULTS / "connectivity-audit.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["photoreceptor_to_lamina"], indent=2), flush=True)
    for sub in ("T4a", "T5a"):
        print(f"\n{sub} top inputs:", flush=True)
        for r in out["t4_t5_inputs"][sub][:6]:
            print(f"   {r['cell_type']:<10} sumW {r['sum_weight']:+9.2f} ({r['polarity']})", flush=True)
    print("\nsaved connectivity-audit.json", flush=True)


if __name__ == "__main__":
    main()
