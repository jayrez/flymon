"""Experiment 22: fixed adapting background, pre-adaptation and tonic excitation.

Stages:
  calibrate - R0 reproduction, step-polarity audit, 288-configuration funnel
              (ON/OFF -> static -> geometry -> temporal gates), freeze
  heldout   - frozen model once; reference comparison R0/R0g/R1/R2/R3 at E21 parameters
              and at the frozen parameters; cold vs adapted; ablations; geometry null;
              oracle comparison

E18 geometry, E19 trajectories/split/metrics and the E21 conductance model are imported
unchanged. No gameplay, RAM, action labels, reward or RL.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from flymon import adapted_reference as ar
from flymon import column_motion as cm
from flymon import motion_nonlinearity as mn
from flymon.optic_columns import SUBTYPES
import run_motion_nonlinearity_experiment as E19
import run_tonic_disinhibition_experiment as E21

RESULTS = Path(__file__).resolve().parent / "results" / "experiment-22-fixed-background-adaptation"
T4 = E21.T4
T5 = tuple(s for s in SUBTYPES if s.startswith("T5"))
OPPOSITE = E19.OPPOSITE
FRAMES = E19.FRAMES
STATIC_DSI_MAX, GEOM_RESIDUAL_MAX, TEMPORAL_MIN_DROP = 0.05, 0.40, 0.40
GATE_PERMUTATIONS, GATE_SEED = 20, 2209
HELDOUT_PERMUTATIONS, HELDOUT_SEED = 200, 22092026
THRESH_T4, SECONDARY_T4, TIE = 0.15, 0.08, 0.005
LEVEL_RANK = {"A1": 0, "A2": 1, "A3": 2}
INHIBITION_PAIRS = ((0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))


# ---------------------------------------------------------------- stimuli
def gray_bank(kind, background=ar.PRIMARY_BACKGROUND):
    """E19 bank (same names, trajectories and split) re-rendered on the adapting background."""
    bank = {}
    def add(name, seq, **meta):
        bank[name] = (seq, meta)
    speeds = E19.CAL_SPEEDS if kind == "calibration" else E19.HELDOUT_SPEEDS
    for pol in ("ON", "OFF"):
        for d in ("left", "right", "up", "down"):
            for sp in speeds:
                add(f"bar_{pol}_{d}_sp{sp}", ar.gray_bar(d, pol, FRAMES, sp, background=background),
                    polarity=pol, direction=d, speed=sp, width=18, contrast=1.0,
                    start_offset=0.0, axis="speed")
            if kind == "heldout":
                for w in E19.HELDOUT_WIDTHS:
                    add(f"bar_{pol}_{d}_w{w}", ar.gray_bar(d, pol, FRAMES, 1.0, bar_px=w,
                                                           background=background),
                        polarity=pol, direction=d, speed=1.0, width=w, contrast=1.0,
                        start_offset=0.0, axis="width")
                for c in E19.HELDOUT_CONTRASTS:
                    add(f"bar_{pol}_{d}_c{c}", ar.gray_bar(d, pol, FRAMES, 1.0, contrast=c,
                                                           background=background),
                        polarity=pol, direction=d, speed=1.0, width=18, contrast=c,
                        start_offset=0.0, axis="contrast")
                for o in E19.HELDOUT_OFFSETS:
                    add(f"bar_{pol}_{d}_o{o}", ar.gray_bar(d, pol, FRAMES, 1.0, start_offset=o,
                                                           background=background),
                        polarity=pol, direction=d, speed=1.0, width=18, contrast=1.0,
                        start_offset=o, axis="position")
    if kind == "heldout":
        for d in ("left", "right"):
            for per, ph in E19.HELDOUT_GRATING:
                add(f"grating_{d}_p{int(per)}_ph{ph:.2f}",
                    ar.gray_grating(d, FRAMES, 1.0, per, ph, background=background),
                    polarity="GRATING", direction=d, speed=1.0, axis="grating")
    canonical = ar.gray_bar("left", "OFF", FRAMES, 1.0, background=background)
    add("ctrl_frozen", cm.temporal_variant(canonical, "frozen_first"), control="frozen")
    add("ctrl_shuffle", cm.temporal_variant(canonical, "shuffle"), control="shuffle")
    add("ctrl_reverse", cm.temporal_variant(canonical, "reverse"), control="reverse")
    add("ctrl_gray", ar.uniform(FRAMES, background), control="gray")
    for pol in ("ON", "OFF"):
        add(f"step_{pol}", ar.full_field_step(pol, FRAMES, 20, 0.5, background), control=f"step_{pol}")
    return bank


def static_bank(bank):
    """Every moving bar held at its mid-trajectory frame (E16 `frozen_mid`), same metadata."""
    return {n: (cm.temporal_variant(s, "frozen_mid"), m) for n, (s, m) in bank.items()
            if m.get("polarity") in ("ON", "OFF")}


# ---------------------------------------------------------------- simulation
class Sim:
    """Duck-types E21.Sim so E21.direction_metrics / gate code can be reused unchanged."""

    def __init__(self, ctx, bank, lum, sigma, M, chem):
        self.ctx, self.bank, self.lum = ctx, bank, lum
        self.sigma, self.M, self.chem = sigma, M, chem
        self._sig = {}

    def signals(self, stim, cfg):
        k = (stim, cfg.reference, cfg.background, cfg.adapt, cfg.tau_fast, cfg.tau_slow)
        if k not in self._sig:
            L0 = ar.measured_L0(cfg.background) if cfg.reference == "fixed" else None
            self._sig[k] = ar.cell_signals(self.lum[stim], cfg.tau_fast, cfg.tau_slow,
                                           cfg.reference, L0, cfg.adapt)
        return self._sig[k]

    def run(self, cfg, sub, stim, proj=None):
        p = (proj or self.ctx["proj"])[sub]
        return ar.simulate(p, self.signals(stim, cfg), cfg, self.sigma, self.M, self.chem)

    def scalar(self, cfg, sub, stim, proj=None):
        Y, _ = self.run(cfg, sub, stim, proj)
        return Y.mean(axis=1)


def moving(bank, pol=None):
    return {n: v for n, v in bank.items()
            if v[1].get("polarity") in (("ON", "OFF") if pol is None else (pol,))}


def on_off(sim, cfg, subtypes=SUBTYPES, bank=None):
    bank = bank or sim.bank
    out = {}
    for s in subtypes:
        if s not in sim.ctx["proj"]:
            continue
        on = [float(np.mean(sim.scalar(cfg, s, n))) for n in moving(bank, "ON")]
        off = [float(np.mean(sim.scalar(cfg, s, n))) for n in moving(bank, "OFF")]
        out[s] = dict(on=float(np.mean(on)), off=float(np.mean(off)))
    return out


def correct_polarity(oo):
    """T4 should prefer ON, T5 OFF."""
    return {s: (v["on"] > v["off"]) if s.startswith("T4") else (v["off"] > v["on"])
            for s, v in oo.items()}


def t4_dsi(sim, cfg, bank=None, proj=None):
    m = E21.direction_metrics(sim, cfg, subtypes=T4, bank=bank or moving(sim.bank, "ON"), proj=proj)
    return E21.summarise(m), m


def permuted_projections(ctx, n, seed):
    rng = np.random.default_rng(seed)
    ncol = len(ctx["colindex"])
    out = []
    for _ in range(n):
        perm = {t: rng.permutation(ncol) for t in cm.ALL_PARTNERS}
        out.append({s: {t: m[:, perm[t]] for t, m in mats.items()} for s, mats in ctx["proj"].items()})
    return out


_G = {}


def _null_one(i):
    sim, cfg, bank, perms = _G["args"]
    return t4_dsi(sim, cfg, bank, perms[i])[0]["t4_mean"]


def geometry_null(sim, cfg, perms, bank, workers=0):
    obs = t4_dsi(sim, cfg, bank)[0]["t4_mean"]
    if workers:
        import multiprocessing as mp
        _G["args"] = (sim, cfg, bank, perms)
        with mp.get_context("fork").Pool(workers) as pool:
            nulls = np.array(pool.map(_null_one, range(len(perms)), chunksize=1))
    else:
        nulls = np.array([t4_dsi(sim, cfg, bank, p)[0]["t4_mean"] for p in perms])
    return dict(observed=float(obs), null_mean=float(nulls.mean()), null_sd=float(nulls.std()),
                null_95=[float(np.quantile(nulls, .025)), float(np.quantile(nulls, .975))],
                z=float((obs - nulls.mean()) / (nulls.std() + 1e-12)),
                p_empirical=float((1 + np.sum(nulls >= obs)) / (len(nulls) + 1)),
                residual_fraction=float(nulls.mean() / obs) if obs > 0 else None,
                permutations=int(len(nulls)), null_samples=nulls.tolist())


def temporal_drop(obs, flat):
    return float((obs - flat) / obs) if obs > 0 else None


# ---------------------------------------------------------------- audits
POLARITY_EXPECT = {"Mi1": "ON", "Mi4": "ON", "Mi9": "OFF", "Tm1": "OFF", "Tm2": "OFF", "Tm9": "OFF"}


def step_audit(ctx, background=ar.PRIMARY_BACKGROUND, reference="fixed", adapt=ar.ADAPT_FRAMES):
    """Presynaptic-class signals for full-field increments/decrements (no model needed)."""
    L0 = ar.measured_L0(background)
    rows, ok = {}, True
    for pol in ("ON", "OFF"):
        seq = ar.full_field_step(pol, FRAMES, 20, 0.5, background)
        lum = cm.column_luminance(cm.to_rgba(seq), ctx["records"], ctx["colindex"], ctx["uv"])
        sig = ar.cell_signals(lum, 4.0, 12.0, reference, L0 if reference == "fixed" else None,
                              adapt if reference == "fixed" else 0)
        for cls, want in POLARITY_EXPECT.items():
            s = ar.drop_adaptation(sig[cls], adapt if reference == "fixed" else 0, axis=0)
            pre, post = float(s[:20].mean()), float(s[20:].mean())
            pre_max = float(np.abs(s[:20]).max())
            sign_ok = (post > 0) if pol == want else (post < 0)
            silent_ok = pre_max < 1e-9 if reference == "fixed" else True
            rows[f"{cls}_{pol}"] = dict(expected_driver=want, pre_mean=pre, pre_abs_max=pre_max,
                                        post_mean=post, sign_ok=bool(sign_ok),
                                        prestep_silent=bool(silent_ok))
            ok &= bool(sign_ok and silent_ok)
    gray = cm.column_luminance(cm.to_rgba(ar.uniform(FRAMES, background)), ctx["records"],
                               ctx["colindex"], ctx["uv"])
    zero = float(np.abs(ar.contrast(gray, "fixed", L0)).max())
    return dict(reference=reference, adapt=adapt, L0=L0, rows=rows, static_L0_max_contrast=zero,
                passed=bool(ok and (zero < 1e-9 or reference != "fixed")))


def rate_audit(sim, cfg):
    """Presynaptic rates of the frozen model before / after full-field steps."""
    tc = cfg.tonic()
    out = {}
    for pol in ("ON", "OFF"):
        sig = sim.signals(f"step_{pol}", cfg)
        for cls in POLARITY_EXPECT:
            s = ar.drop_adaptation(sig[cls], cfg.adapt, axis=0)
            inh = sim.chem.get(cls) == "inh"
            r0 = ar.class_r0(cfg, cls) if inh else cfg.r0_exc
            r = np.maximum(0.0, r0 + tc.beta * s / sim.sigma)
            out[f"{cls}_{pol}"] = dict(r0=float(r0), pre=float(r[:20].mean()), post=float(r[20:].mean()))
    return out


# ---------------------------------------------------------------- grid
def build_grid():
    grid = []
    for bg, G, b, r0e in itertools.product((0.4, 0.5, 0.6), (4.0, 16.0), (1.0, 2.0), (0.5, 1.0, 2.0)):
        base = ar.AdaptedConfig(reference="fixed", background=bg, G=G, beta=b, r0_exc=r0e)
        grid.append(base.replace(level="A1", adapt=0))
        grid.append(base.replace(level="A2", adapt=ar.ADAPT_FRAMES))
        for m9, m4 in INHIBITION_PAIRS:
            grid.append(base.replace(level="A3", adapt=ar.ADAPT_FRAMES, r0_mi9=m9, r0_mi4=m4))
    return grid


def load_constants():
    sel = json.loads((E21.RESULTS / "selected-config.json").read_text())
    return sel["constants"]


def contexts():
    ctx = E19.load_context()
    const = load_constants()
    return ctx, const


def make_sim(ctx, const, bank):
    lum = E19.luminance_cache(bank, ctx["records"], ctx["colindex"], ctx["uv"])
    return Sim(ctx, bank, lum, const["sigma"], const["M"], const["chemistry"])


def e21_params(reference, adapt=0, **kw):
    return ar.e21_a1().replace(reference=reference, adapt=adapt, **kw)


# ---------------------------------------------------------------- calibrate
_W = {}
WORKERS = 20


def evaluate_config(cfg):
    """One calibration configuration through the gate funnel (runs in a worker process)."""
    sim, st, perms = _W["sims"][cfg.background], _W["statics"][cfg.background], _W["perms"]
    oo = on_off(sim, cfg, SUBTYPES)
    pol = correct_polarity(oo)
    s, m = t4_dsi(sim, cfg)
    row = dict(config=cfg.to_json(), level=cfg.level, on_off=oo, correct_polarity=pol,
               t4_on_correct=int(sum(pol[x] for x in T4)), t5_off_correct=int(sum(pol[x] for x in T5)),
               cal_t4_mean_dsi=s["t4_mean"], cal_t4_positive=s["t4_positive"],
               cal_t4_per_subtype={k: v["dsi_mean"] for k, v in m.items()})
    if row["t4_on_correct"] < 4:
        row["rejected"] = "gate_on_off"
        return row
    so = on_off(st, cfg, T4)
    static_on = int(sum(v["on"] > v["off"] for v in so.values()))
    sd = t4_dsi(st, cfg, bank=moving(st.bank, "ON"))[0]["t4_mean"]
    row.update(static_on_off=so, static_t4_on_correct=static_on, static_t4_dsi=sd)
    if static_on < 4 or abs(sd) > STATIC_DSI_MAX:
        row["rejected"] = "gate_static"
        return row
    g = geometry_null(sim, cfg, perms, moving(sim.bank, "ON"))
    g.pop("null_samples")
    row["geometry"] = g
    if g["residual_fraction"] is None or g["residual_fraction"] >= GEOM_RESIDUAL_MAX:
        row["rejected"] = "gate_geometry"
        return row
    flat = t4_dsi(sim, ar.temporal_flat(cfg))[0]["t4_mean"]
    drop = temporal_drop(s["t4_mean"], flat)
    row.update(temporal_flat_t4=flat, temporal_drop=drop)
    if drop is None or drop < TEMPORAL_MIN_DROP:
        row["rejected"] = "gate_temporal"
    return row


def stage_calibrate():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    timing = {}
    ctx, const = contexts()

    # ---- R0: E21 exploratory A1 under this module ----
    t1 = time.time()
    e19bank = E19.stimulus_bank("calibration")
    sim0 = make_sim(ctx, const, e19bank)
    s0, m0 = E21.summarise(E21.direction_metrics(sim0, ar.e21_a1())), None
    saved = json.loads((E21.RESULTS / "exploratory-a1-tonic-excitation.json").read_text())
    ref = saved["calibration"]["summary"]
    r0_check = dict(reproduced=s0, e21_saved=ref,
                    abs_diff_t4=abs(s0["t4_mean"] - ref["t4_mean"]),
                    abs_diff_all8=abs(s0["mean_dsi"] - ref["mean_dsi"]),
                    matches=bool(abs(s0["t4_mean"] - ref["t4_mean"]) < 1e-9
                                 and abs(s0["mean_dsi"] - ref["mean_dsi"]) < 1e-9))
    timing["r0_reproduction"] = time.time() - t1
    print(f"[R0] T4 {s0['t4_mean']:+.6f} vs saved {ref['t4_mean']:+.6f}; all-8 {s0['mean_dsi']:+.6f} "
          f"vs {ref['mean_dsi']:+.6f} -> match {r0_check['matches']}", flush=True)
    if not r0_check["matches"]:
        raise SystemExit("R0 does not reproduce E21 exploratory A1 - stop and debug")

    # ---- step-polarity audit ----
    audit = {f"R2_L{bg:g}": step_audit(ctx, bg) for bg in (0.4, 0.5, 0.6)}
    audit["R1_L0.5_cold_filters"] = step_audit(ctx, 0.5, "fixed", 0)
    audit["R0_sequence_mean"] = step_audit(ctx, 0.5, "sequence_mean", 0)
    audit["luminance_scale"] = dict(
        display_gray_0p5_reads_linear=ar.rendered_linear(0.5),
        background_linear={f"{bg:g}": dict(display=ar.display_for(bg), L0_measured=ar.measured_L0(bg),
                                           bar_on=ar.rendered_linear(ar.display_for(ar.weber_levels(bg, 1)[0])),
                                           bar_off=ar.rendered_linear(ar.display_for(ar.weber_levels(bg, 1)[1])))
                           for bg in (0.4, 0.5, 0.6)})
    required = [k for k in audit if k.startswith("R2") or k.startswith("R1")]
    for k in required:
        print(f"[audit] {k}: passed={audit[k]['passed']}", flush=True)
    print(f"[audit] R0 (descriptive): pre-step Mi1 ON mean "
          f"{audit['R0_sequence_mean']['rows']['Mi1_ON']['pre_mean']:+.4f}", flush=True)
    (RESULTS / "polarity-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not all(audit[k]["passed"] for k in required):
        raise SystemExit("step-polarity audit failed - stop and debug")

    # ---- calibration funnel ----
    t1 = time.time()
    sims, statics = {}, {}
    for bg in (0.4, 0.5, 0.6):
        b = gray_bank("calibration", bg)
        sims[bg] = make_sim(ctx, const, b)
        statics[bg] = make_sim(ctx, const, static_bank(b))
    perms = permuted_projections(ctx, GATE_PERMUTATIONS, GATE_SEED)
    global _W
    _W = dict(sims=sims, statics=statics, perms=perms)
    import multiprocessing as mp
    grid = build_grid()
    with mp.get_context("fork").Pool(WORKERS) as pool:
        rows = pool.map(evaluate_config, grid, chunksize=1)
    table = {}
    for cfg, row in zip(grid, rows):
        table[cfg.key()] = row
        print(f"[cal] {cfg.key():<52} ON {row['t4_on_correct']}/4 T5OFF {row['t5_off_correct']}/4 "
              f"T4 {row['cal_t4_mean_dsi']:+.4f} ({row['cal_t4_positive']}/4) -> "
              f"{row.get('rejected', 'PASS')}", flush=True)
    timing["calibration_funnel"] = time.time() - t1

    survivors = {k: v for k, v in table.items() if "rejected" not in v}
    eligible = {k: v for k, v in survivors.items() if v["cal_t4_positive"] == 4}
    def pick(cands):
        best = max(v["cal_t4_mean_dsi"] for v in cands.values())
        near = [k for k, v in cands.items() if v["cal_t4_mean_dsi"] >= best - TIE]
        return min(near, key=lambda k: (LEVEL_RANK[cands[k]["level"]],
                                        cands[k]["config"]["r0_exc"] + cands[k]["config"]["r0_mi9"]
                                        + cands[k]["config"]["r0_mi4"], -cands[k]["cal_t4_mean_dsi"]))
    onoff_ok = {k: v for k, v in table.items() if v["t4_on_correct"] == 4}
    if eligible:
        selected, status = pick(eligible), "passed_all_gates"
    elif survivors:
        selected, status = pick(survivors), "passed_all_gates_not_4of4_positive"
    elif onoff_ok:
        selected, status = pick(onoff_ok), "descriptive_best_among_on_off_survivors"
    else:
        selected, status = pick(table), "descriptive_best_overall_no_on_off_survivor"
    reasons = {}
    for v in table.values():
        r = v.get("rejected", "passed_all_gates")
        reasons[r] = reasons.get(r, 0) + 1
    by_level = {}
    for lvl in ("A1", "A2", "A3"):
        rows = {k: v for k, v in table.items() if v["level"] == lvl}
        by_level[lvl] = dict(n=len(rows), on_off_pass=sum(v["t4_on_correct"] == 4 for v in rows.values()),
                             all_gates=sum("rejected" not in v for v in rows.values()),
                             best_cal_t4=max(v["cal_t4_mean_dsi"] for v in rows.values()),
                             best_cal_t4_on_off_pass=max([v["cal_t4_mean_dsi"] for v in rows.values()
                                                          if v["t4_on_correct"] == 4], default=None))
    (RESULTS / "calibration-grid.json").write_text(json.dumps(table, indent=2) + "\n")
    (RESULTS / "gate-outcomes.json").write_text(json.dumps(dict(
        order=["on_off", "static", "geometry", "temporal"], counts=reasons, by_level=by_level,
        thresholds=dict(static_dsi_max=STATIC_DSI_MAX, geometry_residual_max=GEOM_RESIDUAL_MAX,
                        temporal_min_drop=TEMPORAL_MIN_DROP, gate_permutations=GATE_PERMUTATIONS,
                        gate_seed=GATE_SEED)), indent=2) + "\n")
    (RESULTS / "selected-config.json").write_text(json.dumps(dict(
        selected=selected, config=table[selected]["config"], selection_status=status,
        calibration={k: table[selected].get(k) for k in
                     ("cal_t4_mean_dsi", "cal_t4_positive", "cal_t4_per_subtype", "t4_on_correct",
                      "t5_off_correct", "static_t4_dsi", "geometry", "temporal_flat_t4", "temporal_drop")},
        constants=const, L0_measured=ar.measured_L0(table[selected]["config"]["background"]),
        r0_reproduction=r0_check, timing_seconds=timing), indent=2) + "\n")
    print(f"gate outcomes: {reasons}", flush=True)
    print(f"by level: {json.dumps(by_level)}", flush=True)
    print(f"SELECTED {selected} [{status}] total {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------- held-out
def full_eval(sim, cfg, subtypes=SUBTYPES):
    m = E21.direction_metrics(sim, cfg, subtypes=subtypes)
    oo = on_off(sim, cfg, subtypes)
    return m, dict(summary=E21.summarise(m), subtypes=E21.strip(m), on_off=oo,
                   correct_polarity=correct_polarity(oo))


def brief(ev):
    s = ev["summary"]; cp = ev["correct_polarity"]
    return dict(t4_mean=s["t4_mean"], t4_positive=s["t4_positive"], t5_mean=s["t5_mean"],
                all8=s["mean_dsi"], t4_on_correct=int(sum(cp.get(x, False) for x in T4)),
                t5_off_correct=int(sum(cp.get(x, False) for x in T5)),
                t4_subtypes={k: v["dsi_mean"] for k, v in ev["subtypes"].items() if k.startswith("T4")})


def stage_heldout():
    t0 = time.time()
    timing = {}
    ctx, const = contexts()
    sel = json.loads((RESULTS / "selected-config.json").read_text())
    cfg = ar.AdaptedConfig.from_json(sel["config"])
    gbank = gray_bank("heldout", cfg.background)
    sim = make_sim(ctx, const, gbank)
    speed_bank = {n: v for n, v in gbank.items() if v[1].get("axis") == "speed"}
    out = dict(selected=sel["selected"], config=sel["config"], selection_status=sel["selection_status"])

    # ---- primary (once) ----
    t1 = time.time()
    primary_m, primary = full_eval(sim, cfg)
    out["heldout"] = primary
    b = brief(primary)
    print(f"[heldout] T4 {b['t4_mean']:+.4f} ({b['t4_positive']}/4) ON>OFF {b['t4_on_correct']}/4 "
          f"all-8 {b['all8']:+.4f} T5 OFF>ON {b['t5_off_correct']}/4", flush=True)
    timing["primary"] = time.time() - t1

    # ---- reference-condition comparison ----
    t1 = time.time()
    e19bank = E19.stimulus_bank("heldout")
    sim_e19 = make_sim(ctx, const, e19bank)
    sim_g05 = sim if cfg.background == 0.5 else make_sim(ctx, const, gray_bank("heldout", 0.5))
    speed_e19 = {n: v for n, v in e19bank.items() if v[1].get("axis") == "speed"}
    speed_g05 = {n: v for n, v in sim_g05.bank.items() if v[1].get("axis") == "speed"}
    perms50 = permuted_projections(ctx, 50, HELDOUT_SEED + 1)
    refs = {}
    def ref_row(label, s_, c_, sb):
        _m, ev = full_eval(s_, c_)
        g = geometry_null(s_, c_, perms50, {n: v for n, v in sb.items() if v[1]["polarity"] == "ON"},
                          workers=WORKERS)
        g.pop("null_samples")
        flat = E21.summarise(E21.direction_metrics(s_, ar.temporal_flat(c_), subtypes=T4))["t4_mean"]
        row = dict(config=c_.to_json(), **brief(ev), on_off=ev["on_off"],
                   t5_subtypes={k: v["dsi_mean"] for k, v in ev["subtypes"].items() if k.startswith("T5")},
                   sign_resp={k: v["sign_consistency_responsive"] for k, v in ev["subtypes"].items()},
                   geometry_50=g, temporal_flat_t4=flat,
                   temporal_drop=temporal_drop(ev["summary"]["t4_mean"], flat))
        print(f"[ref {label:<22}] T4 {row['t4_mean']:+.4f} ({row['t4_positive']}/4) ON>OFF "
              f"{row['t4_on_correct']}/4 geom {g['residual_fraction']} flat-drop {row['temporal_drop']}", flush=True)
        return row
    frozen_params = dict(G=cfg.G, beta=cfg.beta, r0_exc=cfg.r0_exc, r0_mi9=cfg.r0_mi9, r0_mi4=cfg.r0_mi4)
    for pset, params in (("E21_params", {}), ("frozen_params", frozen_params)):
        refs[pset] = {
            "R0": ref_row(f"{pset} R0", sim_e19, e21_params("sequence_mean", **params), speed_e19),
            "R0g": ref_row(f"{pset} R0g", sim_g05, e21_params("sequence_mean", **params), speed_g05),
            "R1": ref_row(f"{pset} R1", sim_g05, e21_params("fixed", 0, **params), speed_g05),
            "R2": ref_row(f"{pset} R2", sim_g05, e21_params("fixed", ar.ADAPT_FRAMES, **params), speed_g05),
            "R3": ref_row(f"{pset} R3", sim_g05, e21_params("fixed", ar.ADAPT_FRAMES,
                                                            **{**params, "r0_exc": 0.0}), speed_g05),
        }
    e21saved = json.loads((E21.RESULTS / "exploratory-a1-tonic-excitation.json").read_text())
    refs["E21_params"]["R0"]["matches_e21_saved_heldout"] = bool(
        abs(refs["E21_params"]["R0"]["t4_mean"] - e21saved["heldout"]["summary"]["t4_mean"]) < 1e-9)
    out["reference_conditions"] = refs
    timing["reference_conditions"] = time.time() - t1

    # ---- initial condition ----
    t1 = time.time()
    init = {}
    for label, c_ in (("adapted_40", cfg.replace(adapt=ar.ADAPT_FRAMES)),
                      ("cold_filters", cfg.replace(adapt=0)),
                      ("cold_filters_and_membrane", cfg.replace(adapt=0, membrane_cold=True)),
                      ("adapted_30", cfg.replace(adapt=30)), ("adapted_60", cfg.replace(adapt=60))):
        _m, ev = full_eval(sim, c_, T4)
        init[label] = dict(config=c_.to_json(), **brief(ev))
        print(f"[init {label:<26}] T4 {init[label]['t4_mean']:+.4f} ({init[label]['t4_positive']}/4) "
              f"ON>OFF {init[label]['t4_on_correct']}/4", flush=True)
    out["initial_condition"] = init
    timing["initial_condition"] = time.time() - t1

    # ---- ablations ----
    t1 = time.time()
    abl = {}
    variants = {
        "B1_sequence_mean_restored": cfg.replace(reference="sequence_mean", adapt=0),
        "B2_no_tonic_excitation": cfg.replace(r0_exc=0.0),
        "B3_cold_start": cfg.replace(adapt=0, membrane_cold=True),
        "B4_temporal_flat": ar.temporal_flat(cfg),
        "B6_remove_Mi9": cfg.replace(drop=("Mi9",)),
        "B7_remove_Mi4": cfg.replace(drop=("Mi4",)),
        "B8_remove_Mi9_and_Mi4": cfg.replace(drop=("Mi9", "Mi4")),
    }
    for name, c_ in variants.items():
        _m, ev = full_eval(sim, c_)
        abl[name] = dict(config=c_.to_json(), **brief(ev))
        print(f"[ablation {name:<26}] T4 {abl[name]['t4_mean']:+.4f} ({abl[name]['t4_positive']}/4) "
              f"ON>OFF {abl[name]['t4_on_correct']}/4", flush=True)
    abl["B9_controls"] = {c: {s: float(np.mean(sim.scalar(cfg, s, f"ctrl_{c}"))) for s in SUBTYPES
                              if s in ctx["proj"]} for c in ("frozen", "gray", "shuffle", "reverse")}
    st = make_sim(ctx, const, static_bank(gbank))
    abl["B9_controls"]["static_frozen_mid_t4_dsi"] = t4_dsi(st, cfg, bank=moving(st.bank, "ON"))[0]["t4_mean"]
    abl["B9_controls"]["static_on_off"] = on_off(st, cfg, SUBTYPES)
    abl["B10_static_steps"] = {s: {pol: float(np.mean(sim.scalar(cfg, s, f"step_{pol}")))
                                   for pol in ("ON", "OFF")} for s in SUBTYPES if s in ctx["proj"]}
    abl["B10_rate_audit"] = rate_audit(sim, cfg)
    timing["ablations"] = time.time() - t1

    # ---- inhibition decomposition (presence and tonic level) ----
    t1 = time.time()
    dec = {}
    for label, c_ in (("excitation_only", cfg.replace(drop=("Mi9", "Mi4"))),
                      ("plus_Mi9", cfg.replace(drop=("Mi4",))),
                      ("plus_Mi4", cfg.replace(drop=("Mi9",))),
                      ("plus_both", cfg)):
        _m, ev = full_eval(sim, c_, T4)
        dec[label] = dict(**brief(ev))
    for m9, m4 in ((0.0, 0.0),) + INHIBITION_PAIRS:
        c_ = cfg.replace(r0_mi9=m9, r0_mi4=m4, adapt=max(cfg.adapt, ar.ADAPT_FRAMES))
        _m, ev = full_eval(sim, c_, T4)
        dec[f"tonic_mi9_{m9:g}_mi4_{m4:g}"] = dict(**brief(ev))
    out["inhibition_decomposition"] = dec
    timing["inhibition_decomposition"] = time.time() - t1

    # ---- geometry null (200) ----
    t1 = time.time()
    perms = permuted_projections(ctx, HELDOUT_PERMUTATIONS, HELDOUT_SEED)
    g = geometry_null(sim, cfg, perms, {n: v for n, v in speed_bank.items() if v[1]["polarity"] == "ON"},
                      workers=WORKERS)
    g["endpoint"] = "T4 mean DSI, held-out speed axis (ON bars)"
    abl["B5_geometry_shuffle"] = g
    print(f"[B5 geometry] observed {g['observed']:+.4f} null {g['null_mean']:+.4f}±{g['null_sd']:.4f} "
          f"residual {g['residual_fraction']}", flush=True)
    timing["geometry_null"] = time.time() - t1
    out["ablations"] = abl

    # ---- generalisation axes ----
    by_axis = {}
    for axis in ("speed", "width", "contrast", "position", "grating"):
        sb = {n: v for n, v in gbank.items() if v[1].get("axis") == axis}
        if sb:
            by_axis[axis] = E21.summarise(E21.direction_metrics(sim, cfg, bank=sb))
    out["by_axis"] = by_axis

    # ---- oracle and prior models ----
    t1 = time.time()
    oracle = mn.Mechanism(family="OPPONENT_SIGNED", tau_fast=cfg.tau_fast, tau_slow=cfg.tau_slow)
    ev_o = E19.Evaluator(ctx, gbank, sim.lum)
    om = E19.direction_metrics(ev_o, oracle, 0)
    from scipy.stats import pearsonr
    corr = {}
    for s in SUBTYPES:
        if s in primary_m and s in om:
            a, b_ = primary_m[s]["per_neuron_dsi"], om[s]["per_neuron_dsi"]
            corr[s] = float(pearsonr(a, b_)[0]) if np.std(a) > 0 and np.std(b_) > 0 else None
    e21h = json.loads((E21.RESULTS / "heldout-results.json").read_text())
    e20h = json.loads((ROOT_RESULTS / "experiment-20-conductance-dendrite" / "heldout-results.json").read_text())
    out["oracle_comparison"] = dict(
        oracle_gray_bank=dict(summary=E19.summarise(om),
                              t4_mean=float(np.mean([om[s]["dsi_mean"] for s in T4 if s in om])),
                              subtypes={k: v["dsi_mean"] for k, v in om.items()}),
        oracle_e19_bank_saved=dict(t4_mean=e21h["oracle"]["t4_mean"], all8=e21h["oracle"]["summary"]["mean_dsi"]),
        per_neuron_pearson_vs_oracle_gray=corr,
        e21_exploratory_a1_saved=e21saved["heldout"]["summary"],
        e21_frozen_saved=e21h["heldout"]["summary"],
        e20_c2_saved=_e20_c2(e20h))
    timing["oracle"] = time.time() - t1

    # ---- traces (held-out speed 1.5) ----
    traces = {}
    for sub in SUBTYPES:
        recs = ctx["audit"][sub]["records"]
        pd = np.array([r.predicted_direction for r in recs])
        pol = E21.polarity(sub)
        for tag in ("pref", "null"):
            acc, cnt = 0, 0
            for d in ("left", "right", "up", "down"):
                sel_n = pd == d
                stim = f"bar_{pol}_{d if tag == 'pref' else OPPOSITE[d]}_sp1.5"
                if not sel_n.any() or stim not in gbank:
                    continue
                Y, _ = sim.run(cfg, sub, stim)
                acc = acc + Y[sel_n].mean(axis=0); cnt += 1
            if cnt:
                traces[f"{sub}_{tag}"] = acc / cnt
        for pol2 in ("ON", "OFF"):
            Y, _ = sim.run(cfg, sub, f"step_{pol2}")
            traces[f"{sub}_step_{pol2}"] = Y.mean(axis=0)
    for pol2 in ("ON", "OFF"):
        sg = sim.signals(f"step_{pol2}", cfg)
        for cls in POLARITY_EXPECT:
            traces[f"signal_{cls}_step_{pol2}"] = ar.drop_adaptation(sg[cls], cfg.adapt, axis=0).mean(axis=1)
    np.savez_compressed(RESULTS / "traces.npz", **traces)
    arrays = {f"e22_{s}_dsi": v["per_neuron_dsi"] for s, v in primary_m.items()}
    arrays.update({f"e22_{s}_effect": v["per_neuron_effect"] for s, v in primary_m.items()})
    arrays.update({f"oracle_{s}_dsi": v["per_neuron_dsi"] for s, v in om.items()})
    np.savez_compressed(RESULTS / "heldout-per-neuron.npz", **arrays)

    timing["total"] = time.time() - t0
    out["timing_seconds"] = timing
    (RESULTS / "heldout-results.json").write_text(json.dumps(out, indent=2) + "\n")
    (RESULTS / "ablation-results.json").write_text(json.dumps(abl, indent=2) + "\n")
    print(f"heldout stage {time.time()-t0:.1f}s", flush=True)


ROOT_RESULTS = RESULTS.parent


def _e20_c2(e20h):
    """E20 frozen (C2_TWOCOMP_NL) held-out summary, ON/OFF and geometry null, as saved."""
    g = e20h["ablations"]["geometry_shuffle"]
    return dict(selected=e20h["selected"], summary=e20h["heldout"]["summary"], on_off=e20h["on_off"],
                geometry=dict(observed=g["observed"], null_mean=g["null_mean"]),
                temporal_flat=e20h["ablations"]["temporal_flat"].get("summary", e20h["ablations"]["temporal_flat"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "heldout"])
    a = ap.parse_args()
    {"calibrate": stage_calibrate, "heldout": stage_heldout}[a.stage]()


if __name__ == "__main__":
    main()
