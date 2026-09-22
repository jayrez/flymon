"""Experiment 20: conductance / compartmental dendritic readout of T4/T5 geometry.

Stages:
  calibrate - C0/C1/C2 parameter grid on calibration stimuli only
  heldout   - frozen selected config once, plus ablations, oracle comparison, null

Reuses Experiment-18 geometry and Experiment-19 stimuli/metrics unchanged.
No gameplay, RAM, action labels, reward or RL.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flymon import column_motion as cm
from flymon import conductance_dendrite as cd
from flymon import motion_nonlinearity as mn
from flymon.optic_columns import SUBTYPES
import run_motion_nonlinearity_experiment as E19

RESULTS = Path(__file__).resolve().parent / "results" / "experiment-20-conductance-dendrite"
GEOMETRY_PERMUTATIONS = 200
OPPOSITE = E19.OPPOSITE
TAU_FAST, TAU_SLOW = 4.0, 12.0


def build_grid():
    grid = []
    for gs in (0.5, 1.0, 2.0, 4.0):
        for Ei in (-0.2, 0.0):
            grid.append(cd.ConductanceConfig(model="C0_SINGLE", tau_fast=TAU_FAST,
                                             tau_slow=TAU_SLOW, g_syn=gs, E_inh=Ei))
            for gc in (0.1, 0.5, 2.0):
                grid.append(cd.ConductanceConfig(model="C1_TWOCOMP", tau_fast=TAU_FAST,
                                                 tau_slow=TAU_SLOW, g_syn=gs, E_inh=Ei,
                                                 g_couple=gc))
                for th in (0.02, 0.08):
                    grid.append(cd.ConductanceConfig(model="C2_TWOCOMP_NL", tau_fast=TAU_FAST,
                                                     tau_slow=TAU_SLOW, g_syn=gs, E_inh=Ei,
                                                     g_couple=gc, threshold=th))
    return grid


class Sim:
    """Caches signals and raw conductances; only membrane integration repeats."""

    def __init__(self, ctx, bank, lum):
        self.ctx, self.bank, self.lum = ctx, bank, lum
        self._sig = {}
        self._cond = {}

    def signals(self, stim, tf, ts):
        k = (stim, tf, ts)
        if k not in self._sig:
            self._sig[k] = cd.signals_for(self.lum[stim],
                                          cd.ConductanceConfig(tau_fast=tf, tau_slow=ts))
        return self._sig[k]

    def conductances(self, stim, sub, cfg, proj=None):
        p = (proj or self.ctx["proj"])
        k = (stim, sub, cfg.tau_fast, cfg.tau_slow, cfg.magnitude_only, id(p))
        if k not in self._cond:
            base = cd.ConductanceConfig(**{**cfg.to_json(), "g_syn": 1.0})
            self._cond[k] = cd.arm_conductances(p[sub], self.signals(stim, cfg.tau_fast,
                                                                     cfg.tau_slow),
                                                sub[:2], base)
        return self._cond[k]

    def response(self, cfg, sub, stim, proj=None):
        raw = self.conductances(stim, sub, cfg, proj)
        scaled = {a: (cfg.g_syn * gE, cfg.g_syn * gI) for a, (gE, gI) in raw.items()}
        Vout, _, _ = cd.integrate(scaled, cfg)
        return cd.response(Vout, cfg)

    def scalar(self, cfg, sub, stim, proj=None):
        Y = self.response(cfg, sub, stim, proj)
        return None if Y is None else Y.mean(axis=1)

    def clear_conductances(self):
        self._cond.clear()


def direction_metrics(sim, cfg, proj=None, audit=None, bank=None):
    audit = audit or sim.ctx["audit"]
    bank = bank or sim.bank
    out = {}
    for sub in SUBTYPES:
        recs = audit[sub]["records"]
        if not recs:
            continue
        pol = "ON" if sub.startswith("T4") else "OFF"
        pref_dirs = np.array([r.predicted_direction for r in recs])
        pref = np.zeros(len(recs)); null = np.zeros(len(recs)); ok = np.zeros(len(recs), bool)
        for d in ("left", "right", "up", "down"):
            names = [n for n, (_s, m) in bank.items()
                     if m.get("direction") == d and m.get("polarity") == pol]
            names_o = [n for n, (_s, m) in bank.items()
                       if m.get("direction") == OPPOSITE[d] and m.get("polarity") == pol]
            sel = pref_dirs == d
            if not names or not names_o or not sel.any():
                continue
            p = np.mean([sim.scalar(cfg, sub, n, proj) for n in names], axis=0)
            q = np.mean([sim.scalar(cfg, sub, n, proj) for n in names_o], axis=0)
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
        return dict(mean_dsi=0.0, mean_abs_dsi=0.0, n_positive=0, n_coherent=0, mean_sign=0.0,
                    min_dsi=0.0, t4_mean=0.0, t5_mean=0.0)
    d = {k: v["dsi_mean"] for k, v in m.items()}
    sg = [v["sign_consistency_responsive"] for v in m.values()]
    t4 = [v for k, v in d.items() if k.startswith("T4")]
    t5 = [v for k, v in d.items() if k.startswith("T5")]
    return dict(mean_dsi=float(np.mean(list(d.values()))),
                mean_abs_dsi=float(np.mean(np.abs(list(d.values())))),
                n_positive=int(sum(1 for v in d.values() if v > 0)),
                n_coherent=int(sum(1 for k, v in d.items()
                                   if v > 0 and m[k]["sign_consistency_responsive"] >= 0.70)),
                mean_sign=float(np.mean(sg)), min_dsi=float(min(d.values())),
                t4_mean=float(np.mean(t4)) if t4 else 0.0,
                t5_mean=float(np.mean(t5)) if t5 else 0.0)


def strip(m):
    return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("per_neuron")}
            for k, v in m.items()}


def stage_calibrate():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ctx = E19.load_context()
    bank = E19.stimulus_bank("calibration")
    lum = E19.luminance_cache(bank, ctx["records"], ctx["colindex"], ctx["uv"])
    sim = Sim(ctx, bank, lum)
    table = {}
    for cfg in build_grid():
        m = direction_metrics(sim, cfg)
        s = summarise(m)
        table[cfg.key()] = dict(config=cfg.to_json(), **s,
                                per_subtype={k: dict(dsi=v["dsi_mean"],
                                                     sign=v["sign_consistency_responsive"])
                                             for k, v in m.items()})
        print(f"[cal] {cfg.key():<52} meanDSI {s['mean_dsi']:+.4f} pos {s['n_positive']}/8 "
              f"coherent {s['n_coherent']}/8 T4 {s['t4_mean']:+.3f} T5 {s['t5_mean']:+.3f}", flush=True)
    (RESULTS / "calibration-results.json").write_text(json.dumps(table, indent=2) + "\n")
    (RESULTS / "model-grid.json").write_text(json.dumps(
        [c.to_json() for c in build_grid()], indent=2) + "\n")

    THRESH, MIN_POS = 0.15, 7
    rank = {m: i for i, m in enumerate(cd.MODELS)}
    ok = [(rank[v["config"]["model"]], k) for k, v in table.items()
          if v["mean_dsi"] >= THRESH and v["n_positive"] >= MIN_POS]
    if ok:
        best_rank = min(r for r, _ in ok)
        cands = [k for r, k in ok if r == best_rank]
        selected = max(cands, key=lambda k: table[k]["mean_dsi"])
        met = True
    else:
        selected = max(table, key=lambda k: table[k]["mean_dsi"])
        met = False
    near = [k for k in table if abs(table[k]["mean_dsi"] - table[selected]["mean_dsi"]) <= 0.01]
    (RESULTS / "selected-config.json").write_text(json.dumps(dict(
        selected=selected, config=table[selected]["config"],
        met_preregistered_threshold=met,
        thresholds=dict(mean_dsi=THRESH, min_positive=MIN_POS),
        calibration=dict(mean_dsi=table[selected]["mean_dsi"],
                         n_positive=table[selected]["n_positive"],
                         n_coherent=table[selected]["n_coherent"]),
        near_equivalent_configs=near[:12], n_near_equivalent=len(near),
        selection_rule="simplest model (C0<C1<C2) with calibration mean DSI>=0.15 and "
                       ">=7/8 positive subtypes; ties within 0.01 reported as sensitivity",
        seconds=time.time() - t0), indent=2) + "\n")
    print(f"SELECTED {selected} (met threshold: {met}); {len(near)} near-equivalent; "
          f"{time.time()-t0:.1f}s", flush=True)


def stage_heldout():
    t0 = time.time()
    ctx = E19.load_context()
    sel = json.loads((RESULTS / "selected-config.json").read_text())
    cfg = cd.ConductanceConfig(**sel["config"])
    bank = E19.stimulus_bank("heldout")
    lum = E19.luminance_cache(bank, ctx["records"], ctx["colindex"], ctx["uv"])
    sim = Sim(ctx, bank, lum)
    out = dict(selected=sel["selected"], config=sel["config"])

    primary = direction_metrics(sim, cfg)
    out["heldout"] = dict(summary=summarise(primary), subtypes=strip(primary))
    print(f"[heldout] meanDSI {summarise(primary)['mean_dsi']:+.4f} "
          f"pos {summarise(primary)['n_positive']}/8 coherent {summarise(primary)['n_coherent']}/8",
          flush=True)

    # all three candidate families on held-out, for the comparison table
    families = {}
    for model in cd.MODELS:
        c = cd.ConductanceConfig(**{**sel["config"], "model": model})
        families[model] = dict(summary=summarise(direction_metrics(sim, c)))
        print(f"[family {model:<16}] meanDSI {families[model]['summary']['mean_dsi']:+.4f} "
              f"pos {families[model]['summary']['n_positive']}/8", flush=True)
    out["families"] = families

    # ---- E19 oracle on the same held-out data ----
    oracle = mn.Mechanism(family="OPPONENT_SIGNED", tau_fast=cfg.tau_fast, tau_slow=cfg.tau_slow)
    ev19 = E19.Evaluator(ctx, bank, lum)
    om = E19.direction_metrics(ev19, oracle, 0)
    out["oracle"] = dict(summary=E19.summarise(om), subtypes=E19.strip(om))
    print(f"[oracle OPPONENT_SIGNED] meanDSI {E19.summarise(om)['mean_dsi']:+.4f} "
          f"coherent {E19.summarise(om)['n_coherent']}/8", flush=True)

    # similarity between conductance model and oracle
    sim_stats = {}
    from scipy.stats import pearsonr, spearmanr
    for sub in SUBTYPES:
        if sub not in primary or sub not in om:
            continue
        a = primary[sub]["per_neuron_dsi"]; b = om[sub]["per_neuron_dsi"]
        n = min(len(a), len(b))
        if n < 10:
            continue
        pr = float(pearsonr(a[:n], b[:n])[0]); sp = float(spearmanr(a[:n], b[:n])[0])
        rmse = float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)) / (np.std(b[:n]) + 1e-12))
        sim_stats[sub] = dict(pearson=pr, spearman=sp, normalised_rmse=rmse, n=int(n))
    out["oracle_similarity_per_neuron"] = sim_stats

    # temporal similarity on subtype-aggregate preferred/null traces
    temporal = {}
    traces = {}
    for sub in SUBTYPES:
        recs = ctx["audit"][sub]["records"]
        if not recs:
            continue
        pol = "ON" if sub.startswith("T4") else "OFF"
        pref_dirs = np.array([r.predicted_direction for r in recs])
        acc = {"cond_pref": 0, "cond_null": 0, "orc_pref": 0, "orc_null": 0, "c": 0}
        for d in ("left", "right", "up", "down"):
            names = [n for n, (_s, m) in bank.items()
                     if m.get("direction") == d and m.get("polarity") == pol and m.get("axis") == "speed"]
            names_o = [n for n, (_s, m) in bank.items()
                       if m.get("direction") == OPPOSITE[d] and m.get("polarity") == pol and m.get("axis") == "speed"]
            sel_n = pref_dirs == d
            if not names or not names_o or not sel_n.any():
                continue
            acc["cond_pref"] += np.mean([sim.response(cfg, sub, n)[sel_n].mean(axis=0) for n in names], axis=0)
            acc["cond_null"] += np.mean([sim.response(cfg, sub, n)[sel_n].mean(axis=0) for n in names_o], axis=0)
            acc["orc_pref"] += np.mean([ev19.responses(oracle, sub, n)[sel_n].mean(axis=0) for n in names], axis=0)
            acc["orc_null"] += np.mean([ev19.responses(oracle, sub, n)[sel_n].mean(axis=0) for n in names_o], axis=0)
            acc["c"] += 1
        if acc["c"]:
            cp = acc["cond_pref"] / acc["c"]; cn = acc["cond_null"] / acc["c"]
            op = acc["orc_pref"] / acc["c"]; on = acc["orc_null"] / acc["c"]
            traces[f"{sub}_cond_pref"] = cp; traces[f"{sub}_cond_null"] = cn
            traces[f"{sub}_orc_pref"] = op; traces[f"{sub}_orc_null"] = on
            dc = cp - cn; do = op - on
            xc = np.correlate(dc - dc.mean(), do - do.mean(), mode="full")
            denom = np.sqrt(np.sum((dc - dc.mean()) ** 2) * np.sum((do - do.mean()) ** 2)) + 1e-12
            lag = int(np.argmax(xc) - (len(dc) - 1))
            temporal[sub] = dict(peak_cross_correlation=float(xc.max() / denom), lag_frames=lag,
                                 zero_lag_correlation=float(np.dot(dc - dc.mean(), do - do.mean()) / denom))
    out["oracle_similarity_temporal"] = temporal
    np.savez_compressed(RESULTS / "time-resolved-traces.npz", **traces)

    # ---- ablations ----
    abl = {}
    for name, c in (("temporal_flat", cd.temporal_flat(cfg)),
                    ("coupling_removed", cd.decouple(cfg)),
                    ("sign_destroyed", cd.destroy_sign(cfg)),
                    ("inhibition_neutralised", cd.neutralise_inhibition(cfg)),
                    ("arm_swapped", cd.swap(cfg))):
        abl[name] = dict(config=c.to_json(), summary=summarise(direction_metrics(sim, c)))
        print(f"[ablation {name:<24}] meanDSI {abl[name]['summary']['mean_dsi']:+.4f} "
              f"pos {abl[name]['summary']['n_positive']}/8", flush=True)

    ctrl = {}
    for c in ("frozen", "shuffle", "reverse", "gray"):
        ctrl[c] = {s: float(np.mean(sim.scalar(cfg, s, f"ctrl_{c}"))) for s in SUBTYPES
                   if s in ctx["proj"]}
    dyn = {s: float(np.mean([np.mean(sim.scalar(cfg, s, n)) for n, (_q, m) in bank.items()
                             if m.get("axis") == "speed"])) for s in SUBTYPES if s in ctx["proj"]}
    abl["controls"] = dict(rates=ctrl, dynamic_reference=dyn)

    # geometry shuffle null (speed subset keeps runtime tractable; documented)
    speed_bank = {n: v for n, v in bank.items() if v[1].get("axis") == "speed"}
    rng = np.random.default_rng(20062026)
    ncol = len(ctx["colindex"])
    nulls = []
    for i in range(GEOMETRY_PERMUTATIONS):
        perm = {t: rng.permutation(ncol) for t in cm.ALL_PARTNERS}
        proj_s = {sub: {t: m[:, perm[t]] for t, m in mats.items()}
                  for sub, mats in ctx["proj"].items()}
        ms = direction_metrics(sim, cfg, proj=proj_s, bank=speed_bank)
        nulls.append(summarise(ms)["mean_dsi"])
        sim.clear_conductances()
        if (i + 1) % 25 == 0:
            print(f"   geometry permutation {i+1}/{GEOMETRY_PERMUTATIONS}", flush=True)
    obs_speed = summarise(direction_metrics(sim, cfg, bank=speed_bank))["mean_dsi"]
    nulls = np.array(nulls)
    abl["geometry_shuffle"] = dict(
        permutations=int(len(nulls)), stimulus_subset="speed axis",
        null_mean=float(nulls.mean()), null_sd=float(nulls.std()),
        null_95=[float(np.quantile(nulls, .025)), float(np.quantile(nulls, .975))],
        observed=float(obs_speed),
        z=float((obs_speed - nulls.mean()) / (nulls.std() + 1e-12)),
        p_empirical=float((1 + np.sum(np.abs(nulls) >= abs(obs_speed))) / (len(nulls) + 1)))
    out["ablations"] = abl

    # ---- generalisation, ON/OFF ----
    by_axis = {}
    for axis in ("speed", "width", "contrast", "position"):
        sb = {n: v for n, v in bank.items() if v[1].get("axis") == axis}
        if sb:
            by_axis[axis] = dict(summary=summarise(direction_metrics(sim, cfg, bank=sb)))
    out["by_axis"] = by_axis
    onoff = {}
    for s in SUBTYPES:
        if s not in ctx["proj"]:
            continue
        on = [np.mean(sim.scalar(cfg, s, n)) for n, (_q, m) in bank.items() if m.get("polarity") == "ON"]
        off = [np.mean(sim.scalar(cfg, s, n)) for n, (_q, m) in bank.items() if m.get("polarity") == "OFF"]
        if on and off:
            onoff[s] = dict(on=float(np.mean(on)), off=float(np.mean(off)))
    out["on_off"] = onoff
    out["seconds"] = time.time() - t0

    arrays = {}
    for s, v in primary.items():
        arrays[f"cond_{s}_dsi"] = v["per_neuron_dsi"]; arrays[f"cond_{s}_effect"] = v["per_neuron_effect"]
    for s, v in om.items():
        arrays[f"oracle_{s}_dsi"] = v["per_neuron_dsi"]; arrays[f"oracle_{s}_effect"] = v["per_neuron_effect"]
    np.savez_compressed(RESULTS / "heldout-per-neuron.npz", **arrays)
    (RESULTS / "heldout-results.json").write_text(json.dumps(out, indent=2) + "\n")
    (RESULTS / "ablation-results.json").write_text(json.dumps(abl, indent=2) + "\n")
    (RESULTS / "generalization-results.json").write_text(json.dumps(by_axis, indent=2) + "\n")
    print(f"heldout stage {time.time()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "heldout"])
    a = ap.parse_args()
    {"calibrate": stage_calibrate, "heldout": stage_heldout}[a.stage]()


if __name__ == "__main__":
    main()
