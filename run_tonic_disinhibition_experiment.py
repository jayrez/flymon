"""Experiment 21: tonic shunting inhibition and release from inhibition in T4.

Stages:
  calibrate - D0 reproduction, mechanism screen (Gates 1-3), ranking, freeze
  heldout   - frozen model once, E20/E19 comparisons, ablations, geometry null

Reuses E18 geometry and E19 stimuli/split/metrics unchanged; imports the E20 module
only to reproduce it. No gameplay, RAM, action labels, reward or RL.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flymon import column_motion as cm
from flymon import motion_nonlinearity as mn
from flymon import tonic_disinhibition as td
from flymon.optic_columns import SUBTYPES
import run_motion_nonlinearity_experiment as E19

RESULTS = Path(__file__).resolve().parent / "results" / "experiment-21-tonic-disinhibition"
T4 = ("T4a", "T4b", "T4c", "T4d")
OPPOSITE = E19.OPPOSITE
WINDOW = (10, 60)                 # preregistered stimulus window (E19 startup exclusion)
GATE_SPEEDS = (0.5, 1.0, 2.0)     # amendment 1
RELEASE_FRACTION, RELEASE_FRAMES, GATE1_PASS = 0.90, 3, 0.50
OVERLAP_MIN, RIN_MIN = 0.50, 0.05
CLAMP_DROP = 0.40
THRESH_T4, MIN_T4_POS = 0.15, 4
GEOMETRY_PERMUTATIONS = 200


def build_grid():
    grid = []
    for G in (1.0, 4.0, 16.0):
        for b in (0.5, 1.0, 2.0):
            for Ei in (0.0, -0.2):
                for r0 in (0.0, 0.5, 1.0, 2.0):
                    grid.append(td.TonicConfig(family="D1_UNIFORM_TONIC", G=G, beta=b,
                                               r0_exc=r0, r0_inh=r0, E_inh=Ei))
                for r0 in (0.5, 1.0, 2.0):
                    grid.append(td.TonicConfig(family="D2_TONIC_INHIBITION", G=G, beta=b,
                                               r0_exc=0.0, r0_inh=r0, E_inh=Ei))
                    grid.append(td.TonicConfig(family="D3_BOUNDED_TONIC_INHIBITION", G=G,
                                               beta=b, r0_exc=0.0, r0_inh=r0, E_inh=Ei,
                                               bounded=True))
    return grid


class Sim:
    def __init__(self, ctx, bank, lum, sigma, M, chem):
        self.ctx, self.bank, self.lum = ctx, bank, lum
        self.sigma, self.M, self.chem = sigma, M, chem
        self._sig = {}

    def signals(self, stim, cfg):
        k = (stim, cfg.tau_fast, cfg.tau_slow)
        if k not in self._sig:
            self._sig[k] = td.signals_for(self.lum[stim], cfg)
        return self._sig[k]

    def run(self, cfg, sub, stim, proj=None):
        p = (proj or self.ctx["proj"])[sub]
        return td.simulate(p, self.signals(stim, cfg), cfg, self.sigma, self.M, self.chem)

    def scalar(self, cfg, sub, stim, proj=None):
        Y, _ = self.run(cfg, sub, stim, proj)
        return Y.mean(axis=1)


def polarity(sub):
    return "ON" if sub.startswith("T4") else "OFF"


def direction_metrics(sim, cfg, subtypes=SUBTYPES, bank=None, proj=None):
    bank = bank or sim.bank
    out = {}
    for sub in subtypes:
        recs = sim.ctx["audit"][sub]["records"]
        if not recs or sub not in sim.ctx["proj"]:
            continue
        pref_dirs = np.array([r.predicted_direction for r in recs])
        pref = np.zeros(len(recs)); null = np.zeros(len(recs)); ok = np.zeros(len(recs), bool)
        for d in ("left", "right", "up", "down"):
            names = [n for n, (_s, m) in bank.items()
                     if m.get("direction") == d and m.get("polarity") == polarity(sub)]
            names_o = [n for n, (_s, m) in bank.items()
                       if m.get("direction") == OPPOSITE[d] and m.get("polarity") == polarity(sub)]
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
                        dsi_median=float(np.median(d_)), responsive_fraction=float(resp.mean()),
                        sign_consistency_responsive=(float(np.mean((p - q)[resp] > 0))
                                                     if resp.any() else 0.0),
                        per_neuron_dsi=d_, per_neuron_effect=(p - q))
    return out


def summarise(m):
    if not m:
        return dict(mean_dsi=0.0, n_positive=0, n_coherent=0, t4_mean=0.0, t5_mean=0.0,
                    t4_positive=0, mean_sign=0.0)
    d = {k: v["dsi_mean"] for k, v in m.items()}
    t4 = [v for k, v in d.items() if k.startswith("T4")]
    t5 = [v for k, v in d.items() if k.startswith("T5")]
    return dict(mean_dsi=float(np.mean(list(d.values()))),
                n_positive=int(sum(v > 0 for v in d.values())),
                n_coherent=int(sum(1 for k, v in d.items()
                                   if v > 0 and m[k]["sign_consistency_responsive"] >= 0.70)),
                t4_mean=float(np.mean(t4)) if t4 else 0.0,
                t5_mean=float(np.mean(t5)) if t5 else 0.0,
                t4_positive=int(sum(v > 0 for v in t4)),
                mean_sign=float(np.mean([v["sign_consistency_responsive"] for v in m.values()])))


def strip(m):
    return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("per_neuron")}
            for k, v in m.items()}


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def gate_metrics(sim, cfg, speeds=GATE_SPEEDS):
    """Gates 1 and 2 over driven (neuron, stimulus) pairs, T4, preferred direction."""
    lo, hi = WINDOW
    release, overlaps, rin_peaks, n_driven = [], [], [], 0
    for sub in T4:
        recs = sim.ctx["audit"][sub]["records"]
        pref_dirs = np.array([r.predicted_direction for r in recs])
        for d in ("left", "right", "up", "down"):
            sel = pref_dirs == d
            if not sel.any():
                continue
            for sp in speeds:
                stim = f"bar_ON_{d}_sp{sp}"
                if stim not in sim.bank:
                    continue
                Y, aux = sim.run(cfg, sub, stim)
                c = aux["classes"]
                if "Mi1" not in c:
                    continue
                ex_tr = (c["Mi1"]["gE"] - c["Mi1"]["gE0"][:, None])[sel, lo:hi]
                ex = ex_tr.max(axis=1)
                pos = ex[ex > 0]
                if not len(pos):
                    continue
                driven = ex >= 0.10 * np.median(pos)
                if not driven.any():
                    continue
                n_driven += int(driven.sum())
                R = (aux["R_in"] / aux["R_rest"][:, None])[sel, lo:hi]
                for k in np.flatnonzero(driven):
                    win = ex_tr[k] >= 0.5 * ex[k]
                    overlaps.append(float(np.mean(R[k][win] > 1.0 + 1e-12)))
                    rin_peaks.append(float(R[k].max() - 1.0))
                if "Mi9" in c:
                    g9 = c["Mi9"]["gI"][sel, lo:hi]; g90 = c["Mi9"]["gI0"][sel]
                    for k in np.flatnonzero(driven & (g90 > 0)):
                        release.append(longest_run(g9[k] <= RELEASE_FRACTION * g90[k]) >= RELEASE_FRAMES)
    frac_release = float(np.mean(release)) if release else 0.0
    med_overlap = float(np.median(overlaps)) if overlaps else 0.0
    med_rin = float(np.median(rin_peaks)) if rin_peaks else 0.0
    return dict(n_driven_pairs=n_driven, n_release_pairs=len(release),
                release_fraction=frac_release, median_overlap=med_overlap,
                median_rin_increase=med_rin,
                gate1=bool(frac_release >= GATE1_PASS),
                gate2=bool(med_overlap >= OVERLAP_MIN and med_rin >= RIN_MIN))


def gate2_diagnostic(sim, cfg, speeds=GATE_SPEEDS):
    """Why the input-resistance window does or does not overlap excitation.

    Over driven (neuron, stimulus) pairs, inside the excitation window, compares the
    conductance *removed* by Mi9 release with the conductance *added* by Mi1 excitation
    and by Mi4, and measures the lag between peak Mi9 release and peak excitation, in the
    preferred and the null direction. Calibration stimuli only."""
    lo, hi = WINDOW
    res = {}
    for tag, flip in (("preferred", False), ("null", True)):
        rows = dict(mi9_release=[], mi1_add=[], mi4_add=[], lag=[], rin_window=[])
        for sub in T4:
            recs = sim.ctx["audit"][sub]["records"]
            pref_dirs = np.array([r.predicted_direction for r in recs])
            for d in ("left", "right", "up", "down"):
                sel = pref_dirs == d
                if not sel.any():
                    continue
                for sp in speeds:
                    stim = f"bar_ON_{OPPOSITE[d] if flip else d}_sp{sp}"
                    if stim not in sim.bank:
                        continue
                    _, aux = sim.run(cfg, sub, stim)
                    c = aux["classes"]
                    if "Mi1" not in c or "Mi9" not in c:
                        continue
                    ex_tr = (c["Mi1"]["gE"] - c["Mi1"]["gE0"][:, None])[sel, lo:hi]
                    ex = ex_tr.max(axis=1); pos = ex[ex > 0]
                    if not len(pos):
                        continue
                    driven = (ex >= 0.10 * np.median(pos)) & (c["Mi9"]["gI0"][sel] > 0)
                    d9 = (c["Mi9"]["gI0"][sel][:, None] - c["Mi9"]["gI"][sel, lo:hi])
                    d4 = ((c["Mi4"]["gI"] - c["Mi4"]["gI0"][:, None])[sel, lo:hi]
                          if "Mi4" in c else np.zeros_like(d9))
                    R = (aux["R_in"] / aux["R_rest"][:, None])[sel, lo:hi]
                    for k in np.flatnonzero(driven):
                        win = ex_tr[k] >= 0.5 * ex[k]
                        rows["mi9_release"].append(float(d9[k][win].mean()))
                        rows["mi1_add"].append(float(ex_tr[k][win].mean()))
                        rows["mi4_add"].append(float(d4[k][win].mean()))
                        rows["lag"].append(int(np.argmax(d9[k]) - np.argmax(ex_tr[k])))
                        rows["rin_window"].append(float(R[k][win].mean() - 1.0))
        res[tag] = {k: dict(median=float(np.median(v)), mean=float(np.mean(v)), n=len(v))
                    for k, v in rows.items() if v}
        if rows["mi9_release"]:
            ratio = np.array(rows["mi9_release"]) / (np.array(rows["mi1_add"])
                                                     + np.maximum(np.array(rows["mi4_add"]), 0) + 1e-12)
            res[tag]["release_to_added_ratio"] = dict(median=float(np.median(ratio)),
                                                      fraction_above_1=float(np.mean(ratio > 1)))
    return res


def compute_sigma(bank, lum, cfg):
    vals = []
    for n, (_s, m) in bank.items():
        if m.get("polarity") in ("ON", "OFF"):
            s = td.signals_for(lum[n], cfg)
            vals.append(np.concatenate([s[t].ravel() for t in td.T4_CLASSES]))
    return float(np.concatenate(vals).std())


def stage_calibrate():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ctx = E19.load_context()
    bank = E19.stimulus_bank("calibration")
    lum = E19.luminance_cache(bank, ctx["records"], ctx["colindex"], ctx["uv"])
    chem = td.class_chemistry(ctx["proj"])
    M = td.normaliser(ctx["proj"])
    sigma = compute_sigma(bank, lum, td.TonicConfig())
    timing = {}

    # ---- D0: reproduce E20 under this runner ----
    t1 = time.time()
    import run_conductance_dendrite_experiment as E20
    from flymon import conductance_dendrite as cd
    e20sel = json.loads((E20.RESULTS / "selected-config.json").read_text())
    s20 = E20.Sim(ctx, bank, lum)
    d0 = E20.summarise(E20.direction_metrics(s20, cd.ConductanceConfig(**e20sel["config"])))
    d0_check = dict(reproduced_mean_dsi=d0["mean_dsi"],
                    e20_recorded=e20sel["calibration"]["mean_dsi"],
                    abs_difference=abs(d0["mean_dsi"] - e20sel["calibration"]["mean_dsi"]),
                    matches=bool(abs(d0["mean_dsi"] - e20sel["calibration"]["mean_dsi"]) < 1e-9))
    timing["d0_reproduction"] = time.time() - t1
    print(f"[D0] E20 selected calibration mean DSI reproduced {d0['mean_dsi']:.9f} "
          f"vs recorded {e20sel['calibration']['mean_dsi']:.9f} match={d0_check['matches']}", flush=True)

    # ---- Stage 1-3 funnel ----
    t1 = time.time()
    sim = Sim(ctx, bank, lum, sigma, M, chem)
    cal_bank_on = {n: v for n, v in bank.items() if v[1].get("polarity") == "ON"}
    table = {}
    for cfg in build_grid():
        g = gate_metrics(sim, cfg)
        row = dict(config=cfg.to_json(), **g)
        # Calibration T4 DSI is computed for every configuration so that the
        # preregistered fallback ("freeze the best-T4-DSI configuration") is well defined.
        # It never overrides a gate: rejection is decided by the gates alone.
        m = direction_metrics(sim, cfg, subtypes=T4, bank=cal_bank_on)
        s = summarise(m)
        row.update(cal_t4_mean_dsi=s["t4_mean"], cal_t4_positive=s["t4_positive"],
                   cal_t4_per_subtype={k: v["dsi_mean"] for k, v in m.items()})
        if not g["gate1"]:
            row["rejected"] = "gate1_no_inhibitory_release"
        elif not g["gate2"]:
            row["rejected"] = "gate2_no_rin_window_overlapping_excitation"
        else:
            mc = summarise(direction_metrics(sim, cfg.replace(clamp_inhibition=True),
                                             subtypes=T4, bank=cal_bank_on))
            drop = (s["t4_mean"] - mc["t4_mean"]) / s["t4_mean"] if s["t4_mean"] > 0 else 0.0
            row.update(clamped_t4_mean_dsi=mc["t4_mean"], clamp_relative_drop=float(drop))
            row["gate3"] = bool(s["t4_mean"] > 0 and drop >= CLAMP_DROP)
            if not row["gate3"]:
                row["rejected"] = "gate3_inhibition_clamp_not_causal"
        table[cfg.key()] = row
        tag = row.get("rejected", "PASS")
        dsi = f"T4 {row['cal_t4_mean_dsi']:+.4f} ({row['cal_t4_positive']}/4)"
        if "clamp_relative_drop" in row:
            dsi += f" clamp {row['clamp_relative_drop']:+.2f}"
        print(f"[cal] {cfg.key():<58} rel {g['release_fraction']:.2f} ovl {g['median_overlap']:.2f} "
              f"rin {g['median_rin_increase']:+.3f} {dsi} -> {tag}", flush=True)
    timing["mechanism_screen_and_calibration"] = time.time() - t1

    survivors = {k: v for k, v in table.items() if "rejected" not in v}
    rank = {f: i for i, f in enumerate(td.FAMILIES)}
    meets = [(rank[v["config"]["family"]], k) for k, v in survivors.items()
             if v["cal_t4_mean_dsi"] >= THRESH_T4 and v["cal_t4_positive"] >= MIN_T4_POS]
    if meets:
        r = min(x for x, _ in meets)
        selected = max([k for x, k in meets if x == r], key=lambda k: survivors[k]["cal_t4_mean_dsi"])
        status = "met_threshold"
    elif survivors:
        selected = max(survivors, key=lambda k: survivors[k]["cal_t4_mean_dsi"])
        status = "best_of_null_set_among_gate_survivors"
    else:
        # Preregistered fallback: best calibration T4 DSI over all configurations,
        # frozen for descriptive held-out evaluation only.
        selected = max(table, key=lambda k: table[k]["cal_t4_mean_dsi"])
        status = "no_mechanism_valid_configuration"
    fam_best = {}
    for f in td.FAMILIES:
        cand = {k: v for k, v in survivors.items() if v["config"]["family"] == f}
        if cand:
            fam_best[f] = max(cand, key=lambda k: cand[k]["cal_t4_mean_dsi"])
    reasons = {}
    for v in table.values():
        reasons[v.get("rejected", "passed_all_gates")] = reasons.get(v.get("rejected", "passed_all_gates"), 0) + 1

    (RESULTS / "calibration-results.json").write_text(json.dumps(table, indent=2) + "\n")
    (RESULTS / "model-grid.json").write_text(json.dumps([c.to_json() for c in build_grid()], indent=2) + "\n")
    (RESULTS / "selected-config.json").write_text(json.dumps(dict(
        selected=selected, config=table[selected]["config"], selection_status=status,
        met_preregistered_threshold=status == "met_threshold",
        calibration={k: table[selected].get(k) for k in
                     ("cal_t4_mean_dsi", "cal_t4_positive", "release_fraction", "median_overlap",
                      "median_rin_increase", "clamp_relative_drop")},
        family_best_survivor=fam_best, gate_outcomes=reasons,
        constants=dict(sigma=sigma, M=M, chemistry=chem),
        d0_reproduction=d0_check, timing_seconds=timing,
        thresholds=dict(t4_mean_dsi=THRESH_T4, t4_positive=MIN_T4_POS, release=RELEASE_FRACTION,
                        release_frames=RELEASE_FRAMES, gate1_fraction=GATE1_PASS,
                        overlap=OVERLAP_MIN, rin=RIN_MIN, clamp_drop=CLAMP_DROP)),
        indent=2) + "\n")
    diag_keys = [selected] + [k for k in ("D2_TONIC_INHIBITION|G16|b0.5|r0e0|r0i2|Ei0",
                                          "D2_TONIC_INHIBITION|G4|b1|r0e0|r0i1|Ei0") if k in table]
    diagnostics = {k: gate2_diagnostic(sim, td.TonicConfig.from_json(table[k]["config"]))
                   for k in dict.fromkeys(diag_keys)}
    (RESULTS / "gate2-diagnostic.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"gate outcomes: {reasons}", flush=True)
    print(f"SELECTED {selected} [{status}]  total {time.time()-t0:.1f}s", flush=True)


def stage_heldout():
    t0 = time.time()
    timing = {}
    ctx = E19.load_context()
    sel = json.loads((RESULTS / "selected-config.json").read_text())
    cfg = td.TonicConfig.from_json(sel["config"])
    const = sel["constants"]
    bank = E19.stimulus_bank("heldout")
    lum = E19.luminance_cache(bank, ctx["records"], ctx["colindex"], ctx["uv"])
    sim = Sim(ctx, bank, lum, const["sigma"], const["M"], const["chemistry"])
    out = dict(selected=sel["selected"], config=sel["config"], selection_status=sel["selection_status"])

    t1 = time.time()
    primary = direction_metrics(sim, cfg)
    out["heldout"] = dict(summary=summarise(primary), subtypes=strip(primary))
    ps = summarise(primary)
    print(f"[heldout] T4 {ps['t4_mean']:+.4f} ({ps['t4_positive']}/4)  all-8 {ps['mean_dsi']:+.4f} "
          f"({ps['n_positive']}/8)", flush=True)

    # best gate-surviving configuration of each family, for the comparison table only
    # Per-family comparison rows. With no gate survivors, each family's calibration-best
    # configuration (chosen on calibration data only) is evaluated descriptively; the frozen
    # selection above is unaffected.
    caltab = json.loads((RESULTS / "calibration-results.json").read_text())
    fam_keys = dict(sel["family_best_survivor"])
    fam_basis = "best gate survivor per family"
    if not fam_keys:
        fam_basis = "best calibration T4 DSI per family (no gate survivors; descriptive only)"
        for f in td.FAMILIES:
            cand = {k: v for k, v in caltab.items() if v["config"]["family"] == f}
            fam_keys[f] = max(cand, key=lambda k: cand[k]["cal_t4_mean_dsi"])
    fams = {}
    for f, key in fam_keys.items():
        c = td.TonicConfig.from_json(caltab[key]["config"])
        fams[f] = dict(key=key, basis=fam_basis, summary=summarise(direction_metrics(sim, c)))
        print(f"[family {f}] {key} T4 {fams[f]['summary']['t4_mean']:+.4f}", flush=True)
    out["families"] = fams
    timing["heldout_primary_and_families"] = time.time() - t1

    # D0: E20 reproduced on held-out with its own module
    t1 = time.time()
    import run_conductance_dendrite_experiment as E20
    from flymon import conductance_dendrite as cd
    e20sel = json.loads((E20.RESULTS / "selected-config.json").read_text())
    s20 = E20.Sim(ctx, bank, lum)
    d0 = {}
    for model in ("C0_SINGLE", "C2_TWOCOMP_NL"):
        m20 = E20.direction_metrics(s20, cd.ConductanceConfig(**{**e20sel["config"], "model": model}))
        d0[model] = E20.summarise(m20)
        print(f"[D0 {model}] all-8 {d0[model]['mean_dsi']:+.4f} T4 {d0[model]['t4_mean']:+.4f}", flush=True)
    out["d0_e20"] = d0
    timing["d0_heldout"] = time.time() - t1

    # E19 oracle
    t1 = time.time()
    oracle = mn.Mechanism(family="OPPONENT_SIGNED", tau_fast=cfg.tau_fast, tau_slow=cfg.tau_slow)
    ev = E19.Evaluator(ctx, bank, lum)
    om = E19.direction_metrics(ev, oracle, 0)
    osum = E19.summarise(om)
    ot4 = [om[s]["dsi_mean"] for s in T4 if s in om]
    out["oracle"] = dict(summary=osum, t4_mean=float(np.mean(ot4)), subtypes=E19.strip(om))
    from scipy.stats import pearsonr, spearmanr
    simr = {}
    for s in SUBTYPES:
        if s in primary and s in om:
            a, b = primary[s]["per_neuron_dsi"], om[s]["per_neuron_dsi"]
            n = min(len(a), len(b))
            simr[s] = dict(pearson=float(pearsonr(a[:n], b[:n])[0]),
                           spearman=float(spearmanr(a[:n], b[:n])[0]))
    out["oracle_similarity_per_neuron"] = simr
    timing["oracle"] = time.time() - t1

    # mechanism traces (held-out speed 1.5) + held-out descriptive gates
    t1 = time.time()
    traces, temporal = {}, {}
    lo, hi = WINDOW
    for sub in SUBTYPES:
        recs = ctx["audit"][sub]["records"]
        pref_dirs = np.array([r.predicted_direction for r in recs])
        acc = {}
        cnt = 0
        for d in ("left", "right", "up", "down"):
            sel_n = pref_dirs == d
            a, b = f"bar_{polarity(sub)}_{d}_sp1.5", f"bar_{polarity(sub)}_{OPPOSITE[d]}_sp1.5"
            if not sel_n.any() or a not in bank or b not in bank:
                continue
            for tag, stim in (("pref", a), ("null", b)):
                Y, aux = sim.run(cfg, sub, stim)
                c = aux["classes"]
                parts = dict(response=Y, V=aux["V"] - aux["V_rest"][:, None],
                             g_total=aux["g_total"] - aux["g_rest"][:, None],
                             R_in=aux["R_in"] / aux["R_rest"][:, None])
                for cls in c:
                    parts[f"g_{cls}"] = (c[cls]["gE"] + c[cls]["gI"]) - (c[cls]["gE0"] + c[cls]["gI0"])[:, None]
                for k, v in parts.items():
                    acc[f"{tag}_{k}"] = acc.get(f"{tag}_{k}", 0) + v[sel_n].mean(axis=0)
            cnt += 1
        if cnt:
            for k, v in acc.items():
                traces[f"{sub}_{k}"] = v / cnt
        # oracle temporal similarity on preferred-null response traces
        oacc = 0; oc = 0
        for d in ("left", "right", "up", "down"):
            sel_n = pref_dirs == d
            a, b = f"bar_{polarity(sub)}_{d}_sp1.5", f"bar_{polarity(sub)}_{OPPOSITE[d]}_sp1.5"
            if not sel_n.any() or a not in bank:
                continue
            oacc = oacc + (ev.responses(oracle, sub, a)[sel_n].mean(axis=0)
                           - ev.responses(oracle, sub, b)[sel_n].mean(axis=0))
            oc += 1
        if oc and f"{sub}_pref_response" in traces:
            dm = traces[f"{sub}_pref_response"] - traces[f"{sub}_null_response"]
            do = oacc / oc
            xc = np.correlate(dm - dm.mean(), do - do.mean(), mode="full")
            den = np.sqrt(np.sum((dm - dm.mean()) ** 2) * np.sum((do - do.mean()) ** 2)) + 1e-12
            temporal[sub] = dict(peak_xcorr=float(xc.max() / den), lag=int(np.argmax(xc) - (len(dm) - 1)),
                                 zero_lag=float(np.dot(dm - dm.mean(), do - do.mean()) / den))
    np.savez_compressed(RESULTS / "mechanism-traces.npz", **traces)
    out["oracle_similarity_temporal"] = temporal
    out["heldout_gates"] = gate_metrics(sim, cfg, speeds=(0.75, 1.5, 3.0))
    timing["traces"] = time.time() - t1

    # ablations
    t1 = time.time()
    abl = {}
    variants = {
        "A1_no_tonic_inhibition": cfg.replace(r0_inh=0.0),
        "A2_no_release": cfg.replace(no_release=True),
        "A3_clamp_inhibition": cfg.replace(clamp_inhibition=True),
        "A4_neutral_inhibition": cfg.replace(neutral_inhibition=True),
        "A5_temporal_flat": td.temporal_flat(cfg),
        "A7_remove_Mi1": cfg.replace(drop_classes=("Mi1",)),
        "A8_remove_Mi9": cfg.replace(drop_classes=("Mi9",)),
        "A9_remove_Mi4": cfg.replace(drop_classes=("Mi4",)),
    }
    for name, c in variants.items():
        m = direction_metrics(sim, c)
        abl[name] = dict(config=c.to_json(), summary=summarise(m),
                         t4_subtypes={k: v["dsi_mean"] for k, v in m.items() if k.startswith("T4")})
        print(f"[ablation {name:<24}] T4 {abl[name]['summary']['t4_mean']:+.4f} "
              f"({abl[name]['summary']['t4_positive']}/4) all-8 {abl[name]['summary']['mean_dsi']:+.4f}", flush=True)
    abl["A4_neutral_inhibition"]["note"] = ("no-op: frozen E_inh already equals E_leak (pure shunt)"
                                            if cfg.E_inh == cfg.E_leak else "E_inh set to E_leak")
    abl["A10_static"] = {c: {s: float(np.mean(sim.scalar(cfg, s, f"ctrl_{c}"))) for s in SUBTYPES
                             if s in ctx["proj"]} for c in ("frozen", "gray", "shuffle", "reverse")}
    timing["ablations"] = time.time() - t1

    # geometry null, T4, speed axis
    t1 = time.time()
    speed_bank = {n: v for n, v in bank.items() if v[1].get("axis") == "speed"}
    obs = summarise(direction_metrics(sim, cfg, subtypes=T4, bank=speed_bank))["t4_mean"]
    rng = np.random.default_rng(21062026)
    ncol = len(ctx["colindex"])
    nulls = []
    for i in range(GEOMETRY_PERMUTATIONS):
        perm = {t: rng.permutation(ncol) for t in cm.ALL_PARTNERS}
        proj_s = {s: {t: m[:, perm[t]] for t, m in mats.items()} for s, mats in ctx["proj"].items()}
        nulls.append(summarise(direction_metrics(sim, cfg, subtypes=T4, bank=speed_bank, proj=proj_s))["t4_mean"])
        if (i + 1) % 50 == 0:
            print(f"   geometry permutation {i+1}/{GEOMETRY_PERMUTATIONS}", flush=True)
    nulls = np.array(nulls)
    abl["A6_geometry_shuffle"] = dict(
        permutations=int(len(nulls)), endpoint="T4 mean DSI, held-out speed axis",
        observed=float(obs), null_mean=float(nulls.mean()), null_sd=float(nulls.std()),
        null_95=[float(np.quantile(nulls, .025)), float(np.quantile(nulls, .975))],
        z=float((obs - nulls.mean()) / (nulls.std() + 1e-12)),
        p_empirical=float((1 + np.sum(nulls >= obs)) / (len(nulls) + 1)),
        residual_fraction=float(nulls.mean() / obs) if obs > 0 else None,
        null_samples=nulls.tolist())
    timing["geometry_permutations"] = time.time() - t1
    out["ablations"] = abl

    # generalisation and ON/OFF
    by_axis = {}
    for axis in ("speed", "width", "contrast", "position"):
        sb = {n: v for n, v in bank.items() if v[1].get("axis") == axis}
        if sb:
            by_axis[axis] = summarise(direction_metrics(sim, cfg, bank=sb))
    out["by_axis"] = by_axis
    onoff = {}
    for s in SUBTYPES:
        on = [np.mean(sim.scalar(cfg, s, n)) for n, (_q, m) in bank.items() if m.get("polarity") == "ON"]
        off = [np.mean(sim.scalar(cfg, s, n)) for n, (_q, m) in bank.items() if m.get("polarity") == "OFF"]
        onoff[s] = dict(on=float(np.mean(on)), off=float(np.mean(off)))
    out["on_off"] = onoff
    timing["total"] = time.time() - t0
    out["timing_seconds"] = timing

    arrays = {}
    for s, v in primary.items():
        arrays[f"e21_{s}_dsi"] = v["per_neuron_dsi"]; arrays[f"e21_{s}_effect"] = v["per_neuron_effect"]
    for s, v in om.items():
        arrays[f"oracle_{s}_dsi"] = v["per_neuron_dsi"]
    np.savez_compressed(RESULTS / "heldout-per-neuron.npz", **arrays)
    (RESULTS / "heldout-results.json").write_text(json.dumps(out, indent=2) + "\n")
    (RESULTS / "ablation-results.json").write_text(json.dumps(abl, indent=2) + "\n")
    print(f"heldout stage {time.time()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "heldout"])
    a = ap.parse_args()
    {"calibrate": stage_calibrate, "heldout": stage_heldout}[a.stage]()


if __name__ == "__main__":
    main()
