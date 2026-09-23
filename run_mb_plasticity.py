"""Experiment 26 Phases 3-6: dopamine-gated mushroom-body plasticity validation.

Stages (results/experiment-26/phase3-6-preregistration.md):
  visual      Precondition A: KCg-d drive through the frozen Pokemon visual pathway (P1)
  bridge      DAN amplitude sweep + baseline DAN activity (default amplitude, theta)
  maxeffect   Phase 6A: MBON01/MBON11 at baseline / zeroed / diagnostic 2x plastic edges
  calibrate   eta then tau on synthetic appetitive conditioning
  condition   Phase 6B: conditioning, controls, timing, dose, reset, per channel

No Pokemon reward event, gameplay, evolution or E25 code. The frozen FlyBrain matrix is only
read; learned changes live in PlasticState.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from flymon import mb_plasticity as mp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiment-26"
SEEDS = tuple(range(2701, 2707))
CS_RNG = 2601
SHUFFLE_RNG = 2602
AMP_GRID = (0.1, 0.2, 0.3, 0.5, 0.8)
ETA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
TAU_GRID = (5, 25, 100)
KC_DRIVE_V = 0.3
CS_SIZE = 40
DAN_WINDOW = 10
CONTROLLER_DN = ("DNa02", "DNg100", "MDN", "DNp01")


def jdump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float) + "\n")


class Harness:
    def __init__(self):
        from scipy import sparse
        from flybrain import FlyBrain
        import run_generation_zero as G
        from run_visual_experiment import DATA
        from flymon.fast_io import FastColumnSampler
        from flymon.frozen_t4 import T4Injection
        self.brain = FlyBrain(data=DATA, device="cuda")
        ct = self.brain.cell_type.astype(str); self.ct = ct
        meta = np.load(DATA / "brain.npz")
        W = sparse.load_npz(DATA / "weights.npz").tocsr()
        self.edges = mp.audited_edge_set(meta["cell_type"], meta["ids"], W)
        assert self.edges.sha256.startswith(mp.AUDITED_EDGE_HASH) and self.edges.n == 418
        self.w_hash = hashlib.sha256(W.data.tobytes()).hexdigest()
        self.pop = dict(KC=mp.population(ct, "KCg-d"), MBON01=mp.population(ct, "MBON01"),
                        MBON11=mp.population(ct, "MBON11"), PAM01=mp.population(ct, "PAM01"),
                        PPL101=mp.population(ct, "PPL101"),
                        DN=np.flatnonzero(np.isin(ct, CONTROLLER_DN)))
        self.dan = {"appetitive": self.pop["PAM01"], "aversive": self.pop["PPL101"]}
        self.slot = np.full(self.brain.n, -1, np.int64)
        self.groups = list(self.pop)
        for g, name in enumerate(self.groups):
            if name != "KC":
                self.slot[self.pop[name]] = g
        self.kc_slot = np.full(self.brain.n, -1, np.int64); self.kc_slot[self.pop["KC"]] = np.arange(len(self.pop["KC"]))
        self.pb = mp.PlasticBrain(self.brain, self.edges)
        self.t4 = G.sensory_setup("native")
        self.sampler = FastColumnSampler(self.t4._records, self.t4._colindex, self.t4._uv)
        self.inj = T4Injection(self.t4.brain_index, G.INJECTION_GAIN, G.INJECTION_CAP, G.INJECTION_LEVELS)
        rng = np.random.default_rng(CS_RNG)
        perm = rng.permutation(len(self.pop["KC"]))
        self.cs = {"plus": self.pop["KC"][perm[:CS_SIZE]], "minus": self.pop["KC"][perm[CS_SIZE:2 * CS_SIZE]]}
        self.cs_slots = {"plus": perm[:CS_SIZE], "minus": perm[CS_SIZE:2 * CS_SIZE]}

    # ---------------------------------------------------------------- one step with recording
    def step(self, pairs, delta, rec):
        fired = self.pb.step(self.pb.prepare(pairs), delta)
        s = self.slot[fired]; s = s[s >= 0]
        counts = np.bincount(s, minlength=len(self.groups))
        kc = self.kc_slot[fired]; kc = kc[kc >= 0]
        kcv = np.zeros(len(self.pop["KC"])); kcv[kc] = 1.0
        counts[self.groups.index("KC")] = len(kc)
        rec.append((counts, kcv))
        return counts, kcv

    # ---------------------------------------------------------------- visual trials
    def visual_trial(self, seed, frames_fn, decisions=60, inject=True):
        self.brain.reset(seed=seed); self.t4.reset()
        rec = []
        kc_total = np.zeros(len(self.pop["KC"])); grp = np.zeros(len(self.groups)); ninj = []
        for d in range(decisions):
            resp = [self.t4.update_from_luminance(self.sampler.luminance(f)) for f in frames_fn(d)]
            pairs = self.inj.pairs(np.mean(resp, axis=0)) if inject else ()
            ninj.append(sum(len(i) for i, _ in pairs))
            for _ in range(10):
                c, k = self.step(pairs, None, rec)
                grp += c; kc_total += k
        return dict(kc_per_cell=kc_total, groups={g: float(grp[i]) for i, g in enumerate(self.groups)},
                    injected_t4=float(np.mean(ninj)))

    # ---------------------------------------------------------------- scheduled synthetic trials
    def run_schedule(self, seed, blocks, state, learner=None, learning=False, theta=None, record_windows=False):
        """blocks: list of dict(n, cs in {None,'plus','minus','all'}, dan=(channel, amp, start, stop) or None,
        tag). Returns per-block spike counts by group (+ per-step KC activity if requested)."""
        self.brain.reset(seed=seed)
        if learner is not None:
            learner.reset_traces()
        out = []
        for b in blocks:
            counts = np.zeros(len(self.groups)); dan_steps = np.zeros(len(self.groups)); moved = 0.0
            kc_sum = np.zeros(len(self.pop["KC"])); gates = []
            for t in range(b["n"]):
                pairs = []
                if b.get("cs") == "all":
                    pairs.append((self.pop["KC"], KC_DRIVE_V))
                elif b.get("cs"):
                    pairs.append((self.cs[b["cs"]], KC_DRIVE_V))
                dan_on = b.get("dan") and b["dan"][2] <= t < b["dan"][3]
                if dan_on:
                    pairs.append((self.dan[b["dan"][0]], b["dan"][1]))
                rec = []
                c, kv = self.step(pairs, state.delta if state is not None else None, rec)
                counts += c; kc_sum += kv
                if dan_on:
                    dan_steps += c
                if learner is not None:
                    frac = {ch: c[self.groups.index("PAM01" if ch == "appetitive" else "PPL101")] / len(self.dan[ch])
                            for ch in ("appetitive", "aversive")}
                    D = learner.dan_signal(frac, theta or {})
                    gates.append(D)
                    moved += learner.step(kv, D, state, learning=learning)
            out.append(dict(tag=b.get("tag"), n=b["n"], counts={g: float(counts[i]) for i, g in enumerate(self.groups)},
                            during_dan={g: float(dan_steps[i]) for i, g in enumerate(self.groups)},
                            kc_active=float((kc_sum > 0).mean()), moved=moved,
                            gate_sum={ch: float(sum(g[ch] for g in gates)) for ch in ("appetitive", "aversive")} if gates else None))
        return out


# ================================================================ stages
def make_stimuli():
    import os
    from flymon.emulator import PokemonEmulator
    from flymon import gameplay_eval as ev
    H, W = 144, 160
    rgba = lambda g: np.concatenate([np.repeat((np.clip(g, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1),
                                     np.full((H, W, 1), 255, np.uint8)], -1)
    xx = np.arange(W)[None, :].repeat(H, 0); yy = np.arange(H)[:, None].repeat(W, 1)
    gray = rgba(np.full((H, W), 0.5)); vert = rgba(((xx // 8) % 2).astype(float)); hor = rgba(((yy // 8) % 2).astype(float))
    drift = [rgba((((xx + k) // 8) % 2).astype(float)) for k in range(16)]
    stim = {}
    with PokemonEmulator() as g:
        g.load_state(ROOT / "states" / "bedroom.state"); g.tick(1)
        bedroom = g.framebuffer()
        walk = []
        # scripted taps only to create the stimulus frames (never a controller)
        for b in ["right"] * 5 + ["up"] * 6:
            g.press(b)
            for _ in range(8):
                g.tick(1); walk.append(g.framebuffer())
            g.release(b)
            for _ in range(4):
                g.tick(1); walk.append(g.framebuffer())
        tel = ev.read_telemetry(g)
        house = g.framebuffer()
    stim_meta = dict(house_map=tel["map"], walk_frames=len(walk))
    stim = {"gray": lambda d: [gray] * 12, "vertical": lambda d: [vert] * 12, "horizontal": lambda d: [hor] * 12,
            "pokemon_bedroom": lambda d: [bedroom] * 12, "pokemon_house1f": lambda d: [house] * 12,
            "drifting_vertical": lambda d: [drift[(12 * d + k) % 16] for k in range(12)],
            "pokemon_walk": lambda d: [walk[(12 * d + k) % len(walk)] for k in range(12)]}
    return stim, stim_meta


def stage_visual(h):
    stim, meta = make_stimuli()
    t0 = time.perf_counter()
    res = {}
    for name in ["no_visual"] + list(stim):
        res[name] = []
        for s in SEEDS:
            fn = stim.get(name, stim["gray"])
            r = h.visual_trial(s, fn, inject=name != "no_visual")
            res[name].append(r)
        tot = [float(r["kc_per_cell"].sum()) for r in res[name]]
        print(f"[visual] {name:<18} KCg-d spikes {np.mean(tot):8.1f} ± {np.std(tot):6.1f}  "
              f"active {np.mean([np.mean(r['kc_per_cell'] > 0) for r in res[name]]):.2f}  "
              f"MBON01 {np.mean([r['groups']['MBON01'] for r in res[name]]):.1f} "
              f"MBON11 {np.mean([r['groups']['MBON11'] for r in res[name]]):.1f} "
              f"PAM01 {np.mean([r['groups']['PAM01'] for r in res[name]]):.1f} inj {res[name][0]['injected_t4']:.0f}", flush=True)
    # gate
    tot = {k: np.array([float(r["kc_per_cell"].sum()) for r in v]) for k, v in res.items()}
    kcL = h.pop["KC"][h.brain.side[h.pop["KC"]] == "L"]
    gate = {}
    for c in stim:
        if c == "gray":
            continue
        row = {}
        for ref in ("no_visual", "gray"):
            d = tot[c] - tot[ref]
            row[ref] = dict(mean_diff=float(d.mean()), same_sign=int(max((d > 0).sum(), (d < 0).sum())),
                            positive=int((d > 0).sum()), dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else
                            (float("inf") if d.mean() > 0 else 0.0))
        rel = (tot[c].mean() - tot["no_visual"].mean()) / max(tot["no_visual"].mean(), 1e-9)
        row["relative_increase_vs_no_visual"] = float(rel)
        row["passes"] = bool(all(row[r]["mean_diff"] > 0 and row[r]["positive"] >= 5 and row[r]["dz"] >= 0.8
                                 for r in ("no_visual", "gray")) and rel >= 0.20)
        gate[c] = row
    patterns = {}
    names = list(stim)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rs = []
            for ra, rb in zip(res[a], res[b]):
                x, y = ra["kc_per_cell"], rb["kc_per_cell"]
                rs.append(float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None)
            patterns[f"{a}~{b}"] = rs
    summary = {k: dict(total_mean=float(tot[k].mean()), total_sd=float(tot[k].std(ddof=1)),
                       cv=float(tot[k].std(ddof=1) / tot[k].mean()) if tot[k].mean() > 0 else None,
                       rate_hz_per_cell=float(tot[k].mean() / len(h.pop["KC"]) / 12.0),
                       fraction_active=float(np.mean([np.mean(r["kc_per_cell"] > 0) for r in res[k]])),
                       median_per_cell=float(np.mean([np.median(r["kc_per_cell"]) for r in res[k]])),
                       left=float(np.mean([r["kc_per_cell"][np.isin(h.pop["KC"], kcL)].sum() for r in res[k]])),
                       right=float(np.mean([r["kc_per_cell"][~np.isin(h.pop["KC"], kcL)].sum() for r in res[k]])),
                       per_seed=tot[k].tolist(),
                       other={g: float(np.mean([r["groups"][g] for r in res[k]])) for g in h.groups if g != "KC"},
                       injected_t4=float(np.mean([r["injected_t4"] for r in res[k]])))
               for k in res}
    p1 = any(v["passes"] for v in gate.values())
    jdump("visual-drive-results.json", dict(seeds=list(SEEDS), stimuli_meta=meta, summary=summary, gate=gate,
                                            pattern_correlations=patterns, P1=p1, seconds=time.perf_counter() - t0))
    print("P1", p1, {k: v["passes"] for k, v in gate.items()})


def stage_bridge(h):
    t0 = time.perf_counter()
    state = mp.PlasticState.zeros(h.edges)
    base = {"appetitive": [], "aversive": []}
    for s in SEEDS:
        h.brain.reset(seed=s); rec = []
        for _ in range(600):
            h.step((), None, rec)
        for ch, g in (("appetitive", "PAM01"), ("aversive", "PPL101")):
            gi = h.groups.index(g)
            base[ch] += [c[gi] / len(h.dan[ch]) for c, _ in rec]
    theta = {ch: float(np.quantile(v, 0.99)) for ch, v in base.items()}
    sweep = {}
    for ch in ("appetitive", "aversive"):
        g = "PAM01" if ch == "appetitive" else "PPL101"
        for a in AMP_GRID:
            fr, spk, mb, once_frac = [], [], [], []
            for s in SEEDS:
                r = h.run_schedule(s, [dict(n=50, tag="pre"), dict(n=DAN_WINDOW, dan=(ch, a, 0, DAN_WINDOW), tag="dan"),
                                       dict(n=50, tag="post")], state)
                fr.append(r[1]["counts"][g] / len(h.dan[ch]) / DAN_WINDOW)
                spk.append(r[1]["counts"][g]); mb.append(r[1]["counts"]["MBON01" if ch == "appetitive" else "MBON11"])
                # fraction of the population spiking at least once in the window (same seed, same schedule)
                h.brain.reset(seed=s); tally = np.zeros(h.brain.n, bool)
                for _ in range(50):
                    h.pb.step(None, None)
                for _ in range(DAN_WINDOW):
                    fired = h.pb.step(h.pb.prepare([(h.dan[ch], a)]), None); tally[fired] = True
                once_frac.append(float(tally[h.dan[ch]].mean()))
            sweep[f"{ch}|{a}"] = dict(channel=ch, amplitude=a, dan_spikes_mean=float(np.mean(spk)),
                                      per_step_fraction=float(np.mean(fr)), fraction_spiking_once=float(np.mean(once_frac)),
                                      target_mbon_spikes_during=float(np.mean(mb)))
            print(f"[bridge] {ch:<10} amp {a:.1f} DAN spikes {np.mean(spk):6.1f} once-frac {np.mean(once_frac):.2f} "
                  f"MBON {np.mean(mb):.1f}", flush=True)
    ok = [a for a in AMP_GRID if all(sweep[f"{ch}|{a}"]["fraction_spiking_once"] >= 0.5 for ch in ("appetitive", "aversive"))]
    default = min(ok) if ok else max(AMP_GRID)
    jdump("bridge-results.json", dict(theta=theta, baseline_fraction_p99=theta,
                                      baseline_mean={ch: float(np.mean(v)) for ch, v in base.items()},
                                      sweep=sweep, default_amplitude=default, rule="smallest grid amplitude with >= 50% of "
                                      "DANs spiking at least once in the 10-step window, both channels",
                                      seconds=time.perf_counter() - t0))
    print("theta", theta, "default amplitude", default)


def stage_maxeffect(h):
    t0 = time.perf_counter()
    conds = {"baseline": np.zeros(h.edges.n), "zeroed": -h.edges.w0.copy(), "double_diagnostic": h.edges.w0.copy()}
    out = {}
    for name, delta in conds.items():
        st = mp.PlasticState(h.edges.sha256, delta)
        rows = []
        for s in SEEDS:
            r = h.run_schedule(s, [dict(n=300, cs="all", tag="kc_drive"), dict(n=300, tag="spont")], st)
            rows.append(r)
        out[name] = dict(
            kc_drive={g: [r[0]["counts"][g] for r in rows] for g in ("KC", "MBON01", "MBON11", "DN")},
            spontaneous={g: [r[1]["counts"][g] for r in rows] for g in ("KC", "MBON01", "MBON11", "DN")},
            kc_active_fraction=float(np.mean([r[0]["kc_active"] for r in rows])))
    # KC counts are tallied separately (KC not in slot groups): recompute KC spikes from kc_active proxy
    res = {}
    for g in ("MBON01", "MBON11", "DN"):
        b = np.array(out["baseline"]["kc_drive"][g]); z = np.array(out["zeroed"]["kc_drive"][g])
        dd = np.array(out["double_diagnostic"]["kc_drive"][g])
        res[g] = dict(baseline=b.tolist(), zeroed=z.tolist(), double=dd.tolist(),
                      zeroed_rel_change=float((z.mean() - b.mean()) / b.mean()) if b.mean() else None,
                      double_rel_change=float((dd.mean() - b.mean()) / b.mean()) if b.mean() else None,
                      zeroed_lower_seeds=int((z < b).sum()))
    tb = np.array(out["baseline"]["kc_drive"]["MBON01"]) + np.array(out["baseline"]["kc_drive"]["MBON11"])
    tz = np.array(out["zeroed"]["kc_drive"]["MBON01"]) + np.array(out["zeroed"]["kc_drive"]["MBON11"])
    rel = float((tz.mean() - tb.mean()) / tb.mean()) if tb.mean() else 0.0
    gate = bool(rel <= -0.10 and (tz < tb).sum() >= 5)
    jdump("max-effect-results.json", dict(conditions=out, summary=res, target_rel_change_zeroed=rel,
                                          target_lower_seeds=int((tz < tb).sum()), gate_passes=gate,
                                          kc_drive_voltage=KC_DRIVE_V, seconds=time.perf_counter() - t0))
    print("max-effect", {g: (v["zeroed_rel_change"], v["double_rel_change"]) for g, v in res.items()}, "gate", gate)


def conditioning_blocks(channel, amp, variant="paired", timing=0, pairings=10):
    test = [dict(n=50, cs="plus", tag="test_plus"), dict(n=50, tag="rest"), dict(n=50, cs="minus", tag="test_minus"),
            dict(n=50, tag="rest")]
    train = []
    for _ in range(pairings):
        cs = None if variant == "dan_only" else "plus"
        dan = None if variant in ("cs_only",) else (channel, amp, 15, 15 + DAN_WINDOW)
        if variant == "unpaired" or timing > 0:
            off = 100 if variant == "unpaired" else timing
            train.append(dict(n=25, cs=cs, tag="train_cs"))
            train.append(dict(n=off + DAN_WINDOW + 5, dan=None if variant == "cs_only" else (channel, amp, off, off + DAN_WINDOW),
                              tag="train_dan"))
            train.append(dict(n=max(0, 50 - off), tag="rest") if off < 50 else dict(n=1, tag="rest"))
        else:
            train.append(dict(n=25, cs=cs, dan=dan, tag="train_cs"))
            train.append(dict(n=50, tag="rest"))
        train.append(dict(n=25, cs="minus", tag="train_minus")); train.append(dict(n=50, tag="rest"))
    return test, train


def run_conditioning(h, seed, channel, cfg, theta, amp, variant="paired", timing=0, reset_test=False):
    st = mp.PlasticState.zeros(h.edges)
    learner = mp.Learner(h.edges, cfg, h.pop["KC"], h.dan)
    test, train = conditioning_blocks(channel, amp, variant, timing)
    tgt = "MBON01" if channel == "appetitive" else "MBON11"
    pre = h.run_schedule(seed, test, st)
    tr = h.run_schedule(seed, train, st, learner, learning=True, theta=theta)
    post = h.run_schedule(seed, test, st)
    res = dict(pre=pre, post=post)
    disc = lambda r: r[0]["counts"][tgt] - r[2]["counts"][tgt]
    res["target"] = tgt
    res["pre_disc"], res["post_disc"] = disc(pre), disc(post)
    res["learned_effect"] = res["post_disc"] - res["pre_disc"]
    res["post_plus_change"] = post[0]["counts"][tgt] - pre[0]["counts"][tgt]
    rel = -st.delta / h.edges.w0
    plus_edges = np.isin(h.edges.pre_idx, h.cs["plus"]); minus_edges = np.isin(h.edges.pre_idx, h.cs["minus"])
    own = h.edges.compartment == channel
    res["weights"] = dict(mean_rel_depression_cs_plus_own=float(rel[plus_edges & own].mean()),
                          mean_rel_depression_cs_minus_own=float(rel[minus_edges & own].mean()),
                          mean_rel_depression_other_compartment=float(rel[~own].mean()),
                          changed_other_compartment=int((st.delta[~own] != 0).sum()),
                          changed_total=int((st.delta != 0).sum()),
                          frac_at_bound_cs_plus=float(np.mean(np.isclose(st.delta[plus_edges & own], -h.edges.w0[plus_edges & own]))),
                          within_bounds=bool(np.all(st.delta >= -h.edges.w0 - 1e-15) and np.all(st.delta <= (cfg.k_max - 1) * h.edges.w0 + 1e-15)),
                          finite=bool(np.all(np.isfinite(st.delta))),
                          histogram=np.histogram(rel[own], bins=10, range=(0, 1))[0].tolist())
    res["acute_target_during_dan"] = float(sum(b["during_dan"][tgt] for b in tr))
    res["dan_spikes_training"] = float(sum(b["during_dan"]["PAM01" if channel == "appetitive" else "PPL101"] for b in tr))
    res["gate_sum"] = float(sum((b["gate_sum"] or {}).get(channel, 0.0) for b in tr))
    if reset_test:
        post2 = h.run_schedule(seed + 1000, test, st)          # neural reset (new noise), weights kept
        res["after_neural_reset_disc"] = disc(post2)
        pre2 = h.run_schedule(seed + 1000, test, mp.PlasticState.zeros(h.edges))
        res["baseline_same_noise_disc"] = disc(pre2)
        st0 = mp.PlasticState.zeros(h.edges)
        post3 = h.run_schedule(seed, test, st0)                # learned weights reset
        res["after_weight_reset_disc"] = disc(post3)
    res["delta"] = st.delta.tolist()
    return res, st


def load_bridge():
    b = json.loads((OUT / "bridge-results.json").read_text())
    return b["theta"], b["default_amplitude"]


def base_config(theta, eta, tau, **kw):
    return mp.PlasticityConfig(eta=eta, tau_elig_steps=tau, theta_appetitive=theta["appetitive"],
                               theta_aversive=theta["aversive"], **kw)


def stage_calibrate(h):
    t0 = time.perf_counter()
    theta, amp = load_bridge()
    th = {"appetitive": theta["appetitive"], "aversive": theta["aversive"]}
    rows, chosen_eta = {}, None
    for eta in ETA_GRID:
        r = [run_conditioning(h, s, "appetitive", base_config(th, eta, 25), th, amp)[0] for s in SEEDS]
        d = float(np.mean([x["weights"]["mean_rel_depression_cs_plus_own"] for x in r]))
        sat = float(np.mean([x["weights"]["frac_at_bound_cs_plus"] for x in r]))
        fin = all(x["weights"]["finite"] for x in r)
        rows[f"eta={eta}"] = dict(mean_rel_depression=d, saturated_fraction=sat, finite=fin,
                                  learned_effect=[x["learned_effect"] for x in r])
        print(f"[calibrate] eta {eta:<5} rel depression {d:.3f} saturated {sat:.2f}", flush=True)
        if chosen_eta is None and 0.10 <= d <= 0.80 and sat <= 0.5 and fin:
            chosen_eta = eta
    if chosen_eta is None:
        chosen_eta = ETA_GRID[-1]
    trow, chosen_tau = {}, None
    for tau in TAU_GRID:
        r = [run_conditioning(h, s, "appetitive", base_config(th, chosen_eta, tau), th, amp)[0] for s in SEEDS]
        d = float(np.mean([x["weights"]["mean_rel_depression_cs_plus_own"] for x in r]))
        sat = float(np.mean([x["weights"]["frac_at_bound_cs_plus"] for x in r]))
        trow[f"tau={tau}"] = dict(mean_rel_depression=d, saturated_fraction=sat)
        print(f"[calibrate] tau {tau:<4} rel depression {d:.3f} saturated {sat:.2f}", flush=True)
    ok = [t for t in TAU_GRID if 0.10 <= trow[f"tau={t}"]["mean_rel_depression"] <= 0.80
          and trow[f"tau={t}"]["saturated_fraction"] <= 0.5]
    chosen_tau = 25 if 25 in ok or not ok else ok[0]
    cfg = base_config(th, chosen_eta, chosen_tau)
    jdump("plasticity-config.json", dict(frozen=cfg.to_json(), dan_amplitude=amp, dan_window_steps=DAN_WINDOW,
                                         eta_grid=rows, tau_grid=trow, selection_rule="see phase3-6-preregistration.md",
                                         edge_hash=h.edges.sha256, frozen_w_sha256=h.w_hash,
                                         seconds=time.perf_counter() - t0))
    print("frozen", cfg)


def stage_condition(h):
    t0 = time.perf_counter()
    pc = json.loads((OUT / "plasticity-config.json").read_text())
    cfg = mp.PlasticityConfig(**pc["frozen"]); amp = pc["dan_amplitude"]
    th = {"appetitive": cfg.theta_appetitive, "aversive": cfg.theta_aversive}
    variants = dict(paired=dict(), dan_only=dict(variant="dan_only"), cs_only=dict(variant="cs_only"),
                    unpaired=dict(variant="unpaired"))
    results, controls, timing, dose = {}, {}, {}, {}
    def summarize(rs):
        eff = np.array([r["learned_effect"] for r in rs])
        return dict(learned_effect=eff.tolist(), mean=float(eff.mean()), sd=float(eff.std(ddof=1)),
                    dz=float(eff.mean() / eff.std(ddof=1)) if eff.std(ddof=1) > 0 else None,
                    negative_seeds=int((eff < 0).sum()),
                    rel_depression_cs_plus=[r["weights"]["mean_rel_depression_cs_plus_own"] for r in rs],
                    rel_depression_cs_minus=[r["weights"]["mean_rel_depression_cs_minus_own"] for r in rs],
                    changed_other_compartment=[r["weights"]["changed_other_compartment"] for r in rs],
                    within_bounds=all(r["weights"]["within_bounds"] for r in rs),
                    finite=all(r["weights"]["finite"] for r in rs),
                    acute_target_during_dan=[r["acute_target_during_dan"] for r in rs],
                    dan_spikes=[r["dan_spikes_training"] for r in rs],
                    pre_disc=[r["pre_disc"] for r in rs], post_disc=[r["post_disc"] for r in rs],
                    post_plus_change=[r["post_plus_change"] for r in rs],
                    weight_histogram=np.sum([r["weights"]["histogram"] for r in rs], axis=0).tolist())
    for ch in ("appetitive", "aversive"):
        # A paired (with reset tests), B-D controls
        paired = []
        for s in SEEDS:
            r, st = run_conditioning(h, s, ch, cfg, th, amp, reset_test=True)
            paired.append(r)
            if s == SEEDS[0]:
                st.save(OUT / f"plastic-state-{ch}-seed{s}.json", cfg, dict(edge_hash=h.edges.sha256, frozen_w_sha256=h.w_hash,
                                                                          channel=ch, seed=s, protocol="paired"))
        results[ch] = summarize(paired)
        results[ch]["reset"] = dict(before=[r["learned_effect"] for r in paired],
                                    after_neural_reset=[r["after_neural_reset_disc"] - r["baseline_same_noise_disc"] for r in paired],
                                    after_weight_reset=[r["after_weight_reset_disc"] - r["pre_disc"] for r in paired])
        results[ch]["example_delta"] = paired[0]["delta"]
        print(f"[{ch}] paired effect {results[ch]['mean']:+.2f} (neg {results[ch]['negative_seeds']}/6) "
              f"rel dep CS+ {np.mean(results[ch]['rel_depression_cs_plus']):.3f} CS- {np.mean(results[ch]['rel_depression_cs_minus']):.3f}", flush=True)
        controls[ch] = {"paired": {k: results[ch][k] for k in ("mean", "negative_seeds", "rel_depression_cs_plus", "changed_other_compartment")}}
        for name, kw in list(variants.items())[1:]:
            rs = [run_conditioning(h, s, ch, cfg, th, amp, **kw)[0] for s in SEEDS]
            controls[ch][name] = summarize(rs)
            print(f"[{ch}] {name:<9} effect {controls[ch][name]['mean']:+.2f} rel dep CS+ "
                  f"{np.mean(controls[ch][name]['rel_depression_cs_plus']):.4f}", flush=True)
        off = mp.PlasticityConfig(**{**cfg.to_json(), "eta": 0.0})
        controls[ch]["plasticity_off"] = summarize([run_conditioning(h, s, ch, off, th, amp)[0] for s in SEEDS])
        shuf = mp.PlasticityConfig(**{**cfg.to_json(), "shuffle_eligibility_seed": SHUFFLE_RNG})
        controls[ch]["shuffled_eligibility"] = summarize([run_conditioning(h, s, ch, shuf, th, amp)[0] for s in SEEDS])
        cross = mp.PlasticityConfig(**{**cfg.to_json(), "cross_compartment": True})
        controls[ch]["cross_compartment_diagnostic"] = summarize([run_conditioning(h, s, ch, cross, th, amp)[0] for s in SEEDS])
        for k in ("plasticity_off", "shuffled_eligibility", "cross_compartment_diagnostic"):
            print(f"[{ch}] {k:<28} effect {controls[ch][k]['mean']:+.2f} rel dep CS+ "
                  f"{np.mean(controls[ch][k]['rel_depression_cs_plus']):.4f} other-comp changed "
                  f"{np.mean(controls[ch][k]['changed_other_compartment']):.0f}", flush=True)
        timing[ch] = {}
        for off_steps in (0, 5, 25, 100, 250):
            rs = [run_conditioning(h, s, ch, cfg, th, amp, timing=off_steps)[0] for s in SEEDS] if off_steps else paired
            timing[ch][off_steps] = dict(rel_depression_cs_plus=float(np.mean([r["weights"]["mean_rel_depression_cs_plus_own"] for r in rs])),
                                         learned_effect=float(np.mean([r["learned_effect"] for r in rs])))
            print(f"[{ch}] timing +{off_steps:<3} rel dep {timing[ch][off_steps]['rel_depression_cs_plus']:.4f} "
                  f"effect {timing[ch][off_steps]['learned_effect']:+.2f}", flush=True)
        dose[ch] = {}
        for a in AMP_GRID:
            rs = [run_conditioning(h, s, ch, cfg, th, a)[0] for s in SEEDS]
            dose[ch][a] = dict(dan_spikes=float(np.mean([r["dan_spikes_training"] for r in rs])),
                               rel_depression_cs_plus=float(np.mean([r["weights"]["mean_rel_depression_cs_plus_own"] for r in rs])),
                               learned_effect=float(np.mean([r["learned_effect"] for r in rs])),
                               acute_target=float(np.mean([r["acute_target_during_dan"] for r in rs])))
            print(f"[{ch}] dose {a} DAN {dose[ch][a]['dan_spikes']:.0f} rel dep {dose[ch][a]['rel_depression_cs_plus']:.4f} "
                  f"effect {dose[ch][a]['learned_effect']:+.2f}", flush=True)
    jdump("conditioning-results.json", dict(config=cfg.to_json(), amplitude=amp, results=results, seeds=list(SEEDS),
                                            frozen_w_sha256_after=hashlib.sha256(
                                                __import__("scipy.sparse", fromlist=["x"]).load_npz(
                                                    ROOT / "redfly-benchmark/data/weights.npz").tocsr().data.tobytes()).hexdigest(),
                                            seconds=time.perf_counter() - t0))
    jdump("controls-results.json", controls)
    jdump("timing-results.json", timing)
    jdump("dose-response.json", dose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["visual", "bridge", "maxeffect", "calibrate", "condition"])
    a = ap.parse_args()
    h = Harness()
    {"visual": stage_visual, "bridge": stage_bridge, "maxeffect": stage_maxeffect,
     "calibrate": stage_calibrate, "condition": stage_condition}[a.stage](h)


if __name__ == "__main__":
    main()
