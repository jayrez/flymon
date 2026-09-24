"""Experiment 29 dynamic stages: KC visual screen, frozen candidate ranking, dynamic leverage, confirmation.

Stages (results/experiment-29-broader-mb-circuit-audit/preregistration.md):
  visual     all-KC visual screen at the 60 ms + APL x8 probe regime, seeds 3001-3006
  rank       apply the frozen visual gate + static audit -> hash-frozen candidate ranking (no simulation)
  leverage   top <= 10 candidates + E26 references, baseline vs zeroed, at 60/8 and 80/3, seeds 3011-3016
  confirm    <= 3 viable candidates on seeds 3021-3030 at exactly 60/8 and 80/3 (confirmation + sensitivity)

Reuses the E27/E28 harness (frozen T4 injection, stimuli, OperatingPointBrain) unchanged. Candidate edges
are zeroed through the non-mutating plastic-delta current; FlyBrain's _W is only read. No learning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

import run_mb_operating_regime as E27
from flymon import mb_plasticity as mp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiment-29-broader-mb-circuit-audit"
VISUAL_SEEDS = tuple(range(3001, 3007))
LEVERAGE_SEEDS = tuple(range(3011, 3017))
CONF_SEEDS = tuple(range(3021, 3031))
REGIMES = {"60/8": E27.OperatingPoint(refractory_ms=60.0, apl_gain=8.0),
           "80/3": E27.OperatingPoint(refractory_ms=80.0, apl_gain=3.0)}
PROBE = "60/8"
STIMULI = ("no_visual", "gray", "vertical", "horizontal", "drifting_vertical", "pokemon_bedroom",
           "pokemon_bedroom_after_walk", "pokemon_walk")
MIN_CLASS_CELLS, MIN_REL, MIN_CONS_FRAC = 10, 0.05, 5 / 6
LEV_MAX_REL = -0.10
DRIVE_V, DRIVE_STEPS = 0.3, 300
DAN_AMP, DAN_WARMUP, DAN_BASE, DAN_WIN, DAN_MIN_PP = 0.3, 200, 100, 10, 10.0
MAX_DYNAMIC, MAX_CONFIRM = 10, 3
REFERENCES = (("KCg-d", "MBON01", "PAM01"), ("KCg-d", "MBON11", "PPL101"))


def min_cons(n_seeds):
    return math.ceil(n_seeds * MIN_CONS_FRAC)


def jdump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def jload(name):
    return json.loads((OUT / name).read_text())


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- candidate edge sets (from frozen W)
def candidate_edges(ct, ids, W, kc_type, mbon_type):
    kc = np.flatnonzero(ct == kc_type); mb = np.flatnonzero(ct == mbon_type)
    sub = W[mb][:, kc].tocoo()
    rows = sorted((int(ids[kc[c]]), int(ids[mb[r]]), float(v), int(kc[c]), int(mb[r])) for r, c, v in zip(sub.row, sub.col, sub.data))
    h = hashlib.sha256(json.dumps([(a, b) for a, b, *_ in rows]).encode()).hexdigest()
    ro = lambda a, dt: np.asarray(a, dt)
    es = mp.EdgeSet(pre_idx=ro([r[3] for r in rows], np.int64), post_idx=ro([r[4] for r in rows], np.int64),
                    pre_body=ro([r[0] for r in rows], np.int64), post_body=ro([r[1] for r in rows], np.int64),
                    w0=ro([r[2] for r in rows], np.float64),
                    compartment=ro([f"{kc_type}->{mbon_type}"] * len(rows), object), sha256=h)
    for a in (es.pre_idx, es.post_idx, es.pre_body, es.post_body, es.w0, es.compartment):
        a.setflags(write=False)
    return es


class Runner:
    def __init__(self, regime):
        from scipy import sparse
        self.regime = regime; self.op = REGIMES[regime]
        self.h = E27.make_harness(self.op)
        self.W = sparse.load_npz(ROOT / "redfly-benchmark" / "data" / "weights.npz").tocsr()
        self.ids = np.load(ROOT / "redfly-benchmark" / "data" / "brain.npz")["ids"].astype(np.int64)
        self.ct = self.h.ct
        self.kc_types = sorted(set(self.ct[np.char.startswith(self.ct, "KC")]))
        self.w_hash = hashlib.sha256(self.W.data.tobytes()).hexdigest()

    def w_hash_now(self):
        return hashlib.sha256(self.h.brain._W.data.get().tobytes()).hexdigest()

    # ------------------------------------------------------------ visual (all KCs)
    def visual_trial(self, seed, frames_fn, inject, decisions=60, pb=None, delta=None):
        h = self.h; pb = pb or h.pb
        h.brain.reset(seed=seed); h.t4.reset()
        counts = np.zeros(h.brain.n, np.int32)
        for d in range(decisions):
            resp = [h.t4.update_from_luminance(h.sampler.luminance(f)) for f in frames_fn(d)]
            pairs = h.inj.pairs(np.mean(resp, axis=0)) if inject else ()
            for _ in range(10):
                counts[pb.step(pb.prepare(list(pairs)), delta)] += 1
        return counts

    def natural_drive(self, kc_type, mbon_type, dan_type, stimulus, seeds):
        """Descriptive (not preregistered, never used for selection): target MBON / DAN under Pokemon-driven
        activity alone, with the candidate's existing edges intact vs zeroed (no artificial KC drive)."""
        stim, _ = E27.E26.make_stimuli()
        stim = {E27.STIM_LABELS.get(k, k): v for k, v in stim.items()}
        es = candidate_edges(self.ct, self.ids, self.W, kc_type, mbon_type)
        pb = E27.OperatingPointBrain(self.h.brain, es, self.op, self.ct, self.W)
        mb = np.flatnonzero(self.ct == mbon_type); da = np.flatnonzero(self.ct == dan_type) if dan_type else mb[:0]
        out = {}
        for cond, name, delta in (("no_visual", "no_visual", None), ("stimulus", stimulus, None),
                                  ("stimulus_zeroed", stimulus, -es.w0.copy())):
            fn = stim.get(name, stim["gray"])
            per = [self.visual_trial(s, fn, inject=name != "no_visual", pb=pb, delta=delta) for s in seeds]
            out[cond] = dict(mbon=[int(c[mb].sum()) for c in per], dan=[int(c[da].sum()) for c in per])
        m = {k: np.mean(v["mbon"]) for k, v in out.items()}
        rel = lambda a, b: float((a - b) / b) if b > 0 else None
        return dict(kc_type=kc_type, mbon_type=mbon_type, dan_type=dan_type, stimulus=stimulus, seeds=list(seeds),
                    counts=out, mbon_visual_rel_vs_no_visual=rel(m["stimulus"], m["no_visual"]),
                    mbon_zeroed_rel_under_visual_drive=rel(m["stimulus_zeroed"], m["stimulus"]),
                    zeroed_lower_seeds=int(sum(z < b for z, b in zip(out["stimulus_zeroed"]["mbon"], out["stimulus"]["mbon"]))),
                    dan_visual_rel_vs_no_visual=rel(np.mean(out["stimulus"]["dan"]), np.mean(out["no_visual"]["dan"])),
                    note="descriptive diagnostic added after the leverage stage; not preregistered; not used for any decision")

    def visual_screen(self, seeds):
        stim, meta = E27.E26.make_stimuli()
        stim = {E27.STIM_LABELS.get(k, k): v for k, v in stim.items()}
        assert set(STIMULI) - {"no_visual"} <= set(stim)
        kc_idx = {t: np.flatnonzero(self.ct == t) for t in self.kc_types}
        cells = {t: {} for t in self.kc_types}
        for name in STIMULI:
            fn = stim.get(name, stim["gray"])
            per = [self.visual_trial(s, fn, inject=name != "no_visual") for s in seeds]
            for t in self.kc_types:
                cells[t][name] = np.stack([c[kc_idx[t]] for c in per]).astype(np.float64)
            print(f"[visual {self.regime}] {name:<28} all-KC spikes {np.mean([c[np.concatenate(list(kc_idx.values()))].sum() for c in per]):.0f}",
                  flush=True)
        return cells, kc_idx, meta

    # ------------------------------------------------------------ leverage (one candidate)
    def leverage(self, kc_type, mbon_type, seeds):
        h = self.h
        es = candidate_edges(self.ct, self.ids, self.W, kc_type, mbon_type)
        pb = E27.OperatingPointBrain(h.brain, es, self.op, self.ct, self.W)
        kc = np.flatnonzero(self.ct == kc_type); mb = np.flatnonzero(self.ct == mbon_type)
        mmask = np.zeros(h.brain.n, bool); mmask[mb] = True
        kmask = np.zeros(h.brain.n, bool); kmask[kc] = True
        out = {}
        for cond, delta in (("baseline", None), ("zeroed", -es.w0.copy())):
            m_counts, k_counts = [], []
            for s in seeds:
                h.brain.reset(seed=s); mc = kc_ = 0
                for _ in range(DRIVE_STEPS):
                    f = pb.step(pb.prepare([(kc, DRIVE_V)]), delta)
                    mc += int(mmask[f].sum()); kc_ += int(kmask[f].sum())
                m_counts.append(mc); k_counts.append(kc_)
            out[cond] = dict(mbon=m_counts, kc=k_counts)
        b, z = np.array(out["baseline"]["mbon"], float), np.array(out["zeroed"]["mbon"], float)
        rel = float((z.mean() - b.mean()) / b.mean()) if b.mean() > 0 else None
        lower = int((z < b).sum())
        return dict(kc_type=kc_type, mbon_type=mbon_type, regime=self.regime, seeds=list(seeds), edges=es.n,
                    edge_sha256=es.sha256, mbon_cells=int(len(mb)), kc_cells=int(len(kc)),
                    baseline_mbon=b.tolist(), zeroed_mbon=z.tolist(), per_seed_delta=(z - b).tolist(),
                    baseline_mbon_rate_hz=float(b.mean() / len(mb) / (DRIVE_STEPS * float(h.brain.dt))),
                    zeroed_mbon_rate_hz=float(z.mean() / len(mb) / (DRIVE_STEPS * float(h.brain.dt))),
                    abs_spike_difference=float(z.mean() - b.mean()), rel_change=rel, lower_seeds=lower,
                    kc_spikes_baseline=out["baseline"]["kc"], kc_spikes_zeroed=out["zeroed"]["kc"],
                    passes=bool(rel is not None and rel <= LEV_MAX_REL and lower >= min_cons(len(seeds))),
                    w_unchanged=self.w_hash_now() == self.w_hash)

    # ------------------------------------------------------------ fixed-amplitude DAN (E28 assay, any DAN type)
    def dan(self, dan_type, seeds):
        h = self.h
        di = np.flatnonzero(self.ct == dan_type); dmask = np.zeros(h.brain.n, bool); dmask[di] = True
        base, stim = [], []
        for s in seeds:
            h.brain.reset(seed=s)
            for _ in range(DAN_WARMUP):
                h.pb.step(None, None)
            bc = sum(int(dmask[h.pb.step(None, None)].sum()) for _ in range(DAN_BASE))
            sc = sum(int(dmask[h.pb.step(h.pb.prepare([(di, DAN_AMP)]), None)].sum()) for _ in range(DAN_WIN))
            base.append(bc / len(di) / DAN_BASE); stim.append(sc / len(di) / DAN_WIN)
        base, stim = np.array(base), np.array(stim); d = stim - base
        return dict(dan_type=dan_type, cells=int(len(di)), amplitude=DAN_AMP, baseline_duty=float(base.mean()),
                    baseline_rate_hz=float(base.mean() / float(h.brain.dt)), stimulated_duty=float(stim.mean()),
                    delta_pp=float(100 * d.mean()), per_seed_delta_pp=(100 * d).tolist(), positive_seeds=int((d > 0).sum()),
                    usable=bool(100 * d.mean() >= DAN_MIN_PP and (d > 0).sum() >= min_cons(len(seeds))))


# ---------------------------------------------------------------- visual metrics / gate
def contrast(tot, c, ref):
    d = tot[c] - tot[ref]; sd = d.std(ddof=1) if len(d) > 1 else 0.0
    m = float(tot[ref].mean())
    return dict(mean_diff=float(d.mean()), rel_diff=float(d.mean() / m) if m > 0 else None,
                consistent_seeds=int(max((d > 0).sum(), (d < 0).sum())), positive_seeds=int((d > 0).sum()),
                dz=float(d.mean() / sd) if sd > 0 else None, per_seed_diff=d.tolist())


def gate(n_cells, nv, gr, n_seeds):
    ok = (n_cells >= MIN_CLASS_CELLS and nv["rel_diff"] is not None and gr["rel_diff"] is not None
          and abs(nv["rel_diff"]) >= MIN_REL and nv["consistent_seeds"] >= min_cons(n_seeds)
          and abs(gr["rel_diff"]) >= MIN_REL and np.sign(nv["mean_diff"]) == np.sign(gr["mean_diff"]) != 0)
    return bool(ok)


def class_metrics(cls_cells, body_ids, n_seeds, dt, reliability=True, rng_seed=29):
    tot = {k: v.sum(1) for k, v in cls_cells.items()}
    n = cls_cells["no_visual"].shape[1]; steps = 600
    rows = {}
    for c in STIMULI[2:]:
        nv, gr = contrast(tot, c, "no_visual"), contrast(tot, c, "gray")
        dc = cls_cells[c] - cls_cells["no_visual"]; base = cls_cells["no_visual"].mean(0)
        sig = np.sign(dc); cons = np.maximum((sig > 0).sum(0), (sig < 0).sum(0)) >= min_cons(n_seeds)
        mod = np.abs(dc.mean(0)) / np.maximum(base, 1.0)
        top = np.argsort(-mod)[:5]
        rows[c] = dict(no_visual=nv, gray=gr, passes_gate=gate(n, nv, gr, n_seeds),
                       fraction_cells_modulated=float(np.mean(cons & (mod >= 0.2))),
                       median_cell_modulation=float(np.median(mod)), q75_cell_modulation=float(np.quantile(mod, .75)),
                       top_cells=[dict(body_id=int(body_ids[i]), modulation=float(mod[i]), mean_diff=float(dc.mean(0)[i]))
                                  for i in top],
                       fraction_active=float(np.mean(cls_cells[c] > 0)))
    passing = [c for c in rows if rows[c]["passes_gate"]]
    pool = passing or list(rows)
    best = max(pool, key=lambda c: abs(rows[c]["no_visual"]["rel_diff"] or 0))
    out = dict(cells=n, baseline_rate_hz=float(tot["no_visual"].mean() / n / (steps * dt)),
               gray_rate_hz=float(tot["gray"].mean() / n / (steps * dt)),
               fraction_active_no_visual=float(np.mean(cls_cells["no_visual"] > 0)),
               best_stimulus=best, passes_gate=bool(passing), passing_stimuli=passing, stimuli=rows)
    if reliability and passing:
        rng = np.random.default_rng(rng_seed); b = best
        boots = []
        for _ in range(10000):
            s = rng.integers(0, n_seeds, n_seeds)
            m0 = tot["no_visual"][s].mean()
            boots.append((tot[b][s].mean() - m0) / m0 if m0 > 0 else np.nan)
        sub_pass = []
        for _ in range(200):
            k = rng.choice(n, max(1, n // 2), replace=False)
            t2 = {x: cls_cells[x][:, k].sum(1) for x in ("no_visual", "gray", b)}
            sub_pass.append(gate(n, contrast(t2, b, "no_visual"), contrast(t2, b, "gray"), n_seeds))
        dmean = np.abs((cls_cells[b] - cls_cells["no_visual"]).mean(0))
        k10 = max(1, int(round(0.1 * n)))
        out["reliability"] = dict(stimulus=b, bootstrap_rel_ci95=[float(np.nanquantile(boots, .025)), float(np.nanquantile(boots, .975))],
                                  per_seed_diff=rows[b]["no_visual"]["per_seed_diff"],
                                  half_subsample_gate_pass_fraction=float(np.mean(sub_pass)),
                                  top10pct_cells_share_of_abs_diff=float(np.sort(dmean)[::-1][:k10].sum() / dmean.sum()) if dmean.sum() else None,
                                  pattern=None)
        out["reliability"]["pattern"] = ("broad" if out["reliability"]["top10pct_cells_share_of_abs_diff"] is not None
                                         and out["reliability"]["top10pct_cells_share_of_abs_diff"] < 0.5 else "subset-dominated")
    return out


def visual_report(R, seeds):
    cells, kc_idx, meta = R.visual_screen(seeds)
    dt = float(R.h.brain.dt)
    classes = {t: class_metrics(cells[t], R.ids[kc_idx[t]], len(seeds), dt) for t in R.kc_types}
    return dict(regime=R.regime, operating_point=R.op.__dict__, seeds=list(seeds), stimuli=list(STIMULI), stimuli_meta=meta,
                gate_rule=dict(min_class_cells=MIN_CLASS_CELLS, min_abs_rel=MIN_REL, min_consistent_seeds=min_cons(len(seeds)),
                               same_sign_both_references=True),
                classes=classes, passing_classes=[t for t in R.kc_types if classes[t]["passes_gate"]],
                t4_sha256=R.h.t4.sha256(), w_sha256=R.w_hash, w_unchanged=R.w_hash_now() == R.w_hash)


# ---------------------------------------------------------------- stages
def stage_visual():
    t0 = time.perf_counter()
    R = Runner(PROBE)
    rep = visual_report(R, VISUAL_SEEDS)
    rep["seconds"] = time.perf_counter() - t0
    jdump("kc-visual-response.json", rep)
    for t, c in rep["classes"].items():
        s = c["stimuli"][c["best_stimulus"]]
        print(f"{t:<12} n={c['cells']:<5} base {c['baseline_rate_hz']:.2f}Hz best {c['best_stimulus']:<27} "
              f"nv {100 * (s['no_visual']['rel_diff'] or 0):+7.2f}% ({s['no_visual']['consistent_seeds']}/6) "
              f"gray {100 * (s['gray']['rel_diff'] or 0):+7.2f}% mod {s['fraction_cells_modulated']:.2f} gate {c['passes_gate']}")


def build_ranking():
    vis = jload("kc-visual-response.json"); lev = jload("static-leverage-screen.json")["pairs"]
    conn = jload("kc-mbon-connectivity.json")["pairs"]; dan = jload("dan-compartment-audit.json")["pairs"]
    rows = []
    for key, s in lev.items():
        k, m = s["kc_type"], s["mbon_type"]
        c = conn[key]; d = dan[key]
        rows.append(dict(key=key, kc_type=k, mbon_type=m, dan_type=d["primary_dan"],
                         visual_gate=bool(vis["classes"][k]["passes_gate"]), dan_match=d["primary_dan"] is not None,
                         static_input_fraction=s["pooled"], static_input_median=s["median"],
                         kc_mbon_abs_weight=c["abs_weight_sum"], edges=c["edges"], source_kc_cells=c["source_kc_cells"],
                         kc_classes=1, mbon_classes=1))
    key = lambda r: (not r["visual_gate"], not r["dan_match"], -r["static_input_fraction"], -r["kc_mbon_abs_weight"],
                     -r["source_kc_cells"], r["key"])
    rows.sort(key=key)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    eligible = [r for r in rows if r["visual_gate"] and r["dan_match"]]
    return rows, eligible


def stage_rank():
    assert not (OUT / "dynamic-leverage-results.json").exists(), "ranking must be frozen before any leverage result"
    rows, eligible = build_ranking()
    top = eligible[:MAX_DYNAMIC]
    refs = []
    for k, m, d in REFERENCES:
        r = next(x for x in rows if x["key"] == f"{k}->{m}")
        refs.append(dict(r, reference_dan=d, in_top10=any(t["key"] == r["key"] for t in top)))
    lines = ["# Experiment 29 — Candidate ranking (preregistered, hash-frozen before any leverage simulation)", "",
             "Inputs: `kc-visual-response.json` (frozen visual gate, seeds 3001–3006, 60/8), `static-leverage-screen.json`,",
             "`kc-mbon-connectivity.json`, `dan-compartment-audit.json`. No dynamic leverage result exists at freeze time.", "",
             "Rule (from `preregistration.md`): eligible = KC class passes the visual gate AND >= 1 KC->MBON edge AND a",
             "topology-matched DAN exists. Order: visual gate, matched DAN, pooled static MBON input fraction (desc),",
             "KC->MBON |W| (desc), participating KC cells (desc), key. All units are single KC type -> single MBON type",
             "(complexity, #KC classes, #MBON classes tie). Top 10 eligible are tested dynamically; the E26 rows",
             "KCg-d->MBON01 (PAM01) and KCg-d->MBON11 (PPL101) are always run as labelled references.", "",
             f"Visually passing KC classes: {', '.join(t for t, c in jload('kc-visual-response.json')['classes'].items() if c['passes_gate']) or 'none'}",
             f"Eligible units: {len(eligible)}; dynamically tested: {len(top)}", "",
             "| rank | KC -> MBON | DAN (topology) | static input % (pooled / median) | KC->MBON abs W | edges | KC cells | tested |",
             "|---|---|---|---|---|---|---|---|"]
    for r in eligible[:30]:
        lines.append(f"| {r['rank']} | {r['key']} | {r['dan_type']} | {100 * r['static_input_fraction']:.2f} / "
                     f"{100 * r['static_input_median']:.2f} | {r['kc_mbon_abs_weight']:.4f} | {r['edges']} | {r['source_kc_cells']} | "
                     f"{'yes' if r in top else 'no'} |")
    lines += ["", "References (always run):", ""]
    for r in refs:
        lines.append(f"- {r['key']} (reference DAN {r['reference_dan']}; topology DAN {r['dan_type']}): static "
                     f"{100 * r['static_input_fraction']:.2f} %, overall rank {r['rank']}, eligible "
                     f"{r['visual_gate'] and r['dan_match']}, in top 10: {r['in_top10']}")
    md = "\n".join(lines) + "\n"
    (OUT / "candidate-ranking-preregistration.md").write_text(md)
    jdump("dynamic-candidate-list.json", dict(cap=MAX_DYNAMIC, candidates=top, references=refs, seeds=list(LEVERAGE_SEEDS),
                                              regimes=list(REGIMES)))
    jdump("candidate-ranking-freeze.json", dict(
        preregistration_sha256=sha_file(OUT / "candidate-ranking-preregistration.md"),
        dynamic_candidate_list_sha256=sha_file(OUT / "dynamic-candidate-list.json"),
        ranked_keys=[r["key"] for r in rows], eligible_keys=[r["key"] for r in eligible],
        tested_keys=[r["key"] for r in top], full_ranking=rows,
        frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), leverage_results_existed=False))
    print(md)


def check_freeze():
    fz = jload("candidate-ranking-freeze.json")
    assert sha_file(OUT / "candidate-ranking-preregistration.md") == fz["preregistration_sha256"]
    assert sha_file(OUT / "dynamic-candidate-list.json") == fz["dynamic_candidate_list_sha256"]
    return fz


def stage_leverage():
    t0 = time.perf_counter()
    fz = check_freeze(); dl = jload("dynamic-candidate-list.json")
    units = [dict(c, role="candidate") for c in dl["candidates"]]
    units += [dict(r, role="reference") for r in dl["references"] if not r["in_top10"]]
    assert len([u for u in units if u["role"] == "candidate"]) <= MAX_DYNAMIC
    assert [u["key"] for u in units if u["role"] == "candidate"] == fz["tested_keys"]
    res = []
    for regime in REGIMES:
        R = Runner(regime)
        for u in units:
            r = R.leverage(u["kc_type"], u["mbon_type"], LEVERAGE_SEEDS)
            r.update(role=u["role"], rank=u["rank"], dan_type=u.get("reference_dan") or u["dan_type"],
                     static_input_fraction=u["static_input_fraction"])
            res.append(r)
            print(f"[leverage {regime}] {u['role']:<9} #{u['rank']:<4} {u['key']:<24} base {np.mean(r['baseline_mbon']):7.1f} "
                  f"zero {np.mean(r['zeroed_mbon']):7.1f} rel {100 * (r['rel_change'] or 0):+7.2f}% lower {r['lower_seeds']}/6 "
                  f"pass {r['passes']}", flush=True)
        del R
    jdump("dynamic-leverage-results.json", dict(seeds=list(LEVERAGE_SEEDS), regimes=list(REGIMES), primary_regime=PROBE,
                                                results=res, seconds=time.perf_counter() - t0))
    # ---- viable circuits
    vis = jload("kc-visual-response.json")
    circuits = []
    for u in units:
        lv = {r["regime"]: r for r in res if r["kc_type"] == u["kc_type"] and r["mbon_type"] == u["mbon_type"]}
        v = bool(vis["classes"][u["kc_type"]]["passes_gate"])
        dan_ok = (u.get("dan_type") or u.get("reference_dan")) is not None and u["dan_match"]
        status = ("coexisting" if v and dan_ok and lv[PROBE]["passes"] else
                  "split" if v and dan_ok and lv["80/3"]["passes"] else None)
        circuits.append(dict(key=u["key"], role=u["role"], rank=u["rank"], kc_type=u["kc_type"], mbon_type=u["mbon_type"],
                             dan_type=u.get("reference_dan") or u["dan_type"], topology_dan=u["dan_type"], visual_gate=v,
                             dan_match=dan_ok, leverage={k: dict(rel=x["rel_change"], lower=x["lower_seeds"], passes=x["passes"])
                                                         for k, x in lv.items()},
                             viable=status is not None, status=status))
    viable = [c for c in circuits if c["viable"]]
    order = sorted(viable, key=lambda c: (c["status"] != "coexisting", c["rank"]))
    jdump("candidate-circuits.json", dict(rule="viable = visual gate AND leverage pass (60/8 -> coexisting; else 80/3 -> split) "
                                          "AND topology-matched DAN; confirmation order: coexisting first, then split, each by "
                                          "frozen rank; cap 3", circuits=circuits, viable=[c["key"] for c in viable],
                                          confirmation_set=[c["key"] for c in order[:MAX_CONFIRM]]))
    print("viable", [(c["key"], c["status"]) for c in viable])


def stage_confirm():
    t0 = time.perf_counter()
    cc = jload("candidate-circuits.json")
    chosen = [c for k in cc["confirmation_set"] for c in cc["circuits"] if c["key"] == k]
    descriptive = not chosen
    if descriptive:                                   # preregistered fallback: best by rank among tested candidates
        pool = [c for c in cc["circuits"] if c["role"] == "candidate"] or cc["circuits"]
        chosen = [sorted(pool, key=lambda c: c["rank"])[0]]
    assert len(chosen) <= MAX_CONFIRM
    per_regime = {}
    for regime in REGIMES:
        R = Runner(regime)
        vr = visual_report(R, CONF_SEEDS)
        rows = {}
        for c in chosen:
            lev = R.leverage(c["kc_type"], c["mbon_type"], CONF_SEEDS)
            dan = R.dan(c["dan_type"], CONF_SEEDS) if c["dan_type"] else None
            vc = vr["classes"][c["kc_type"]]
            nat = R.natural_drive(c["kc_type"], c["mbon_type"], c["dan_type"], vc["best_stimulus"], CONF_SEEDS)
            rows[c["key"]] = dict(visual_gate=vc["passes_gate"], visual=vc, leverage=lev, dan=dan, natural_drive_descriptive=nat)
            print(f"[confirm {regime}] {c['key']:<24} visual {vc['passes_gate']} ({vc['best_stimulus']} "
                  f"{100 * (vc['stimuli'][vc['best_stimulus']]['no_visual']['rel_diff'] or 0):+.2f}%) leverage "
                  f"{100 * (lev['rel_change'] or 0):+.2f}% ({lev['lower_seeds']}/10) {lev['passes']} DAN "
                  f"{dan['delta_pp'] if dan else float('nan'):+.1f}pp usable {dan['usable'] if dan else None}", flush=True)
        per_regime[regime] = dict(rows=rows, visual_passing_classes=vr["passing_classes"], w_unchanged=vr["w_unchanged"],
                                  t4_sha256=vr["t4_sha256"])
        del R
    conf, sens = [], []
    for c in chosen:
        a = per_regime[PROBE]["rows"][c["key"]]; b = per_regime["80/3"]["rows"][c["key"]]
        lev_regime = PROBE if c["status"] in ("coexisting", None) else "80/3"
        lev_ok = per_regime[lev_regime]["rows"][c["key"]]["leverage"]["passes"]
        conf.append(dict(key=c["key"], status=c["status"], descriptive_only=descriptive, visual_regime=PROBE,
                         leverage_regime=lev_regime, visual_confirms=a["visual_gate"], leverage_confirms=lev_ok,
                         confirmed=bool(not descriptive and a["visual_gate"] and lev_ok)))
        for rg, x in ((PROBE, a), ("80/3", b)):
            vb = x["visual"]["stimuli"][x["visual"]["best_stimulus"]]
            sens.append(dict(key=c["key"], regime=rg, visual_gate=x["visual_gate"], visual_best=x["visual"]["best_stimulus"],
                             visual_rel_no_visual=vb["no_visual"]["rel_diff"], visual_rel_gray=vb["gray"]["rel_diff"],
                             visual_consistent=vb["no_visual"]["consistent_seeds"], kc_rate_hz=x["visual"]["baseline_rate_hz"],
                             leverage_rel=x["leverage"]["rel_change"], leverage_lower=x["leverage"]["lower_seeds"],
                             leverage_passes=x["leverage"]["passes"], mbon_rate_hz=x["leverage"]["baseline_mbon_rate_hz"],
                             dan_delta_pp=x["dan"]["delta_pp"] if x["dan"] else None,
                             dan_baseline_hz=x["dan"]["baseline_rate_hz"] if x["dan"] else None,
                             dan_usable=x["dan"]["usable"] if x["dan"] else None,
                             coexist=bool(x["visual_gate"] and x["leverage"]["passes"]),
                             natural_mbon_visual_rel=x["natural_drive_descriptive"]["mbon_visual_rel_vs_no_visual"],
                             natural_mbon_zeroed_rel=x["natural_drive_descriptive"]["mbon_zeroed_rel_under_visual_drive"]))
    jdump("confirmation-results.json", dict(seeds=list(CONF_SEEDS), descriptive_confirmation_only=descriptive,
                                            chosen=[c["key"] for c in chosen], confirmation=conf,
                                            per_regime=per_regime, seconds=time.perf_counter() - t0))
    jdump("operating-regime-sensitivity.json", dict(seeds=list(CONF_SEEDS), regimes={k: v.__dict__ for k, v in REGIMES.items()},
                                                    rows=sens, any_coexist=any(s["coexist"] for s in sens)))
    print(json.dumps(conf, indent=1)); print(json.dumps(sens, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["visual", "rank", "leverage", "confirm"])
    a = ap.parse_args()
    {"visual": stage_visual, "rank": stage_rank, "leverage": stage_leverage, "confirm": stage_confirm}[a.stage]()


if __name__ == "__main__":
    main()
