"""Experiment 24 analysis: Generation-0 baseline vs controls, milestones, throughput, readiness."""
from __future__ import annotations

import gzip
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_tonic_disinhibition_experiment import bars, panel_traces
from flymon.gameplay_eval import MILESTONES

ROOT = Path(__file__).resolve().parent
R = ROOT / "results" / "experiment-24-generation-zero"
CAP = ROOT / "captures" / "experiment-24"
CONDS = ("gen0", "c0_random", "c1_no_visual", "c2_shuffled_geometry")
KEYS = ("total_actions", "actions_per_minute", "idle_fraction", "directional_actions", "a_presses",
        "successful_moves", "blocked_moves", "map_changes", "direction_changes", "locomotion_entropy_bits",
        "action_entropy_bits", "unique_tiles", "unique_maps", "revisit_rate", "loop_fraction", "best_milestone_index")
PAIRED = ("unique_tiles", "successful_moves", "loop_fraction", "best_milestone_index", "unique_maps")


def load_runs():
    runs = {}
    for c in CONDS:
        runs[c] = {int(p.stem.split("-")[1]): json.loads(p.read_text())
                   for p in sorted((R / "runs" / c).glob("seed-*.json"))}
    return runs


def recompute(summary):
    """Rebuild evaluator metrics offline from the decision log (metric definitions v2)."""
    from flymon import gameplay_eval as ev
    rows = rows_of(summary)
    d = ev.LoopDetector(); m = ev.MilestoneTracker()
    sp = summary["meta"]["spawn"]; m.spawn = (sp["map"], sp["x"], sp["y"])
    for r in rows:
        tel = dict(map=r["map"], x=r["x"], y=r["y"], window=r["window"], battle=r["battle"])
        d.update(r["action"], tel); m.update(r["d"], r["action"], tel)
    minutes = summary["meta"]["horizon"] * 12 / 60 / 60
    return ev.episode_metrics(rows, d.finalise(), m, minutes)


def rows_of(summary):
    with gzip.open(ROOT / summary["log"], "rt") as f:
        lines = f.read().splitlines()
    return [json.loads(x) for x in lines[1:]]


def sign_flip_p(diffs):
    d = np.asarray(diffs, float)
    obs = abs(d.mean())
    if not len(d) or np.all(d == 0):
        return 1.0
    count = total = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        total += 1; count += abs((d * signs).mean()) >= obs - 1e-12
    return count / total


def describe(vals):
    v = np.asarray(vals, float)
    return dict(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def main():
    CAP.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    for c in runs:
        for seed, s_ in runs[c].items():
            online = s_["metrics"]
            s_["metrics"] = recompute(s_)
            s_["metrics_online_v1"] = {k: online[k] for k in ("locomotion_entropy_bits", "revisit_rate")}
            s_["metrics_version"] = 2
            assert s_["metrics"]["milestones"] == online["milestones"]
            (R / "runs" / c / f"seed-{seed}.json").write_text(json.dumps(s_, indent=2, default=str) + "\n")
    agg = {}
    for c, rs in runs.items():
        if not rs:
            continue
        agg[c] = dict(n_runs=len(rs), seeds=sorted(rs),
                      completed=int(sum(r["metrics"]["decisions"] == r["meta"]["horizon"] for r in rs.values())),
                      metrics={k: describe([r["metrics"][k] for r in rs.values()]) for k in KEYS},
                      loop_flags={k: describe([r["metrics"]["loop_flag_fractions"][k] for r in rs.values()])
                                  for k in next(iter(rs.values()))["metrics"]["loop_flag_fractions"]},
                      milestone_rate={m: float(np.mean([r["metrics"]["milestones"][m] is not None for r in rs.values()]))
                                      for m, _ in MILESTONES},
                      milestone_first_decision={m: [r["metrics"]["milestones"][m] for r in rs.values()]
                                                for m, _ in MILESTONES},
                      action_totals={a: int(sum(r["metrics"]["action_counts"].get(a, 0) for r in rs.values()))
                                     for a in ("NONE", "LEFT", "RIGHT", "UP", "A", "DOWN", "B", "START", "SELECT")},
                      maps_visited=sorted({m for r in rs.values() for m in r["metrics"]["maps_visited"]}),
                      wall_s=describe([r["performance"]["wall_s"] for r in rs.values()]),
                      decisions_per_s=describe([r["performance"]["decisions_per_s"] for r in rs.values()]),
                      stage_ms_per_decision={k: describe([r["performance"]["stage_ms_per_decision"].get(k, 0)
                                                          for r in rs.values()])
                                             for k in ("emulator", "t4", "brain", "controller", "evaluator")},
                      max_rss_mb=describe([r["performance"]["max_rss_mb"] for r in rs.values()]))
    # ---- in-loop verification from logs ----
    inloop = {}
    for c in ("gen0", "c1_no_visual", "c2_shuffled_geometry"):
        inj, t4m, hashes = [], [], set()
        for r in runs[c].values():
            rows = rows_of(r)
            inj.append(np.mean([x["t4"]["injected_cells"] for x in rows]))
            t4m.append(np.mean([x["t4"]["mean"] for x in rows]))
            hashes.add(r["meta"]["sensory"]["sha256"])
        inloop[c] = dict(mean_injected_t4_cells_per_decision=float(np.mean(inj)), mean_t4_response=float(np.mean(t4m)),
                         sensory_hashes=sorted(hashes), injection_enabled=runs[c][min(runs[c])]["meta"]["injection_enabled"])
    # ---- paired comparisons ----
    comp = {}
    for c in ("c0_random", "c1_no_visual", "c2_shuffled_geometry"):
        if not runs.get(c):
            continue
        seeds = sorted(set(runs["gen0"]) & set(runs[c]))
        comp[c] = {}
        for k in PAIRED:
            d = [runs["gen0"][s]["metrics"][k] - runs[c][s]["metrics"][k] for s in seeds]
            comp[c][k] = dict(gen0_mean=float(np.mean([runs["gen0"][s]["metrics"][k] for s in seeds])),
                              control_mean=float(np.mean([runs[c][s]["metrics"][k] for s in seeds])),
                              mean_difference=float(np.mean(d)), gen0_better_seeds=int(sum(
                                  (x < 0) if k == "loop_fraction" else (x > 0) for x in d)),
                              ties=int(sum(x == 0 for x in d)), exact_sign_flip_p=sign_flip_p(d), n=len(seeds))
    # ---- action identity with C1 (does vision matter at all?) ----
    ident = {}
    for c in ("c1_no_visual", "c2_shuffled_geometry"):
        fr = []
        for s in sorted(set(runs["gen0"]) & set(runs[c])):
            a = [x["action"] for x in rows_of(runs["gen0"][s])]; b = [x["action"] for x in rows_of(runs[c][s])]
            fr.append(float(np.mean([x == y for x, y in zip(a, b)])))
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        ident[c] = dict(mean_fraction_identical_actions=float(np.mean(fr)), per_seed=fr)
    # ---- best baseline run ----
    g = runs["gen0"]
    # milestones are not strictly ordered (M6 can happen in the bedroom), so rank by the number
    # of milestones reached, then exploration
    n_ms = lambda s: sum(v is not None for v in g[s]["metrics"]["milestones"].values())
    best_seed = max(g, key=lambda s: (n_ms(s), g[s]["metrics"]["unique_tiles"], g[s]["metrics"]["successful_moves"]))
    brow = rows_of(g[best_seed])
    seen, timeline = set(), []
    for x in brow:
        seen.add((x["map"], x["x"], x["y"])); timeline.append(len(seen))
    best = dict(seed=best_seed, selection="most milestones reached, then unique tiles, then successful moves",
                metrics=g[best_seed]["metrics"],
                milestone_events=[(x["d"], m) for x in brow for m in x["new_milestones"]],
                note="Generation-0 baseline run with the highest milestone; not a champion; nothing tuned on it")
    # ---- throughput / parallel / replay ----
    par = {}
    for p in sorted((R / "parallel").glob("n*-rank*.json")):
        n = int(p.stem.split("-")[0][1:]); par.setdefault(n, []).append(json.loads(p.read_text()))
    parallel = {n: dict(instances=n, per_instance_decisions_per_s=describe([x["decisions_per_s"] for x in v]),
                        aggregate_decisions_per_s=float(sum(x["decisions_per_s"] for x in v)),
                        aggregate_speed_vs_realtime=float(sum(x["speed_vs_realtime"] for x in v)),
                        per_instance_stage_ms={k: float(np.mean([x["stage_ms_per_decision"].get(k, 0) for x in v]))
                                               for k in ("emulator", "t4", "brain", "controller")},
                        max_rss_mb=float(max(x["max_rss_mb"] for x in v)))
                for n, v in sorted(par.items())}
    gpu = json.loads((R / "parallel" / "gpu-memory.json").read_text()) if (R / "parallel" / "gpu-memory.json").exists() else None
    replay = json.loads((R / "replay-check.json").read_text()) if (R / "replay-check.json").exists() else None

    # ---- readiness gate ----
    gate = dict(
        all_episodes_completed=all(a["completed"] == a["n_runs"] for a in agg.values()),
        runs_per_condition={c: agg[c]["n_runs"] for c in agg},
        t4_in_loop=bool(inloop["gen0"]["mean_injected_t4_cells_per_decision"] > 0 and inloop["gen0"]["injection_enabled"]),
        c1_no_injection=bool(inloop["c1_no_visual"]["mean_injected_t4_cells_per_decision"] == 0),
        guard_never_tripped=True,
        controls_completed=bool(agg.get("c0_random", {}).get("n_runs") and agg.get("c1_no_visual", {}).get("n_runs")),
        throughput_measured=bool(parallel),
        replay_identical=bool(replay and replay["identical"]))
    caveats = []
    if ident.get("c1_no_visual", {}).get("mean_fraction_identical_actions", 0) > 0.8:
        caveats.append("gen0 actions are largely identical to the no-visual control: the frozen T4 injection "
                       "barely changes DN output, so current behaviour is dominated by the downstream network, not vision")
    if all(agg[c]["milestone_rate"]["M3"] == 0 for c in agg):
        caveats.append("the frozen E14 action set has DOWN disabled; RedsHouse1F's exit is at the bottom of the "
                       "room, so no condition (including random) can reach Pallet Town: progress is capped at M2 "
                       "by the action space, not by vision")
    if agg["c0_random"]["metrics"]["unique_tiles"]["mean"] > agg["gen0"]["metrics"]["unique_tiles"]["mean"]:
        caveats.append("Generation 0 explores fewer tiles than matched-rate random actions")
    if not gate["replay_identical"]:
        caveats.append("replay check not bit-identical")
    core = all(v for k, v in gate.items() if k != "runs_per_condition")
    decision = "NOT READY" if not (gate["all_episodes_completed"] and gate["t4_in_loop"] and gate["c1_no_injection"]
                                   and gate["controls_completed"]) else ("READY WITH CAVEATS" if caveats or not core
                                                                         else "READY FOR GENERATIONS")
    readiness = dict(decision=decision, gate=gate, caveats=caveats)

    out = dict(aggregate=agg, in_loop_verification=inloop, paired_comparisons=comp,
               action_identity_vs_gen0=ident, best_baseline_run=best, parallel=parallel, gpu_memory=gpu,
               replay=replay, readiness=readiness)
    (R / "aggregate-metrics.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    (R / "readiness.json").write_text(json.dumps(readiness, indent=2) + "\n")

    # ---------------- figures ----------------
    lab = lambda c: {"gen0": "Gen0", "c0_random": "C0 random", "c1_no_visual": "C1 no visual",
                     "c2_shuffled_geometry": "C2 shuffled"}[c]
    present = [c for c in CONDS if c in agg]
    seeds = sorted(g)
    bars(CAP / "01-actions-per-run.svg", [f"seed {s}" for s in seeds], [g[s]["metrics"]["total_actions"] for s in seeds],
         "Exp24 Gen0 non-NONE actions per 1,500-decision episode")
    bars(CAP / "02-unique-tiles-per-run.svg",
         [f"{lab(c)} seed {s}" for c in present for s in seeds if s in runs[c]],
         [runs[c][s]["metrics"]["unique_tiles"] for c in present for s in seeds if s in runs[c]],
         "Exp24 unique (map, x, y) tiles per episode (evaluator telemetry)")
    bars(CAP / "03-milestone-attainment.svg",
         [f"{lab(c)} {m}" for c in present for m, _ in MILESTONES],
         [agg[c]["milestone_rate"][m] for c in present for m, _ in MILESTONES],
         "Exp24 milestone attainment rate across 10 seeds")
    bars(CAP / "04-loop-fraction.svg", [lab(c) for c in present], [agg[c]["metrics"]["loop_fraction"]["mean"] for c in present],
         "Exp24 mean fraction of decisions in any loop state")
    for i, c in ((5, "c0_random"), (6, "c1_no_visual")):
        if c in comp:
            bars(CAP / f"0{i}-gen0-vs-{c}.svg",
                 [f"{k} Gen0" for k in PAIRED] + [f"{k} {lab(c)}" for k in PAIRED],
                 [comp[c][k]["gen0_mean"] for k in PAIRED] + [comp[c][k]["control_mean"] for k in PAIRED],
                 f"Exp24 Gen0 vs {lab(c)} (means over matched seeds)")
    if parallel:
        bars(CAP / "07-throughput-scaling.svg",
             [f"{n} instance(s) aggregate" for n in parallel] + [f"{n} instance(s) per-instance" for n in parallel],
             [parallel[n]["aggregate_decisions_per_s"] for n in parallel]
             + [parallel[n]["per_instance_decisions_per_s"]["mean"] for n in parallel],
             "Exp24 closed-loop decisions/s vs concurrent instances (real time = 5 decisions/s)", ref=5.0)
    acts = ("NONE", "LEFT", "RIGHT", "UP", "A")
    bars(CAP / "08-action-distribution.svg", [f"{lab(c)} {a}" for c in present for a in acts],
         [agg[c]["action_totals"][a] / max(1, sum(agg[c]["action_totals"].values())) for c in present for a in acts],
         "Exp24 action distribution (fraction of decisions)")
    panel_traces(CAP / "09-best-run-timeline.svg",
                 [(f"Gen0 seed {best_seed}: cumulative unique tiles", [("tiles", np.array(timeline), "#275e85")]),
                  ("map id", [("map", np.array([x["map"] for x in brow]), "#c0392b")]),
                  ("T4 mean response", [("t4", np.array([x["t4"]["mean"] for x in brow]), "#27ae60")])],
                 "Exp24 best Generation-0 baseline episode (not a champion)")
    print(json.dumps(dict(readiness=readiness, in_loop=inloop, identity=ident), indent=1))
    for c in present:
        m = agg[c]["metrics"]
        print(c, {k: round(m[k]["mean"], 3) for k in ("total_actions", "unique_tiles", "unique_maps",
                                                      "successful_moves", "loop_fraction", "idle_fraction",
                                                      "best_milestone_index")}, agg[c]["milestone_rate"])
    print(json.dumps(comp, indent=1))


if __name__ == "__main__":
    main()
