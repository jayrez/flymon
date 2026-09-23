"""Experiment 25 analysis: training trajectories, held-out champion evaluation, vision causality."""
from __future__ import annotations

import gzip
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_tonic_disinhibition_experiment import _svg, bars, histogram, panel_traces
from flymon import evolution as evo

ROOT = Path(__file__).resolve().parent
R = ROOT / "results" / "experiment-25-evolution"
CAP = ROOT / "captures" / "experiment-25"
MAJOR = ("M3", "M4", "M5", "M7", "M8")


def load_jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def sign_flip_p(d):
    d = np.asarray(d, float)
    if not len(d) or np.all(d == 0):
        return 1.0
    obs = abs(d.mean())
    if len(d) <= 20:
        tot = cnt = 0
        for s in itertools.product((1, -1), repeat=len(d)):
            tot += 1; cnt += abs((d * np.array(s)).mean()) >= obs - 1e-12
        return cnt / tot
    rng = np.random.default_rng(0)
    sims = np.abs((d * rng.choice([1, -1], size=(100000, len(d)))).mean(axis=1))
    return float((1 + np.sum(sims >= obs - 1e-12)) / (1 + len(sims)))


def stats(rs):
    f = np.array([r["fitness"] for r in rs])
    return dict(n=len(rs), mean=float(f.mean()), median=float(np.median(f)), p10=float(np.quantile(f, 0.1)),
                worst=float(f.min()), best=float(f.max()),
                milestone_rate={m: float(np.mean([r["milestones"][m] is not None for r in rs]))
                                for m in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")},
                major_seeds=int(sum(any(r["milestones"][m] is not None for m in MAJOR) for r in rs)),
                unique_tiles=float(np.mean([r["unique_tiles"] for r in rs])),
                unique_maps=float(np.mean([r["unique_maps"] for r in rs])),
                loop_fraction_median=float(np.median([r["loop_fraction_nonidle"] for r in rs])),
                idle_fraction=float(np.mean([r["idle_fraction"] for r in rs])),
                lock_seed_fraction=float(np.mean([r["dialogue_locks"] > 0 for r in rs])),
                completed_interactions=float(np.mean([r["completed_window_cycles"] for r in rs])),
                action_fraction={a: float(np.mean([r["action_counts"][a] / r["decisions"] for r in rs]))
                                 for a in evo.ACTIONS})


def by_seed(rs):
    return {r["task"]["seed"]: r for r in rs}


def main():
    CAP.mkdir(parents=True, exist_ok=True)
    screen = json.loads((R / "architecture-screen.json").read_text())
    arch = screen["winner"]
    gens = {a: load_jsonl(R / f"screen-{a}-generations.jsonl") for a in ("dn", "t4", "t4dn")}
    main_rows = load_jsonl(R / "main-generations.jsonl")
    lineage = gens[arch] + main_rows                       # winning lineage, screen then main
    for i, row in enumerate(lineage):
        row["lineage_generation"] = i

    # ---- first milestone occurrences in the winning lineage ----
    def first(pred):
        return next((r["lineage_generation"] for r in lineage if pred(r)), None)
    firsts = dict(first_down_use=first(lambda r: r["any_down"]),
                  first_consistent_bedroom_exit=first(lambda r: r["milestone_rate"]["M2"] >= 0.5),
                  first_bedroom_exit=first(lambda r: r["milestone_rate"]["M2"] > 0),
                  first_house_exit=first(lambda r: r["milestone_rate"]["M3"] > 0),
                  first_outdoor_navigation=first(lambda r: r["milestone_rate"]["M4"] > 0),
                  first_other_map=first(lambda r: r["milestone_rate"]["M5"] > 0),
                  first_route1=first(lambda r: r["milestone_rate"]["M7"] > 0),
                  first_completed_interaction=first(lambda r: r["any_completed_interaction"]),
                  first_battle=first(lambda r: r["milestone_rate"]["M8"] > 0))

    # ---- held-out ----
    raw = json.loads(gzip.open(R / "heldout-raw.json.gz").read())
    res = raw["results"]
    groups = {}
    for r in res:
        t = r["task"]
        key = (t["role"], t.get("rank"), t.get("vision", "normal"), t.get("ablate"))
        groups.setdefault(key, []).append(r)
    fin = json.loads((R / "finalists.json").read_text())["finalists"]
    gen0 = by_seed(groups[("gen0", None, "normal", None)])
    rnd_m = by_seed(groups[("random_matched", None, "normal", None)])
    rnd_u = by_seed(groups[("random_uniform", None, "normal", None)])
    gen0_median = float(np.median([r["fitness"] for r in gen0.values()]))
    finalists = []
    for f in fin:
        k = f["rank"]
        v0, v1, v2 = (by_seed(groups[("finalist", k, v, None)]) for v in ("normal", "none", "shuffled"))
        s0, s1, s2 = stats(list(v0.values())), stats(list(v1.values())), stats(list(v2.values()))
        drop = lambda s: (s0["median"] - s["median"]) / abs(s0["median"]) if s0["median"] else None
        reject = []
        if s0["worst"] < gen0_median:
            reject.append("worst case below Gen0 median")
        if s0["major_seeds"] < 3:
            reject.append("M3+ on fewer than 3 seeds")
        if abs(s1["median"] - s0["median"]) <= 0.10 * abs(s0["median"]):
            reject.append("no-vision median within 10% of normal")
        if abs(s2["median"] - s0["median"]) <= 0.10 * abs(s0["median"]):
            reject.append("shuffled median within 10% of normal")
        if s0["loop_fraction_median"] > 0.5:
            reject.append("median non-idle loop fraction > 0.5")
        if s0["lock_seed_fraction"] > 0.2:
            reject.append("dialogue lock on > 20% of seeds")
        finalists.append(dict(rank=k, sha256=f["sha256"], training_mean_fitness=f["training_mean_fitness"],
                              normal=s0, no_vision=s1, shuffled=s2, no_vision_drop=drop(s1), shuffled_drop=drop(s2),
                              rejections=reject))
    champ = sorted(finalists, key=lambda f: (-f["normal"]["p10"], -f["normal"]["median"], -f["normal"]["mean"],
                                              f["rank"]))[0]
    seeds = sorted(gen0)
    cv0 = by_seed(groups[("finalist", champ["rank"], "normal", None)])
    cv1 = by_seed(groups[("finalist", champ["rank"], "none", None)])
    cv2 = by_seed(groups[("finalist", champ["rank"], "shuffled", None)])
    def paired(other):
        d = [cv0[s]["fitness"] - other[s]["fitness"] for s in seeds]
        return dict(mean_difference=float(np.mean(d)), champion_better_seeds=int(sum(x > 0 for x in d)),
                    exact_sign_flip_p=sign_flip_p(d))
    comparisons = dict(vs_gen0=paired(gen0), vs_random_matched=paired(rnd_m), vs_random_uniform=paired(rnd_u),
                       vs_no_vision=paired(cv1), vs_shuffled=paired(cv2))
    baselines = dict(gen0=stats(list(gen0.values())), random_matched=stats(list(rnd_m.values())),
                     random_uniform=stats(list(rnd_u.values())))
    for role in sorted({k[0] for k in groups if k[0].startswith("screen-best") or k[0].startswith("flybrain")}):
        rs = [r for k, v in groups.items() if k[0] == role for r in v]
        baselines[role] = stats(rs)
    cs = champ["normal"]
    crit = dict(
        S1=bool(cs["median"] > baselines["gen0"]["median"] and cs["median"] > baselines["random_matched"]["median"]
                and cs["median"] > baselines["random_uniform"]["median"]
                and all(comparisons[k]["exact_sign_flip_p"] < 0.05 and comparisons[k]["mean_difference"] > 0
                        for k in ("vs_gen0", "vs_random_matched", "vs_random_uniform"))),
        S2=bool(np.mean([any(cv0[s]["milestones"][m] is not None for m in MAJOR) for s in seeds]) >= 0.25),
        S3=bool(cs["action_fraction"]["DOWN"] >= 0.01),
        S4=bool(champ["no_vision_drop"] is not None and champ["no_vision_drop"] >= 0.30
                and champ["shuffled_drop"] is not None and champ["shuffled_drop"] >= 0.15))
    screen_first = gens[arch][0]["median"] if gens[arch] else None
    last3_median = float(np.mean([r["median"] for r in main_rows[-3:]])) if main_rows else None
    training_improves = bool(last3_median is not None and screen_first is not None and last3_median > screen_first)
    if all(crit.values()) and not champ["rejections"]:
        readiness = "READY FOR CHAMPION STAGING"
    elif training_improves and crit["S1"] or (training_improves and comparisons["vs_gen0"]["mean_difference"] > 0
                                                and comparisons["vs_gen0"]["exact_sign_flip_p"] < 0.05):
        readiness = "EVOLUTION WORKS, NOT STREAM-READY"
    else:
        readiness = "NO USEFUL EVOLUTION"
    vision_dependent = crit["S4"]
    out = dict(architecture=arch, firsts=firsts, finalists=finalists, champion=dict(
        rank=champ["rank"], sha256=champ["sha256"], normal=champ["normal"], no_vision=champ["no_vision"],
        shuffled=champ["shuffled"], no_vision_drop=champ["no_vision_drop"], shuffled_drop=champ["shuffled_drop"],
        rejections=champ["rejections"]), baselines=baselines, comparisons=comparisons, criteria=crit,
        training=dict(screen_generation0_median=screen_first, main_last3_median=last3_median,
                      improves=training_improves),
        vision_dependent=vision_dependent, readiness=readiness, heldout_wall_s=raw["wall_s"],
        heldout_episodes=len(res), selection_rule="10th percentile, then median, then mean (normal vision)")
    (R / "champion-evaluation.json").write_text(json.dumps(out, indent=2, default=str) + "\n")

    # ---- champion weights ----
    g = evo.Genome.from_json(next(f["genome"] for f in json.loads((R / "finalists.json").read_text())["finalists"]
                                   if f["sha256"] == champ["sha256"]))
    names = evo.feature_names(g.arch)
    W = g.W
    top = []
    for ai, a in enumerate(evo.ACTIONS):
        order = np.argsort(-np.abs(W[ai]))[:5]
        top.append(dict(action=a, bias=float(g.b[ai]), strongest=[(names[j], float(W[ai, j])) for j in order]))
    groupnorm = {}
    for grp, pred in (("t4", lambda n: n in evo.t4_feature_names()), ("dn", lambda n: n in evo.dn_feature_names()),
                      ("prev_action", lambda n: n.startswith("prev_"))):
        idx = [i for i, n in enumerate(names) if pred(n)]
        if idx:
            groupnorm[grp] = float(np.abs(W[:, idx]).mean())
    subtype = {s: float(np.abs(W[:, [i for i, n in enumerate(names) if n.startswith(s)]]).mean())
               for s in evo.T4_SUBTYPES if any(n.startswith(s) for n in names)}
    interp = dict(genome_sha256=g.sha256(), arch=g.arch, temperature=g.temperature(), feature_names=names,
                  W=W.tolist(), b=g.b.tolist(), strongest_per_action=top, mean_abs_weight_by_group=groupnorm,
                  mean_abs_weight_by_t4_subtype=subtype,
                  note="weights describe the decoder, not causal effects; causality comes from the ablations")
    (R / "champion-weights.json").write_text(json.dumps(interp, indent=2) + "\n")

    # ---------------- figures ----------------
    x = np.arange(len(lineage))
    panel_traces(CAP / "01-fitness-by-generation.svg",
                 [("best", [("best", np.array([r["best"] for r in lineage]), "#275e85")]),
                  ("median", [("median", np.array([r["median"] for r in lineage]), "#c0392b")]),
                  ("top-quartile mean", [("topQ", np.array([r["top_quartile_mean"] for r in lineage]), "#27ae60")])],
                 f"Exp25 winning lineage ({arch}): screen generations 0-{len(gens[arch]) - 1}, then main")
    panel_traces(CAP / "02-milestones-by-generation.svg",
                 [(m, [(m, np.array([r["milestone_rate"][m] for r in lineage]), "#275e85")]) for m in
                  ("M1", "M2", "M3", "M4", "M5", "M7")],
                 "Exp25 fraction of training episodes reaching each milestone, per generation")
    bars(CAP / "03-architecture-screen.svg", [f"{a} score" for a in screen["architectures"]] +
         [f"{a} best" for a in screen["architectures"]],
         [v["score"] for v in screen["architectures"].values()] + [v["best"] for v in screen["architectures"].values()],
         "Exp25 architecture screen (training seeds only): top-quartile score (last 3 gens) and best fitness")
    panel_traces(CAP / "04-tiles-maps-by-generation.svg",
                 [("mean unique tiles", [("tiles", np.array([r["mean_unique_tiles"] for r in lineage]), "#275e85")]),
                  ("mean unique maps", [("maps", np.array([r["mean_unique_maps"] for r in lineage]), "#c0392b")])],
                 "Exp25 exploration per generation (training)")
    panel_traces(CAP / "05-idle-loop-by-generation.svg",
                 [("idle fraction", [("idle", np.array([r["mean_idle"] for r in lineage]), "#275e85")]),
                  ("non-idle loop fraction", [("loop", np.array([r["mean_loop"] for r in lineage]), "#c0392b")]),
                  ("DOWN fraction", [("down", np.array([r["action_fraction"]["DOWN"] for r in lineage]), "#27ae60")])],
                 "Exp25 idle / loop / DOWN usage per generation (training)")
    labs = ["Gen0", "random matched", "random uniform"] + [k for k in baselines if k.startswith("screen-best")] + ["champion"]
    vals = [baselines["gen0"]["median"], baselines["random_matched"]["median"], baselines["random_uniform"]["median"]] + \
        [baselines[k]["median"] for k in baselines if k.startswith("screen-best")] + [cs["median"]]
    bars(CAP / "06-gen0-random-evolved.svg", labs, vals, "Exp25 held-out median fitness (20 unseen seeds, 1,500 decisions)")
    bars(CAP / "07-vision-causality.svg", ["normal T4", "no vision", "shuffled geometry"],
         [champ["normal"]["median"], champ["no_vision"]["median"], champ["shuffled"]["median"]],
         "Exp25 champion held-out median fitness under vision ablations")
    bars(CAP / "08-finalists-heldout.svg",
         [f"#{f['rank']} {c}" for f in finalists for c in ("p10", "median", "mean")],
         [f["normal"][c] for f in finalists for c in ("p10", "median", "mean")],
         "Exp25 finalists on held-out seeds (normal vision)")
    histogram(CAP / "09-champion-robustness.svg", [cv0[s]["fitness"] for s in seeds], float(cs["median"]),
              "Exp25 champion fitness across 20 held-out seeds (line = median)")
    lab, val = [], []
    for a in top:
        for n, w in a["strongest"][:3]:
            lab.append(f"{a['action']} <- {n}"); val.append(w)
    bars(CAP / "10-champion-weights.svg", lab, val, "Exp25 champion: 3 strongest decoder weights per action")
    print(json.dumps(dict(readiness=readiness, criteria=crit, champion=dict(rank=champ["rank"], sha=champ["sha256"][:12],
                                                                          normal=cs["median"], p10=cs["p10"],
                                                                          nv=champ["no_vision"]["median"],
                                                                          sh=champ["shuffled"]["median"],
                                                                          rej=champ["rejections"]),
                          baselines={k: round(v["median"], 1) for k, v in baselines.items()}, firsts=firsts,
                          comparisons=comparisons), indent=1))


if __name__ == "__main__":
    main()
