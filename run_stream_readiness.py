"""E30: recover the E25 champion on current main and measure stream readiness.

Stages:
  provenance   extract the champion from the committed E25 artifact (commit 40d644b), verify its hash,
               write champion-genome.json, e25-champion-provenance.json, historical-e25-baseline.json,
               repository-metadata.json
  reproduce    held-out seeds 2601-2620 x {champion normal / no vision / shuffled T4, matched random,
               uniform random} with the new live runtime (flymon.live_runtime) -> per-episode results
  analyze      reproduction-results.json, vision-ablation-results.json, dialogue-stall-analysis.json

The runtime never feeds RAM to the controller; evaluator RAM is read only after each action.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "e30-stream-readiness"
E25 = ROOT / "results" / "experiment-25-evolution"
E25_COMMIT = "40d644b2aa5c715091ca35e3b2b7e8360ace9f4a"
HELDOUT_SEEDS = tuple(range(2601, 2621))
BUDGET = 1500
CONDITIONS = (("champion", "normal"), ("champion", "none"), ("champion", "shuffled"),
              ("random_matched", "none"), ("random_uniform", "none"))
WORKERS = 8


def jdump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float) + "\n"); tmp.replace(OUT / name)


def jload(name):
    return json.loads((OUT / name).read_text())


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def sha256_file(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- provenance
def stage_provenance():
    from flymon import evolution as evo
    from flymon.live_runtime import CHAMPION_SHA256
    src = "results/experiment-25-evolution/champion-weights.json"
    committed = json.loads(git("show", f"{E25_COMMIT}:{src}"))
    fin = json.loads(git("show", f"{E25_COMMIT}:results/experiment-25-evolution/finalists.json"))["finalists"]
    ce = json.loads(git("show", f"{E25_COMMIT}:results/experiment-25-evolution/champion-evaluation.json"))
    f5 = next(f for f in fin if f["sha256"] == CHAMPION_SHA256)
    g = evo.Genome.from_json(f5["genome"])                         # hash verified inside from_json
    assert g.sha256() == CHAMPION_SHA256 == committed["genome_sha256"] == ce["champion"]["sha256"]
    Wc = np.asarray(committed["W"]); bc = np.asarray(committed["b"])
    assert np.array_equal(Wc, g.W) and np.array_equal(bc, g.b)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "champion-genome.json").write_text(json.dumps(g.to_json(), indent=1) + "\n")
    jdump("e25-champion-provenance.json", dict(
        historical_commit=E25_COMMIT, historical_pr=9,
        source_artifact=f"{E25_COMMIT}:results/experiment-25-evolution/finalists.json (finalist rank 5); cross-checked "
                        f"against {src} and champion-evaluation.json",
        recovery_method="committed E25 artifact (preference 1); genome re-serialised with flymon.evolution.Genome.to_json "
                        "and its SHA-256 re-verified; no manual weights",
        genome_id=CHAMPION_SHA256, genome_sha256=g.sha256(), finalist_rank=5, arch=g.arch,
        parameter_count=int(g.flat().size), shape=dict(W=list(g.W.shape), b=len(g.b), log_temp=1),
        temperature=g.temperature(), action_order=list(evo.ACTIONS), feature_order=evo.feature_names("t4"),
        feature_architecture="14 pooled frozen-T4 features (T4a-d x L/R eye; 2 x 3 screen-region grid of RF centres), "
                             "their one-decision deltas, and a one-hot of the controller's own previous action (35); "
                             "logits = W x + b, softmax(logits / T)",
        frozen_t4_sha256_prefix="87829806e398f66b", champion_genome_file="results/e30-stream-readiness/champion-genome.json",
        champion_genome_file_sha256=sha256_file(OUT / "champion-genome.json")))
    heldout = ce["champion"]
    jdump("historical-e25-baseline.json", dict(
        source=f"{E25_COMMIT}:results/experiment-25-evolution/champion-evaluation.json and analysis.md",
        architecture="t4 (frozen E23/E24 T4 pathway; 253-parameter linear softmax decoder)", actions=list(evo.ACTIONS),
        generations="38 (8 screen + 30 main)", main_population=32, heldout_seeds=list(HELDOUT_SEEDS), budget=BUDGET,
        champion=dict(normal=heldout["normal"], no_vision=heldout["no_vision"], shuffled=heldout["shuffled"],
                      no_vision_drop=heldout["no_vision_drop"], shuffled_drop=heldout["shuffled_drop"]),
        baselines=ce["baselines"], comparisons=ce["comparisons"],
        headline=dict(champion_median=5428.46, gen0_median=-22.6, matched_random_median=905, uniform_random_median=1384,
                      p_vs_uniform=0.06, house_exit_rate=0.65, oak_lab_rate=0.25, route1=0, battle=0,
                      no_vision_degradation=-0.76, shuffled_degradation=-0.77, finalists_lacking_vision_dependence="3/8",
                      long_text_window_seed_fraction=0.40)))
    rom = os.environ.get("POKEMON_ROM")
    import flybrain, pyboy
    jdump("repository-metadata.json", dict(
        branch=git("rev-parse", "--abbrev-ref", "HEAD"), base_commit=git("rev-parse", "HEAD"),
        origin_main=git("rev-parse", "origin/main"), e29_commit="d5389b410e5ccf96646dbeaa519ad5f58807cabd",
        e29_merged_in_main=subprocess.run(["git", "merge-base", "--is-ancestor", "d5389b410e5ccf96646dbeaa519ad5f58807cabd",
                                           "origin/main"], cwd=ROOT).returncode == 0,
        e25_commit=E25_COMMIT, python=sys.version.split()[0], pyboy=getattr(pyboy, "__version__", "unknown"),
        numpy=np.__version__, flybrain="0.1.0",
        rom_sha256=sha256_file(rom) if rom and Path(rom).is_file() else None,
        start_state_sha256=sha256_file(ROOT / "states" / "bedroom.state")))
    print("champion", g.sha256()[:16], g.flat().size)


# ---------------------------------------------------------------- reproduction (worker pool)
_S = {}


def _init():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from flymon.emulator import PokemonEmulator
    from flymon.live_runtime import SensoryContext, load_champion
    _S.update(sensory=SensoryContext(), game=PokemonEmulator(), genome=load_champion())


def run_one(task):
    from flymon.live_runtime import Episode, FrozenController, RandomController
    from flymon.stream_eval import StreamEvaluator
    t0 = time.perf_counter()
    kind, vision, seed = task["kind"], task["vision"], task["seed"]
    ctl = FrozenController(_S["genome"], _S["sensory"], vision) if kind == "champion" else RandomController(kind)
    ep = Episode(_S["game"], ctl, seed, task.get("budget", BUDGET), evaluator_factory=StreamEvaluator)
    ep.start()
    while not ep.done:
        ep.step()
    r = ep.evaluator.result()
    r.update(kind=kind, vision=vision, seed=seed, start_wait_frames=ep.start_wait, sensory_sha256=ctl.sensory_sha256(),
             genome_sha256=ctl.sha256, wall_s=time.perf_counter() - t0,
             log=[list(x[:7]) for x in ep.evaluator.rows])
    return r


def stage_reproduce():
    tasks = [dict(kind=k, vision=v, seed=s) for (k, v) in CONDITIONS for s in HELDOUT_SEEDS]
    t0 = time.perf_counter(); res = []
    with mp.get_context("spawn").Pool(WORKERS, initializer=_init) as p:
        for i, r in enumerate(p.imap_unordered(run_one, tasks, chunksize=1)):
            res.append(r)
            if (i + 1) % 10 == 0:
                print(f"[reproduce] {i + 1}/{len(tasks)} {time.perf_counter() - t0:.0f}s", flush=True)
    with gzip.open(OUT / "reproduction-raw.json.gz", "wt") as f:
        json.dump(dict(results=res, wall_s=time.perf_counter() - t0, workers=WORKERS), f)
    print(f"done {time.perf_counter() - t0:.0f}s")


# ---------------------------------------------------------------- statistics
def sign_flip_p(d):
    d = np.asarray(d, float)
    obs = abs(d.mean()); n = len(d)
    if n > 22:
        raise ValueError("exact enumeration only")
    signs = np.array(list(itertools.product((1, -1), repeat=n)))
    return float(np.mean(np.abs((signs * d).mean(1)) >= obs - 1e-12))


def summarize(rs):
    f = np.array([r["fitness"] for r in rs])
    sm = lambda k: float(np.mean([r["stream_milestones"][k] is not None for r in rs]))
    act = {a: float(np.mean([r["action_counts"][a] / r["decisions"] for r in rs])) for a in rs[0]["action_counts"]}
    return dict(n=len(rs), median=float(np.median(f)), mean=float(f.mean()), p10=float(np.percentile(f, 10)),
                worst=float(f.min()), best=float(f.max()), per_seed={str(r["seed"]): r["fitness"] for r in rs},
                bedroom_exit=sm("bedroom_exited"), house_exit=sm("house_exited"), oak_lab=sm("oak_lab"),
                route1=sm("route1"), battle=sm("battle"),
                e25_milestone_rate={m: float(np.mean([r["milestones"][m] is not None for r in rs])) for m in
                                    ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")},
                unique_tiles=float(np.mean([r["unique_tiles"] for r in rs])),
                window_decisions=float(np.mean([r["window_decisions"] for r in rs])),
                e25_lock_seed_fraction=float(np.mean([r["dialogue_locks"] > 0 for r in rs])),
                action_fraction=act)


def stage_analyze():
    raw = json.load(gzip.open(OUT / "reproduction-raw.json.gz", "rt"))["results"]
    by = {}
    for r in raw:
        by.setdefault(f"{r['kind']}|{r['vision']}", []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["seed"])
    cond = {k: summarize(v) for k, v in by.items()}
    hist_raw = json.load(gzip.open(E25 / "heldout-raw.json.gz", "rt"))["results"]
    hist_logs = json.load(gzip.open(E25 / "champion-heldout-logs.json.gz", "rt"))
    from flymon.live_runtime import CHAMPION_SHA256
    hmap = {"champion|normal": ("genome", "normal"), "champion|none": ("genome", "none"),
            "champion|shuffled": ("genome", "shuffled"), "random_matched|none": ("random_matched", "normal"),
            "random_uniform|none": ("random_uniform", "normal")}
    per_cond = {}
    for key, (hk, hv) in hmap.items():
        hist = {r["task"]["seed"]: r for r in hist_raw if r["task"]["kind"] == hk and r["task"].get("vision", "normal") == hv
                and (hk != "genome" or r["genome_sha256"] == CHAMPION_SHA256)}
        rows = []
        for r in by[key]:
            h = hist[r["seed"]]
            rows.append(dict(seed=r["seed"], historical=h["fitness"], recovered=r["fitness"],
                             identical_fitness=bool(abs(h["fitness"] - r["fitness"]) <= 1e-9 * max(1, abs(h["fitness"]))),
                             historical_milestones={k: v for k, v in h["milestones"].items() if v is not None},
                             recovered_milestones={k: v for k, v in r["milestones"].items() if v is not None},
                             identical_milestones=h["milestones"] == r["milestones"],
                             identical_actions=h["action_counts"] == r["action_counts"]))
        hmed = float(np.median([x["historical"] for x in rows])); rmed = float(np.median([x["recovered"] for x in rows]))
        per_cond[key] = dict(rows=rows, historical_median=hmed, recovered_median=rmed,
                             rel_diff=(rmed - hmed) / abs(hmed) if hmed else None,
                             identical_fitness_seeds=sum(x["identical_fitness"] for x in rows),
                             identical_milestone_seeds=sum(x["identical_milestones"] for x in rows))
    logs_identical = []
    for r in by["champion|normal"]:
        h = hist_logs[str(r["seed"])]["log"]
        logs_identical.append(dict(seed=r["seed"], identical_decisions=int(sum(tuple(a) == tuple([b[0], b[1] or "NONE"] + b[2:]) for a, b in zip(r["log"], h))),
                                   note="E25 logged NONE as null; normalised to \"NONE\" before comparison",
                                   of=len(h)))
    champ = cond["champion|normal"]
    s1 = dict(scalar_within_10pct=abs(per_cond["champion|normal"]["rel_diff"]) <= 0.10,
              action_logs_identical_all_seeds=all(x["identical_decisions"] == x["of"] for x in logs_identical),
              house_exits_common=champ["house_exit"] >= 0.4, oak_lab_sometimes=0 < champ["oak_lab"] < 1,
              route1_not_reliable=champ["route1"] < 0.5)
    s1["pass"] = bool(s1["scalar_within_10pct"] and s1["house_exits_common"] and s1["oak_lab_sometimes"] and s1["route1_not_reliable"])
    jdump("reproduction-results.json", dict(
        seeds=list(HELDOUT_SEEDS), budget=BUDGET, conditions=per_cond, champion_action_logs=logs_identical,
        champion=champ, historical=jload("historical-e25-baseline.json")["champion"]["normal"], gate_S1=s1))
    f = {k: np.array([r["fitness"] for r in v]) for k, v in by.items()}
    comp = {}
    for other in ("champion|none", "champion|shuffled", "random_matched|none", "random_uniform|none"):
        d = f["champion|normal"] - f[other]
        comp[other] = dict(median_other=float(np.median(f[other])), rel_change_vs_intact=float(np.median(f[other]) / np.median(f["champion|normal"]) - 1),
                           mean_diff=float(d.mean()), median_diff=float(np.median(d)), seeds_favouring_champion=int((d > 0).sum()),
                           dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None, p_exact_sign_flip=sign_flip_p(d),
                           house_exit=cond[other]["house_exit"], oak_lab=cond[other]["oak_lab"])
    s2 = dict(no_vision_drop=-comp["champion|none"]["rel_change_vs_intact"], shuffled_drop=-comp["champion|shuffled"]["rel_change_vs_intact"])
    s2["pass"] = bool(s2["no_vision_drop"] >= 0.5 and s2["shuffled_drop"] >= 0.5)
    jdump("vision-ablation-results.json", dict(conditions=cond, paired_vs_intact=comp, gate_S2=s2,
                                               test="exact two-sided paired sign-flip test on per-seed fitness (2^20 permutations)"))
    dia = {}
    for k, v in by.items():
        ds = [r["dialogue"] for r in v]
        n_long = sum(d["n_long"] for d in ds)
        cnt = {c: sum(d["counts"][c] for d in ds) for c in ("scripted", "navigable", "stall")}
        dec = {c: sum(d["decisions"][c] for d in ds) for c in ("scripted", "navigable", "stall")}
        dia[k] = dict(long_events=n_long, counts=cnt, decisions=dec,
                      scripted_fraction_of_long=cnt["scripted"] / n_long if n_long else None,
                      stall_fraction_of_long=cnt["stall"] / n_long if n_long else None,
                      seeds_with_long_window=int(sum(d["n_long"] > 0 for d in ds)),
                      seeds_with_genuine_stall=int(sum(d["any_genuine_stall"] for d in ds)),
                      genuine_stall_decision_fraction=float(sum(d["genuine_stall_decisions"] for d in ds) / sum(d["decisions_total"] for d in ds)),
                      overworld_stall_runs=int(sum(len(d["overworld_stall_runs"]) for d in ds)),
                      stall_reasons={r_: sum(e["reason"] == r_ for d in ds for e in d["events"] if e["cls"] == "stall")
                                     for r_ in ("frozen_window", "repeat_loop")},
                      events=[dict(seed=r["seed"], **e) for r in v for e in r["dialogue"]["events"]])
    jdump("dialogue-stall-analysis.json", dict(rules=dict(long_window=30, frozen_window=30, repeat_limit=2, overworld_stall=150,
                                                          source="flymon/stream_eval.py; frozen in reproduction-preregistration.md"),
                                               conditions=dia))
    print(json.dumps(dict(S1=s1, S2=s2, champion={k: champ[k] for k in ("median", "house_exit", "oak_lab", "route1", "battle")},
                          dialogue={k: {x: dia[k][x] for x in ("long_events", "counts", "seeds_with_genuine_stall")} for k in dia}),
                     indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["provenance", "reproduce", "analyze"])
    a = ap.parse_args()
    {"provenance": stage_provenance, "reproduce": stage_reproduce, "analyze": stage_analyze}[a.stage]()


if __name__ == "__main__":
    main()
