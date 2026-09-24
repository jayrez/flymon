"""Experiment 25: evolve a downstream Pokémon controller on the frozen E23 T4 pathway.

Stages (see results/experiment-25-evolution/preregistration.md):
  screen    architecture screen (dn, t4, t4dn), training seeds only
  main      continue the winning screen lineage
  finalists freeze the top 8
  heldout   finalists x unseen seeds x {normal, no vision, shuffled T4 geometry} + baselines
            + FlyBrain-contribution feature ablations

Episodes run in a pool of isolated worker processes (one emulator, one frozen T4 model and,
when needed, one FlyBrain per worker). The controller sees only preregistered neural features
(`flymon.evolution.FeatureBuilder`); RAM telemetry is read after each action by the evaluator.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "experiment-25-evolution"
CHECKPOINTS = RESULTS / "checkpoints"
TRAIN_SEEDS = tuple(range(2501, 2509))
SEED_PAIRS = ((2501, 2502), (2503, 2504), (2505, 2506), (2507, 2508))
HELDOUT_SEEDS = tuple(range(2601, 2621))
TRAIN_BUDGET, HELDOUT_BUDGET = 500, 1500
START_WAIT_MAX = 60
WORKERS = 8
SCREEN = dict(population=24, generations=8)
MAIN = dict(population=32, generations=30, early_stop_after=15, patience=12)
SHUFFLE_BASE = 25_000_000
HOLD, RELEASE, NEURAL_STEPS = 8, 4, 10


# ================================================================ worker side
_W = {}


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import run_fixed_background_experiment as E22
    from flymon.emulator import PokemonEmulator
    from flymon.fast_io import FastColumnSampler
    from flymon.frozen_t4 import FrozenT4Readout
    from flymon.evolution import T4Pooling
    ctx, const = E22.contexts()
    t4 = FrozenT4Readout(ctx, const)
    side = ctx["meta"]["side"][t4.brain_index].astype(str)
    rf = np.concatenate([ctx["rf_uv"][s] for s in ("T4a", "T4b", "T4c", "T4d")])
    _W.update(ctx=ctx, const=const, t4=t4, sampler=FastColumnSampler(t4._records, t4._colindex, t4._uv),
              pooling=T4Pooling(t4.subtype, side, rf), game=PokemonEmulator(), shuffled={}, brain=None,
              native_hash=t4.sha256())


def _brain():
    if _W["brain"] is None:
        import run_generation_zero as G
        from flymon.fast_io import VectorInjector
        brain, pops, proj = G.brain_setup()
        _W["brain"] = (brain, pops, proj, VectorInjector(brain))
    return _W["brain"]


def _readout(vision, seed):
    if vision == "none":
        return None
    if vision == "normal":
        return _W["t4"]
    if seed not in _W["shuffled"]:
        import run_fixed_background_experiment as E22
        from flymon.frozen_t4 import FrozenT4Readout
        proj = E22.permuted_projections(_W["ctx"], 1, SHUFFLE_BASE + seed)[0]
        _W["shuffled"] = {seed: FrozenT4Readout(_W["ctx"], _W["const"], proj=proj,
                                                geometry_label=f"column-shuffle-{SHUFFLE_BASE + seed}")}
    return _W["shuffled"][seed]


def _window(brain, vi, prepared, populations, cache):
    slots = cache.setdefault("motor_slots", {})
    if not slots:
        for typ, idx in populations.items():
            slot = np.full(brain.n, -1, np.int32); slot[idx] = np.arange(len(idx)); slots[typ] = slot
    per_cell = {typ: np.zeros(len(idx), np.int32) for typ, idx in populations.items()}
    for _ in range(NEURAL_STEPS):
        fired = vi.step(prepared)
        for typ, slot in slots.items():
            hit = slot[fired]; hit = hit[hit >= 0]
            per_cell[typ] += np.bincount(hit, minlength=len(per_cell[typ])).astype(np.int32)
    seconds = NEURAL_STEPS * float(brain.dt)
    rates = {}
    for typ, counts in per_cell.items():
        for s in ("L", "R"):
            rates[f"{typ}_{s}"] = float(counts[brain.side[populations[typ]] == s].sum() / seconds)
        rates[typ] = float(counts.sum() / seconds)
    return {"per_cell_counts": {k: v.tolist() for k, v in per_cell.items()}, "controller_rates_hz": rates}


def run_task(task):
    """One evaluation episode. Returns evaluator metrics; never feeds them back."""
    from flymon import gameplay_eval as ev
    from flymon.evolution import ACTIONS, FeatureBuilder, FitnessTracker, Genome, Policy, DN_KEYS
    from flymon.frozen_t4 import T4Injection
    import run_generation_zero as G
    t0 = time.perf_counter(); timers = dict(emulator=0.0, t4=0.0, brain=0.0, controller=0.0)
    kind, seed, budget, vision = task["kind"], task["seed"], task["budget"], task.get("vision", "normal")
    ablate = task.get("ablate")
    genome = Genome.from_json(task["genome"]) if kind == "genome" else None
    arch = genome.arch if genome else None
    uses_brain = kind == "gen0" or arch in ("dn", "t4dn")
    readout = _readout(vision, seed) if (uses_brain or arch in ("t4", "t4dn")) else None
    game, sampler = _W["game"], _W["sampler"]
    game.load_state(G.STATE); game.tick(1)
    wait = int(np.random.default_rng([seed, 2500]).integers(0, START_WAIT_MAX))
    frames = [game.framebuffer()]
    for _ in range(wait):
        game.tick(1); frames.append(game.framebuffer())
    tel0 = ev.read_telemetry(game)
    tracker = FitnessTracker((tel0["map"], tel0["x"], tel0["y"]))
    rng = np.random.default_rng([seed, 2501])
    buf = []
    if readout is not None:
        readout.reset()
        t = time.perf_counter()
        for f in frames:
            buf.append(readout.update_from_luminance(sampler.luminance(f)))
        timers["t4"] += time.perf_counter() - t
    if uses_brain:
        brain, pops, proj, vi = _brain()
        from run_population_event_experiment import active_record, baseline_trial, population_config
        # E14's per-episode no-vision baseline (brain.reset(seed) + 10 windows), used both for the
        # frozen E14 controller (gen0) and to z-score DN features
        vbase, _, rate_windows, cell_windows, _, cache = baseline_trial(brain, proj, pops, seed)
        bmean = {k: float(np.mean([w[k] for w in rate_windows])) for k in DN_KEYS}
        bsd = {k: float(np.std([w[k] for w in rate_windows])) for k in DN_KEYS}
        from run_progression_experiment import load_frozen_configuration
        _dc, _es, floor = load_frozen_configuration()
        if kind == "gen0":
            from flymon.exploration import FrozenExplorationController, FrozenInterfaceController
            from flymon.interaction import PopulationEventController
            from flymon.locomotion import estimate_population_baseline
            from flymon.steering import estimate_baseline
            direction = FrozenExplorationController(estimate_baseline(rate_windows),
                                                    estimate_population_baseline(rate_windows, cell_windows),
                                                    _dc, enable_down=False)
            controller = FrozenInterfaceController(
                direction, None, PopulationEventController(vbase, population_config(_es, floor), ablated=False))
        injection = T4Injection(_W["t4"].brain_index, G.INJECTION_GAIN, G.INJECTION_CAP, G.INJECTION_LEVELS)
        guard = None
    if kind == "genome":
        fb = FeatureBuilder(arch, _W["pooling"]); policy = Policy(genome, rng)
    elif kind == "random_matched":
        law = json.loads((ROOT / "results/experiment-24-generation-zero/c0-action-law.json").read_text())["law"]
        labels = list(law); probs = np.array([law[k] for k in labels]); probs /= probs.sum()
    log = [] if task.get("log") else None
    for d in range(budget):
        t4_mean = np.mean(buf, axis=0) if buf else None
        buf = []
        dnvec = None
        if uses_brain:
            t = time.perf_counter()
            pairs = injection.pairs(t4_mean) if t4_mean is not None else ()
            neural = _window(brain, vi, vi.prepare(pairs), pops, cache)
            pop = active_record(neural, vbase, floor)
            rates = neural["controller_rates_hz"] | {"event_population_rates_hz": pop["rates_hz"]}
            if guard is None:
                guard = ev.ControllerInputGuard(rates.keys())
            rates = guard.check(rates)
            timers["brain"] += time.perf_counter() - t
            p20 = float(np.sqrt(np.mean(np.square(pop["z"]))))
        t = time.perf_counter()
        if kind == "genome":
            if arch in ("dn", "t4dn"):
                dnvec = fb.dn_vector(rates, bmean, bsd, p20)
                if ablate == "zero_dn":
                    dnvec = np.zeros_like(dnvec)
            t4_in = None if ablate == "zero_t4" else t4_mean
            x = fb.build(t4_response=t4_in, dn=dnvec)
            action = policy.act(x); fb.observe_action(action)
        elif kind == "gen0":
            action, _state = controller.decode(rates)
        elif kind == "random_matched":
            a = labels[int(rng.choice(len(labels), p=probs))]; action = None if a == "NONE" else a
        elif kind == "random_uniform":
            a = ACTIONS[int(rng.integers(0, len(ACTIONS)))]; action = None if a == "NONE" else a
        else:
            raise ValueError(kind)
        timers["controller"] += time.perf_counter() - t
        if action in ("START", "SELECT"):
            raise RuntimeError("START/SELECT are disabled")
        if action is not None:
            game.press(action.lower())
        for k in range(HOLD + RELEASE):
            if k == HOLD and action is not None:
                game.release(action.lower())
            t = time.perf_counter(); game.tick(1); timers["emulator"] += time.perf_counter() - t
            if readout is not None:
                t = time.perf_counter()
                buf.append(readout.update_from_luminance(sampler.luminance(game.framebuffer())))
                timers["t4"] += time.perf_counter() - t
        tel = ev.read_telemetry(game)                     # evaluator only, after the action
        tracker.update(d, action, tel)
        if log is not None:
            log.append((d, action, tel["map"], tel["x"], tel["y"], int(tel["window"]), tel["battle"]))
    res = tracker.result()
    res.update(task={k: v for k, v in task.items() if k != "genome"},
               genome_sha256=genome.sha256() if genome else None, start_wait_frames=wait,
               sensory_sha256=readout.sha256() if readout is not None else None,
               wall_s=time.perf_counter() - t0, stage_s=timers)
    if log is not None:
        res["log"] = log
    return res


# ================================================================ driver side
def atomic_json(path, obj, gz=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(obj, default=str).encode() if gz else (json.dumps(obj, indent=2, default=str) + "\n").encode()
    if gz:
        with gzip.open(tmp, "wb") as f:
            f.write(data)
    else:
        tmp.write_bytes(data)
    tmp.replace(path)


def pool():
    ctx = mp.get_context("spawn")
    return ctx.Pool(WORKERS, initializer=_worker_init)


def evaluate(p, tasks):
    return p.map(run_task, tasks, chunksize=1)


def brief(r):
    return {k: r[k] for k in ("fitness", "unique_tiles", "unique_maps", "idle_fraction", "loop_fraction_nonidle",
                              "completed_window_cycles", "dialogue_locks", "successful_moves", "revisit_rate")} | dict(
        milestones={k: v for k, v in r["milestones"].items() if v is not None}, actions=r["action_counts"],
        seed=r["task"]["seed"], wall_s=round(r["wall_s"], 2), terms=r["terms"])


def run_lineage(phase, arch, population, generations, rng_seed, p, start_from=None, early_stop=None):
    """Mutation + elitism. Resumable from the last checkpoint of this phase."""
    from flymon.evolution import Genome, next_population, rank
    ckdir = CHECKPOINTS / phase
    existing = sorted(ckdir.glob("gen-*.json.gz"))
    summaries_path = RESULTS / f"{phase}-generations.jsonl"
    if existing:
        ck = json.loads(gzip.open(existing[-1]).read())
        gen = ck["generation"] + 1
        pop = [Genome.from_json(g) for g in ck["next_population"]]
        rng = np.random.default_rng(); rng.bit_generator.state = ck["rng_state"]
        history = ck["history"]
        print(f"[{phase}] resume at generation {gen}", flush=True)
    else:
        rng = np.random.default_rng(rng_seed)
        if start_from:
            # continue a screen lineage: keep its elites, fill with their mutants and immigrants
            pop, _ = next_population([Genome.from_json(g) for g in start_from], rng, population, arch=arch)
        else:
            pop = [Genome.random(arch, rng) for _ in range(population)]
        gen, history = 0, dict(score=[], best=[])
        summaries_path.unlink(missing_ok=True)
    while gen < generations:
        t0 = time.perf_counter()
        seeds = SEED_PAIRS[gen % len(SEED_PAIRS)]
        tasks = [dict(kind="genome", genome=g.to_json(), seed=s, budget=TRAIN_BUDGET, vision="normal")
                 for g in pop for s in seeds]
        res = evaluate(p, tasks)
        per = {}
        for r in res:
            per.setdefault(r["genome_sha256"], []).append(r)
        fitness = [float(np.mean([r["fitness"] for r in per[g.sha256()]])) for g in pop]
        ranked, order = rank(pop, fitness)
        fs = np.array(fitness)
        top = np.sort(fs)[::-1][:max(1, len(fs) // 4)]
        score = float(top.mean())
        history["score"].append(score); history["best"].append(float(fs.max()))
        ms_rate = {m: float(np.mean([r["milestones"][m] is not None for r in res]))
                   for m in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")}
        summary = dict(phase=phase, arch=arch, generation=gen, seeds=list(seeds), population=len(pop),
                       best=float(fs.max()), median=float(np.median(fs)), worst=float(fs.min()), top_quartile_mean=score,
                       best_genome=ranked[0].sha256(), milestone_rate=ms_rate,
                       any_down=bool(any(r["action_counts"]["DOWN"] > 0 for r in res)),
                       any_completed_interaction=bool(any(r["completed_window_cycles"] > 0 for r in res)),
                       mean_unique_tiles=float(np.mean([r["unique_tiles"] for r in res])),
                       mean_unique_maps=float(np.mean([r["unique_maps"] for r in res])),
                       mean_idle=float(np.mean([r["idle_fraction"] for r in res])),
                       mean_loop=float(np.mean([r["loop_fraction_nonidle"] for r in res])),
                       action_fraction={a: float(np.mean([r["action_counts"][a] / r["decisions"] for r in res]))
                                        for a in res[0]["action_counts"]},
                       runtime_s=time.perf_counter() - t0, episodes=len(res),
                       sensory_sha256=sorted({r["sensory_sha256"] for r in res if r["sensory_sha256"]}))
        summaries_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summaries_path, "a") as f:
            f.write(json.dumps(summary) + "\n")
        nxt, n_elite = next_population(ranked, rng, population, arch=arch)
        atomic_json(ckdir / f"gen-{gen:03d}.json.gz", dict(
            phase=phase, arch=arch, generation=gen, seeds=list(seeds), population=[g.to_json() for g in pop],
            fitness=fitness, per_seed={h: [brief(r) for r in rs] for h, rs in per.items()},
            elites=[g.sha256() for g in ranked[:n_elite]], next_population=[g.to_json() for g in nxt],
            rng_state=rng.bit_generator.state, history=history, sensory_sha256=_native_hash()), gz=True)
        print(f"[{phase}] gen {gen:3d} seeds {seeds} best {fs.max():8.1f} median {np.median(fs):8.1f} "
              f"topQ {score:8.1f} M2 {ms_rate['M2']:.2f} M3 {ms_rate['M3']:.2f} M5 {ms_rate['M5']:.2f} "
              f"M7 {ms_rate['M7']:.2f} tiles {summary['mean_unique_tiles']:.1f} {summary['runtime_s']:.0f}s", flush=True)
        pop = nxt; gen += 1
        if early_stop and gen >= early_stop["after"]:
            s = history["score"]
            best_i = int(np.argmax(s))
            if len(s) - 1 - best_i >= early_stop["patience"]:
                print(f"[{phase}] early stop at generation {gen - 1}", flush=True)
                break
    return history


def _native_hash():
    p = RESULTS / "sensory-hash.json"
    return json.loads(p.read_text())["native_sha256"] if p.exists() else None


def stage_screen(archs):
    with pool() as p:
        for i, arch in enumerate(archs):
            run_lineage(f"screen-{arch}", arch, SCREEN["population"], SCREEN["generations"], 25_000 + i, p)
    summary = {}
    for arch in ("dn", "t4", "t4dn"):
        path = RESULTS / f"screen-{arch}-generations.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        summary[arch] = dict(score=float(np.mean([r["top_quartile_mean"] for r in rows[-3:]])),
                             best=max(r["best"] for r in rows), median_last=rows[-1]["median"],
                             best_milestone_rates={m: max(r["milestone_rate"][m] for r in rows)
                                                   for m in ("M2", "M3", "M4", "M5", "M7", "M8")},
                             generations=len(rows))
    if len(summary) == 3:
        winner = max(summary, key=lambda a: summary[a]["score"])
        atomic_json(RESULTS / "architecture-screen.json", dict(
            rule="mean over the last 3 screen generations of the top-quartile mean training fitness",
            architectures=summary, winner=winner))
        print("screen winner", winner, {a: round(v["score"], 1) for a, v in summary.items()}, flush=True)


def stage_main():
    screen = json.loads((RESULTS / "architecture-screen.json").read_text())
    arch = screen["winner"]
    last = sorted((CHECKPOINTS / f"screen-{arch}").glob("gen-*.json.gz"))[-1]
    ck = json.loads(gzip.open(last).read())
    ranked = [g for _, g in sorted(zip(ck["fitness"], ck["population"]), key=lambda t: (-t[0], t[1]["sha256"]))]
    with pool() as p:
        run_lineage("main", arch, MAIN["population"], MAIN["generations"], 25_100, p, start_from=ranked,
                    early_stop=dict(after=MAIN["early_stop_after"], patience=MAIN["patience"]))


def stage_finalists():
    cks = sorted((CHECKPOINTS / "main").glob("gen-*.json.gz"))
    last3 = [json.loads(gzip.open(c).read()) for c in cks[-3:]]
    evals, genomes = {}, {}
    for ck in last3:
        for g in ck["population"]:
            genomes[g["sha256"]] = g
        for h, rs in ck["per_seed"].items():
            evals.setdefault(h, []).extend(r["fitness"] for r in rs)
    multi = [h for h in evals if len(evals[h]) >= 4]
    multi.sort(key=lambda h: (-float(np.mean(evals[h])), h))
    chosen = multi[:8]
    if len(chosen) < 8:
        rest = [h for h in sorted(evals, key=lambda h: (-float(np.mean(evals[h])), h)) if h not in chosen]
        chosen += rest[:8 - len(chosen)]
    fin = [dict(rank=i + 1, sha256=h, training_evaluations=len(evals[h]),
                training_mean_fitness=float(np.mean(evals[h])), genome=genomes[h]) for i, h in enumerate(chosen)]
    atomic_json(RESULTS / "finalists.json", dict(
        rule="genomes evaluated on >= 2 seed pairs (>= 4 episodes) in the last 3 main generations, ranked by "
             "mean training fitness there; filled from the rest if fewer than 8", finalists=fin))
    print([(f["rank"], f["sha256"][:10], f["training_evaluations"], round(f["training_mean_fitness"], 1)) for f in fin])


def stage_heldout():
    fin = json.loads((RESULTS / "finalists.json").read_text())["finalists"]
    screen = json.loads((RESULTS / "architecture-screen.json").read_text())
    arch = screen["winner"]
    tasks = []
    for f in fin:
        for s in HELDOUT_SEEDS:
            for v in ("normal", "none", "shuffled"):
                tasks.append(dict(kind="genome", genome=f["genome"], seed=s, budget=HELDOUT_BUDGET, vision=v,
                                  role="finalist", rank=f["rank"]))
    for s in HELDOUT_SEEDS:
        for k in ("gen0", "random_matched", "random_uniform"):
            tasks.append(dict(kind=k, seed=s, budget=HELDOUT_BUDGET, vision="normal", role=k))
        if arch == "t4dn":
            for ab in ("zero_dn", "zero_t4"):
                tasks.append(dict(kind="genome", genome=fin[0]["genome"], seed=s, budget=HELDOUT_BUDGET,
                                  vision="normal", ablate=ab, role=f"flybrain-{ab}", rank=1))
    # best final-generation genome of each non-winning screen architecture (training-only choice)
    for other in ("dn", "t4", "t4dn"):
        if other == arch:
            continue
        ck = json.loads(gzip.open(sorted((CHECKPOINTS / f"screen-{other}").glob("gen-*.json.gz"))[-1]).read())
        best = max(zip(ck["fitness"], ck["population"]), key=lambda t: (t[0], t[1]["sha256"]))[1]
        for s in HELDOUT_SEEDS:
            tasks.append(dict(kind="genome", genome=best, seed=s, budget=HELDOUT_BUDGET, vision="normal",
                              role=f"screen-best-{other}"))
    t0 = time.perf_counter()
    with pool() as p:
        res = []
        for i, r in enumerate(p.imap_unordered(run_task, tasks, chunksize=1)):
            res.append(r)
            if (i + 1) % 40 == 0:
                print(f"[heldout] {i + 1}/{len(tasks)} {time.perf_counter() - t0:.0f}s", flush=True)
    atomic_json(RESULTS / "heldout-raw.json.gz", dict(results=res, seeds=list(HELDOUT_SEEDS), budget=HELDOUT_BUDGET,
                                                      wall_s=time.perf_counter() - t0), gz=True)
    print(f"held-out done {time.perf_counter() - t0:.0f}s", flush=True)


def stage_champion_log():
    """Replayable per-decision log of the selected champion on every held-out seed (normal vision)."""
    ch = json.loads((RESULTS / "champion-evaluation.json").read_text())
    g = next(f["genome"] for f in json.loads((RESULTS / "finalists.json").read_text())["finalists"]
             if f["sha256"] == ch["champion"]["sha256"])
    tasks = [dict(kind="genome", genome=g, seed=s, budget=HELDOUT_BUDGET, vision="normal", log=True, role="champion")
             for s in HELDOUT_SEEDS]
    with pool() as p:
        res = p.map(run_task, tasks, chunksize=1)
    atomic_json(RESULTS / "champion-heldout-logs.json.gz",
                {str(r["task"]["seed"]): dict(fitness=r["fitness"], log=r["log"]) for r in res}, gz=True)


def stage_throughput():
    from flymon.evolution import Genome
    g = {a: Genome.random(a, np.random.default_rng(1)).to_json() for a in ("dn", "t4", "t4dn")}
    out = {}
    for n in (1, 4, 8):
        global WORKERS
        WORKERS = n
        for a in ("t4", "t4dn"):
            tasks = [dict(kind="genome", genome=g[a], seed=2501 + i, budget=200) for i in range(n * 2)]
            with pool() as p:
                p.map(run_task, [dict(tasks[0], budget=5)] * n)          # warm-up / worker init
                t0 = time.perf_counter(); res = p.map(run_task, tasks, chunksize=1); wall = time.perf_counter() - t0
            out[f"{a}-{n}"] = dict(workers=n, arch=a, episodes=len(tasks), decisions=200 * len(tasks), wall_s=wall,
                                   aggregate_decisions_per_s=200 * len(tasks) / wall,
                                   per_episode_stage_ms_per_decision={k: 1e3 * float(np.mean([r["stage_s"][k] for r in res])) / 200
                                                                      for k in res[0]["stage_s"]})
            print(a, n, round(out[f"{a}-{n}"]["aggregate_decisions_per_s"], 1), flush=True)
    atomic_json(RESULTS / "throughput.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["screen", "main", "finalists", "heldout", "champion-log", "throughput", "smoke"])
    ap.add_argument("--archs", nargs="+", default=["dn", "t4", "t4dn"])
    a = ap.parse_args()
    if a.stage == "screen":
        stage_screen(a.archs)
    elif a.stage == "main":
        stage_main()
    elif a.stage == "finalists":
        stage_finalists()
    elif a.stage == "heldout":
        stage_heldout()
    elif a.stage == "champion-log":
        stage_champion_log()
    elif a.stage == "throughput":
        stage_throughput()
    elif a.stage == "smoke":
        global SCREEN, TRAIN_BUDGET
        SCREEN = dict(population=6, generations=2); TRAIN_BUDGET = 60
        global RESULTS, CHECKPOINTS
        RESULTS = ROOT / "results" / "experiment-25-evolution" / "smoke"; CHECKPOINTS = RESULTS / "checkpoints"
        stage_screen(a.archs)


if __name__ == "__main__":
    main()
