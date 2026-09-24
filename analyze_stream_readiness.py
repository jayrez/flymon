"""E30 analysis: soak summary, stream-readiness gates and 12 figures (captures/e30-stream-readiness/).

    python analyze_stream_readiness.py soak --dir <soak dir>     # -> soak-test-results.json
    python analyze_stream_readiness.py t4-example                  # 300-decision champion T4 trace (needs POKEMON_ROM)
    python analyze_stream_readiness.py gates                       # -> stream-readiness-gates.json
    python analyze_stream_readiness.py figures
"""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from pathlib import Path

import numpy as np

from analyze_tonic_disinhibition_experiment import bars, panel_traces

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "e30-stream-readiness"
C = ROOT / "captures" / "e30-stream-readiness"
MS = ("bedroom", "bedroom_exited", "house_exited", "pallet_explored", "oak_lab", "route1", "battle")
load = lambda n: json.loads((OUT / n).read_text()) if (OUT / n).exists() else None


def jdump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float) + "\n")


# ---------------------------------------------------------------- soak summary
def stage_soak(d):
    d = Path(d)
    db = sqlite3.connect(str(d / "rt" / "stream-state.sqlite"))
    eps = [dict(zip(("id", "seed", "started", "ended", "decisions", "fitness", "furthest", "end_reason", "watchdog", "wall_s"), r))
           for r in db.execute("SELECT id, seed, started, ended, decisions, fitness, furthest, end_reason, watchdog, wall_s "
                               "FROM episodes ORDER BY id")]
    events = [dict(ts=r[0], kind=r[1], detail=json.loads(r[2])) for r in db.execute("SELECT ts, kind, detail FROM events ORDER BY id")]
    ts = [json.loads(x) for x in (d / "timeseries.jsonl").read_text().splitlines()]
    start = next(e["ts"] for e in events if e["kind"] == "supervisor_start")
    stop = next((e for e in events if e["kind"] == "supervisor_stop"), None)
    uptime = stop["detail"]["uptime_s"] if stop else ts[-1]["ts"] - start
    done = [e for e in eps if e["end_reason"] == "budget"]
    interrupts = sorted([start] + [e["ts"] for e in events if e["kind"] in ("worker_crash", "watchdog_restart_hang",
                                                                            "watchdog_restart_startup_timeout")] + [start + uptime])
    longest = float(max(b - a for a, b in zip(interrupts, interrupts[1:])))
    faults = [e for e in events if e["kind"] == "fault_injected"]
    wd = [e for e in events if e["kind"].startswith("watchdog_")]
    rss = [(r["ts"] - start, r["worker_rss_mb"]) for r in ts if r["worker_rss_mb"]]
    first_h = [v for t, v in rss if 600 <= t < 3600]; last_h = [v for t, v in rss if t >= uptime - 3600]
    dps = [r["decisions_per_second"] for r in ts if r["decisions_per_second"]]
    gpu = [r["gpu"] for r in ts if r.get("gpu")]
    cpu = [r["worker_cpu_pct"] for r in ts if r.get("worker_cpu_pct") is not None]
    prog = {m: sum(e["furthest"] == m for e in done) for m in MS}
    hours = uptime / 3600
    per_hour = [sum(1 for e in done if start + 3600 * h <= e["ended"] < start + 3600 * (h + 1)) for h in range(int(np.ceil(hours)))]
    recovered = all(any(w["ts"] > f["ts"] and w["kind"].startswith("watchdog_restart") for w in wd) for f in faults)
    res = dict(
        plan=load("runtime-config.json")["soak_plan"], uptime_s=uptime, uptime_h=hours,
        episodes_started=len(eps), episodes_completed=len(done), completed_per_hour=per_hour,
        mean_episode_s=float(np.mean([e["wall_s"] for e in done])) if done else None,
        episode_end_reasons={r: sum(e["end_reason"] == r for e in eps) for r in sorted({str(e["end_reason"]) for e in eps})},
        crashes=sum(e["kind"] == "worker_crash" for e in events), restarts=sum(e["kind"] == "worker_restart" for e in events),
        watchdog_interventions=len(wd), watchdog_events=[dict(t_s=e["ts"] - start, kind=e["kind"], detail=e["detail"]) for e in wd],
        faults_injected=[dict(t_s=e["ts"] - start, **e["detail"]) for e in faults], all_faults_recovered=bool(recovered),
        unplanned_crashes_or_hangs=max(0, sum(e["kind"] in ("worker_crash",) for e in events) + sum(e["kind"] == "watchdog_restart_hang"
                                                                                                    for e in events) - len(faults)),
        longest_uninterrupted_s=longest, progress_distribution=prog,
        progress_fraction={m: prog[m] / len(done) for m in MS} if done else None,
        reached_at_least={m: sum(MS.index(e["furthest"]) >= MS.index(m) for e in done) / len(done) for m in MS} if done else None,
        fitness_median=float(np.median([e["fitness"] for e in done])) if done else None,
        worker_rss_mb=dict(first_hour_median=float(np.median(first_h)) if first_h else None,
                           last_hour_median=float(np.median(last_h)) if last_h else None, max=max(v for _, v in rss)),
        supervisor_rss_mb_max=max(r["supervisor_rss_mb"] for r in ts if r["supervisor_rss_mb"]),
        worker_cpu_pct_median=float(np.median(cpu)) if cpu else None,
        gpu=dict(util_pct_max=max(g["util_pct"] for g in gpu), mem_used_mb_max=max(g["mem_used_mb"] for g in gpu)) if gpu else None,
        decisions_per_second=dict(median=float(np.median(dps)), p5=float(np.percentile(dps, 5)), min=float(min(dps))),
        speed_ratio_median=float(np.median([r["speed_ratio"] for r in ts if r["speed_ratio"]])),
        samples=len(ts))
    growth = (res["worker_rss_mb"]["last_hour_median"] or 0) - (res["worker_rss_mb"]["first_hour_median"] or 0)
    res["worker_rss_growth_mb"] = growth
    res["gate_S3"] = dict(full_duration=hours >= res["plan"]["duration_h"] - 0.01, all_faults_recovered=res["all_faults_recovered"],
                          no_unplanned_failures=res["unplanned_crashes_or_hangs"] == 0,
                          completed_episode_every_hour=all(n >= 1 for n in per_hour[:int(hours)]),
                          memory_growth_below_200mb=growth < 200)
    res["gate_S3"]["pass"] = bool(all(v for v in res["gate_S3"].values()))
    jdump("soak-test-results.json", res)
    (OUT / "soak-timeseries.json").write_text(json.dumps(dict(start=start, rows=ts, episodes=eps)) + "\n")
    print(json.dumps({k: res[k] for k in ("uptime_h", "episodes_completed", "crashes", "restarts", "watchdog_interventions",
                                          "longest_uninterrupted_s", "progress_distribution", "worker_rss_mb", "gate_S3")}, indent=1))


# ---------------------------------------------------------------- T4 activity example
def stage_t4_example():
    from flymon.emulator import PokemonEmulator
    from flymon.live_runtime import Episode, FrozenController, SensoryContext, load_champion
    from flymon.stream_eval import StreamEvaluator
    with PokemonEmulator() as game:
        ctl = FrozenController(load_champion(), SensoryContext(), "normal")
        ep = Episode(game, ctl, 2601, 300, evaluator_factory=StreamEvaluator).start()
        rows = []
        while not ep.done:
            d, tel, snap = ep.step()
            rows.append(dict(t4=d.t4_summary, action=d.action or "NONE", p=d.probabilities, map=snap["map_name"]))
    jdump("t4-activity-example.json", dict(seed=2601, decisions=len(rows), rows=rows,
                                           note="champion, normal vision, first 300 decisions of held-out seed 2601"))


# ---------------------------------------------------------------- live observability check (against a running stream)
def stage_observe(port):
    import time
    import urllib.error
    import urllib.request
    from flymon.telemetry import validate
    base = f"http://127.0.0.1:{port}"
    ep = {}
    for path in ("/state", "/schema", "/provenance", "/frame.jpg", "/overlay", "/dashboard", "/static/app.js"):
        try:
            r = urllib.request.urlopen(base + path, timeout=10); body = r.read(); ep[path] = r.status == 200 and len(body) > 0
        except Exception:
            ep[path] = False
    r = urllib.request.urlopen(base + "/stream.mjpg", timeout=10); chunk = r.read(4096); r.close()
    ep["/stream.mjpg"] = b"--frame" in chunk and b"image/jpeg" in chunk
    r = urllib.request.urlopen(base + "/events", timeout=10); msg = None
    t0 = time.time()
    while time.time() - t0 < 10:
        line = r.readline()
        if line.startswith(b"data:"):
            m = json.loads(line[5:])
            if m.get("type") == "decision":
                msg = m; break
    r.close(); ep["/events"] = msg is not None
    try:
        urllib.request.urlopen(urllib.request.Request(base + "/state", data=b"{}", method="POST"), timeout=5); post = False
    except urllib.error.HTTPError as e:
        post = e.code == 405
    required = {"current action": "action", "recent action history": "recent_actions", "run seed": "seed", "episode": "episode",
                "runtime": "uptime_s", "decisions made": "decision", "decisions/sec": "decisions_per_second",
                "visual pathway status": "visual_pathway", "T4 activity summary": "t4",
                "controller feature activity": ("controller", "features"), "controller output scores": ("controller", "probabilities"),
                "current milestone": ("milestone", "current"), "furthest milestone this episode": ("milestone", "furthest_episode"),
                "furthest milestone all-time": ("milestone", "furthest_all_time"), "unique tiles": "unique_tiles",
                "watchdog status": "watchdog"}
    have = {}
    for k, path in required.items():
        v = msg
        for part in (path if isinstance(path, tuple) else (path,)):
            v = v.get(part) if isinstance(v, dict) else None
        have[k] = v is not None
    ok = True
    try:
        validate(msg)
    except Exception:
        ok = False
    jdump("observability-check.json", dict(checked_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), port=port,
                                            endpoints=ep, post_rejected=post, message_validates=ok, required_fields=have,
                                            required_fields_ok=all(have.values()), framebuffer="MJPEG /stream.mjpg + /frame.jpg "
                                            "(not inside JSON)", sample_message=msg))
    print(json.dumps(dict(endpoints=ep, post_rejected=post, validates=ok, required=all(have.values())), indent=1))


# ---------------------------------------------------------------- gates
def stage_gates():
    rep, vis, soak = load("reproduction-results.json"), load("vision-ablation-results.json"), load("soak-test-results.json")
    obs = load("observability-check.json")
    s4 = dict(schema_fields_present=bool(obs and obs.get("message_validates")), endpoints=bool(obs and all(obs["endpoints"].values())),
              broadcast_and_research_pages=bool(obs and obs["endpoints"].get("/overlay") and obs["endpoints"].get("/dashboard")),
              overlay_read_only=bool(obs and obs.get("post_rejected")),
              required_fields=bool(obs and obs.get("required_fields_ok")))
    s4["pass"] = bool(all(s4.values()))
    prov = load("scientific-provenance.json")
    s5 = dict(provenance_json=prov is not None, overlay_panel="provenance" in (ROOT / "overlay/dashboard.html").read_text(),
              broadcast_footer="No learning happens live" in (ROOT / "overlay/overlay.html").read_text(),
              brain_sim_disclosed=bool(prov and "NOT run" in prov["brain_simulation"]),
              readme_boundary="never enters the controller" in (ROOT / "README.md").read_text())
    s5["pass"] = bool(all(s5.values()))
    g = dict(S1=rep["gate_S1"], S2=vis["gate_S2"], S3=soak["gate_S3"] if soak else dict(pass_=False, note="soak not run"),
             S4=s4, S5=s5)
    passed = [k for k in g if g[k].get("pass")]
    weak = True                                                  # gameplay: never Route 1 / battle; p = 0.06 vs uniform
    verdict = ("STREAM READY WITH CAVEATS" if len(passed) == 5 and weak else "STREAM READY" if len(passed) == 5 else "NOT STREAM READY")
    g.update(passed=passed, verdict=verdict,
             caveats=["gameplay weak: never Route 1 or a battle; loops on Oak's Lab starter / exit dialogue",
                      "not conventionally better than uniform-random buttons on aggregate fitness (p = 0.06)",
                      "3/8 E25 finalists were not vision-dependent (the champion is)"])
    jdump("stream-readiness-gates.json", g)
    print(json.dumps({k: g[k] if k in ("passed", "verdict") else g[k].get("pass") for k in list(g)[:5] + ["passed", "verdict"]}, indent=1))


# ---------------------------------------------------------------- figures
def stage_figures():
    C.mkdir(parents=True, exist_ok=True)
    rep, vis, dia = load("reproduction-results.json"), load("vision-ablation-results.json"), load("dialogue-stall-analysis.json")
    rows = rep["conditions"]["champion|normal"]["rows"]
    bars(C / "01-historical-vs-recovered-fitness.svg", [f"{r['seed']} {w}" for r in rows for w in ("E25", "E30")],
         [v for r in rows for v in (r["historical"], r["recovered"])],
         "E30 recovered champion vs historical E25, per held-out seed (identical on 20/20)")
    conds = list(vis["conditions"])
    ms = ("bedroom_exit", "house_exit", "oak_lab", "route1", "battle")
    bars(C / "02-heldout-milestone-rates.svg", [f"{c} {m}" for c in conds for m in ms],
         [vis["conditions"][c][m] for c in conds for m in ms], "E30 held-out milestone rates (20 seeds x 1,500 decisions)")
    bars(C / "03-intact-vs-controls.svg", conds, [vis["conditions"][c]["median"] for c in conds],
         "E30 median held-out fitness: intact champion vs no vision / shuffled T4 / random baselines")
    acts = ("NONE", "UP", "DOWN", "LEFT", "RIGHT", "A", "B")
    bars(C / "04-action-distribution.svg", [f"{c} {a}" for c in conds for a in acts],
         [vis["conditions"][c]["action_fraction"][a] for c in conds for a in acts], "E30 action frequencies per condition")
    bars(C / "05-dialogue-scripted-vs-stall.svg", [f"{c} {k}" for c in dia["conditions"] for k in ("scripted", "navigable", "stall")],
         [dia["conditions"][c]["decisions"][k] for c in dia["conditions"] for k in ("scripted", "navigable", "stall")],
         "E30 decisions inside long (>= 30) text/menu windows by class, 20 seeds")
    soak = load("soak-test-results.json")
    if soak:
        s = json.loads((OUT / "soak-timeseries.json").read_text()); t0 = s["start"]
        done = [e for e in s["episodes"] if e["end_reason"] == "budget"]
        bars(C / "06-soak-milestone-per-episode.svg", [f"ep {e['id']} ({e['seed']})" for e in done],
             [MS.index(e["furthest"]) for e in done], "E30 soak: furthest milestone per completed episode "
             "(0 bedroom, 1 left bedroom, 2 left house, 3 Pallet explored, 4 Oak's Lab, 5 Route 1, 6 battle)")
        hrs = np.array([(r["ts"] - t0) / 3600 for r in s["rows"]])
        ser = lambda k, f=lambda v: v: np.array([f(r[k]) if r.get(k) is not None else np.nan for r in s["rows"]], float)
        panel_traces(C / "07-soak-decisions-per-second.svg", [("decisions / s", [("dps", ser("decisions_per_second"), "#275e85")]),
                                                              ("speed ratio", [("x real time", ser("speed_ratio"), "#c0392b")])],
                     f"E30 soak: decisions/s and real-time ratio over {soak['uptime_h']:.2f} h (x = 30 s samples)")
        panel_traces(C / "08-soak-gpu-utilisation.svg", [("GPU util %", [("util", ser("gpu", lambda g: g["util_pct"]), "#275e85")]),
                                                          ("GPU mem MB", [("mem", ser("gpu", lambda g: g["mem_used_mb"]), "#c0392b")]),
                                                          ("worker CPU %", [("cpu", ser("worker_cpu_pct"), "#1e8449")])],
                     "E30 soak: GPU utilisation / memory (runtime is CPU-only) and worker CPU")
        panel_traces(C / "09-soak-memory.svg", [("worker RSS MB", [("worker", ser("worker_rss_mb"), "#275e85")]),
                                                ("supervisor RSS MB", [("sup", ser("supervisor_rss_mb"), "#c0392b")])],
                     "E30 soak: memory over time")
        ev = soak["watchdog_events"]
        bars(C / "10-watchdog-interventions.svg", [f"{e['t_s'] / 3600:.2f} h {e['kind']}" for e in ev] or ["none"],
             [1] * len(ev) or [0], f"E30 soak: watchdog interventions ({len(ev)}; faults injected at "
             f"{', '.join(str(int(f['t_s'])) + ' s ' + f['kind'] for f in soak['faults_injected'])})")
    t4 = load("t4-activity-example.json")
    if t4:
        rr = t4["rows"]
        panel_traces(C / "11-t4-activity-example.svg",
                     [(s_, [(s_, np.array([r["t4"][s_] for r in rr]), c)]) for s_, c in
                      zip(("T4a", "T4b", "T4c", "T4d"), ("#275e85", "#c0392b", "#1e8449", "#8e44ad"))] +
                     [("P(UP) / P(B)", [("UP", np.array([r["p"]["UP"] for r in rr]), "#275e85"),
                                        ("B", np.array([r["p"]["B"] for r in rr]), "#c0392b")])],
                     "E30: pooled T4 activity (what the controller reads) and two output probabilities, champion, seed 2601, "
                     "first 300 decisions")
    g = load("stream-readiness-gates.json")
    if g:
        bars(C / "12-stream-readiness-gates.svg", [f"{k} {'PASS' if g[k].get('pass') else 'FAIL'}" for k in ("S1", "S2", "S3", "S4", "S5")],
             [1 if g[k].get("pass") else 0 for k in ("S1", "S2", "S3", "S4", "S5")], f"E30 stream-readiness gates: {g['verdict']}")
    print(sorted(p.name for p in C.glob("*.svg")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["soak", "t4-example", "observe", "gates", "figures"])
    ap.add_argument("--dir")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.stage == "soak":
        stage_soak(a.dir)
    elif a.stage == "t4-example":
        stage_t4_example()
    elif a.stage == "observe":
        stage_observe(a.port)
    elif a.stage == "gates":
        stage_gates()
    else:
        stage_figures()


if __name__ == "__main__":
    main()
