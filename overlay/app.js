// Fly Brain Plays Pokémon — read-only telemetry client (SSE). Never sends anything back.
const ACTIONS = ["NONE", "UP", "DOWN", "LEFT", "RIGHT", "A", "B"];
const MILESTONES = ["bedroom", "bedroom_exited", "house_exited", "pallet_explored", "oak_lab", "route1", "battle"];
const LABEL = {bedroom: "Bedroom", bedroom_exited: "Left bedroom", house_exited: "Left house",
  pallet_explored: "Explored Pallet", oak_lab: "Oak's Lab", route1: "Route 1", battle: "Battle"};
const $ = (id) => document.getElementById(id);
const fmtTime = (s) => { s = Math.max(0, Math.floor(s || 0)); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
  return (h ? h + "h " : "") + String(m).padStart(h ? 2 : 1, "0") + "m " + String(s % 60).padStart(2, "0") + "s"; };

function bars(el, entries, max) {
  if (!el) return;
  el.innerHTML = entries.map(([k, v]) => {
    const w = Math.max(0, Math.min(100, 100 * v / (max || 1)));
    return `<div class="bar"><span class="k">${k}</span><span class="track"><span class="fill" style="width:${w}%"></span></span><span class="v">${v.toFixed(2)}</span></div>`;
  }).join("");
}

function milestoneStrip(el, furthest, allTime) {
  if (!el) return;
  const fi = MILESTONES.indexOf(furthest), ai = MILESTONES.indexOf(allTime);
  el.innerHTML = MILESTONES.map((m, i) =>
    `<span class="ms ${i <= fi ? "done" : ""} ${i === ai ? "best" : ""}">${LABEL[m]}</span>`).join("");
}

function set(id, text) { const e = $(id); if (e) e.textContent = text; }

function onDecision(m) {
  set("action", m.action); set("episode", m.episode); set("seed", m.seed); set("decision", m.decision);
  set("runtime", fmtTime(m.episode_runtime_s)); set("uptime", fmtTime(m.uptime_s));
  set("dps", (m.decisions_per_second || 0).toFixed(1)); set("speed", (m.speed_ratio || 0).toFixed(2) + "x");
  set("tiles", m.unique_tiles); set("mapname", m.game.map_name + (m.game.window ? " · text/menu" : ""));
  set("recent", m.recent_actions.join(" "));
  set("vision", `${m.visual_pathway.status} · ${m.visual_pathway.mode} · ${(m.visual_pathway.sensory_sha256 || "none").slice(0, 12)}`);
  set("ctlhash", (m.controller.sha256 || "").slice(0, 12)); set("frozen", m.controller.frozen ? "frozen" : "UNFROZEN");
  const wd = m.watchdog || {};
  set("watchdog", `${wd.healthy ? "healthy" : "attention"} · restarts ${wd.restarts ?? 0} · interventions ${wd.interventions ?? 0}`
    + (wd.last ? ` · last ${wd.last.kind}` : ""));
  milestoneStrip($("milestones"), m.milestone.furthest_episode, m.milestone.furthest_all_time);
  set("furthest", LABEL[m.milestone.furthest_episode] || m.milestone.furthest_episode);
  set("alltime", LABEL[m.milestone.furthest_all_time] || "—");
  const t4 = Object.entries(m.t4 || {}); const t4max = Math.max(0.5, ...t4.map(([, v]) => Math.abs(v)));
  bars($("t4"), t4.map(([k, v]) => [k, Math.abs(v)]), t4max);
  const probs = ACTIONS.map((a) => [a, m.controller.probabilities[a] || 0]);
  bars($("probs"), probs, 1.0);
  const feat = $("features");
  if (feat) {
    const f = Object.entries(m.controller.features);
    feat.innerHTML = f.map(([k, v]) => {
      const a = Math.min(1, Math.abs(v) / 3), col = v >= 0 ? `rgba(80,170,255,${a})` : `rgba(255,120,90,${a})`;
      return `<span class="cell" title="${k}: ${v.toFixed(3)}" style="background:${col}"></span>`; }).join("");
  }
  document.querySelectorAll(".btn").forEach((b) => b.classList.toggle("on", b.dataset.a === m.action));
}

function onStats(s) {
  set("episodes", s.episodes_all_time); set("alltime", LABEL[s.furthest_milestone_all_time] || "—");
  set("besttotal", s.best_fitness != null ? Math.round(s.best_fitness) : "—");
  set("restarts", s.restarts); set("crashes", s.crashes); set("allruntime", fmtTime(s.all_time_runtime_s));
}

function onEvent(m) {
  const log = $("log"); if (!log) return;
  const line = document.createElement("div");
  const t = new Date((m.ts || Date.now() / 1000) * 1000).toLocaleTimeString();
  if (m.type === "episode_end") line.textContent = `${t} episode ${m.episode} (seed ${m.seed}) ended: ${m.end_reason}, furthest ${LABEL[m.furthest_milestone] || m.furthest_milestone}`;
  else if (m.type === "watchdog") { line.textContent = `${t} WATCHDOG ${m.kind}`; line.className = "warn"; }
  else if (m.type === "milestone_first") line.textContent = `${t} all-time first: ${LABEL[m.name]} (episode ${m.episode})`;
  else return;
  log.prepend(line); while (log.children.length > 12) log.lastChild.remove();
}

function connect() {
  const es = new EventSource("/events");
  es.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type === "decision") onDecision(m);
    else if (m.type === "stats") onStats(m);
    else onEvent(m);
    set("conn", "live");
  };
  es.onerror = () => { set("conn", "reconnecting…"); };
}

fetch("/provenance").then((r) => r.json()).then((p) => {
  const el = $("provenance"); if (!el) return;
  const rows = [["Vision", p.vision], ["Brain simulation", p.brain_simulation], ["Controller", p.controller],
    ["Training", p.training], ["Within-run learning", p.within_run_learning], ["RAM", p.ram],
    ["Biological-learning research", p.biological_learning_research]];
  el.innerHTML = rows.map(([k, v]) => `<div><b>${k}</b> ${v}</div>`).join("");
  set("oneliner", p.one_liner);
});
connect();
