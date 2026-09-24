"""EVALUATOR / OVERLAY ONLY (E30): fitness, stream milestones and dialogue-stall classification.

Everything here may read Pokémon Red RAM (through `gameplay_eval.read_telemetry`) and the
framebuffer. Nothing here is ever passed to a controller: `flymon.live_runtime.Episode` calls the
evaluator only after the controller's decision and action have been applied, and controllers
accept nothing but framebuffers.

Dialogue / menu classification (rules frozen in results/e30-stream-readiness/reproduction-preregistration.md):
  a *long window event* is a maximal run of >= LONG_WINDOW (30, E25 LOCK_LEN) consecutive decisions
  with the window layer shown (rWY < 144). Each long event is classified, in this order:
    stall      the window-region image does not change for >= FROZEN_WINDOW (30) consecutive
               decisions inside the event, OR the event starts at a (map, x, y) where >= 2 earlier
               long events started in this episode (repeated re-triggering without progress);
    scripted   Oak's intercept at the Pallet Town north exit (map 0, y <= 1) or the first long event
               after entering Oak's Lab (map 40) (the arrival story sequence);
    navigable  anything else (text / menu that keeps advancing).
  An *overworld stall* is >= OVERWORLD_STALL (150) consecutive decisions without a window and without
  any change of (map, x, y).
"""
from __future__ import annotations

import hashlib

import numpy as np

from flymon import evolution as evo
from flymon import gameplay_eval as ev

LONG_WINDOW, FROZEN_WINDOW, REPEAT_LIMIT, OVERWORLD_STALL = 30, 30, 2, 150
OAK_LAB, PALLET, ROUTE1, HOUSE1F, BEDROOM = 40, 0, 12, 37, 38
STREAM_MILESTONES = ("bedroom", "bedroom_exited", "house_exited", "pallet_explored", "oak_lab", "route1", "battle")


def window_region_hash(frame, wy):
    """Hash of the window layer's screen region (rows wy..143); empty if no window."""
    if wy >= 144:
        return ""
    return hashlib.sha1(np.ascontiguousarray(frame[int(wy):, :, :3]).tobytes()).hexdigest()


def stream_milestone_index(name):
    return STREAM_MILESTONES.index(name)


class StreamEvaluator:
    def __init__(self, tel0):
        self.fit = evo.FitnessTracker((tel0["map"], tel0["x"], tel0["y"]))
        self.rows = []                    # (d, action, map, x, y, window, battle, wy, whash)
        self.reached = {"bedroom": 0}
        self.pallet_tiles = set()

    # ---------------------------------------------------------------- per decision
    def update(self, d, action, tel, frame):
        self.fit.update(d, action, tel)
        wh = window_region_hash(frame, tel["wy"]) if tel["window"] else ""
        self.rows.append((d, action or "NONE", tel["map"], tel["x"], tel["y"], int(tel["window"]), int(tel["battle"]),
                          int(tel["wy"]), wh))
        m = tel["map"]
        if m == PALLET:
            self.pallet_tiles.add((tel["x"], tel["y"]))
        checks = {"bedroom_exited": m != BEDROOM, "house_exited": m not in (BEDROOM, HOUSE1F),
                  "pallet_explored": len(self.pallet_tiles) >= 10, "oak_lab": m == OAK_LAB,
                  "route1": m == ROUTE1, "battle": tel["battle"] != 0}
        for k, ok in checks.items():
            if ok and k not in self.reached:
                self.reached[k] = d
        return self.snapshot(tel)

    def furthest(self):
        return max(self.reached, key=stream_milestone_index)

    def snapshot(self, tel):
        """Overlay-only view (never sent to the controller)."""
        return dict(map=tel["map"], map_name=ev.MAP_NAMES.get(tel["map"], str(tel["map"])), x=tel["x"], y=tel["y"],
                    window=bool(tel["window"]), battle=bool(tel["battle"]), furthest_milestone=self.furthest(),
                    unique_tiles=int(sum(len(t) for t in self.fit.tiles.values())), unique_maps=len(self.fit.tiles),
                    window_run=self.fit.window_run, dialogue_locks=self.fit.locks)

    # ---------------------------------------------------------------- episode end
    def dialogue_analysis(self):
        return classify_dialogue(self.rows)

    def result(self):
        r = self.fit.result()
        r["stream_milestones"] = {k: self.reached.get(k) for k in STREAM_MILESTONES}
        r["furthest_stream_milestone"] = self.furthest()
        r["dialogue"] = self.dialogue_analysis()
        r["window_decisions"] = int(sum(x[5] for x in self.rows))
        return r


def classify_dialogue(rows):
    """rows: (d, action, map, x, y, window, battle, wy, whash). Returns events + summary."""
    events, i, n = [], 0, len(rows)
    starts_at, entered_lab, lab_events = {}, None, 0
    while i < n:
        if rows[i][2] == OAK_LAB and entered_lab is None:
            entered_lab = i
        if not rows[i][5]:
            i += 1; continue
        j = i
        while j < n and rows[j][5]:
            if rows[j][2] == OAK_LAB and entered_lab is None:
                entered_lab = j
            j += 1
        length = j - i
        if length >= LONG_WINDOW:
            key = rows[i][2:5]
            prior = starts_at.get(key, 0)
            frozen = run = 1
            for k in range(i + 1, j):
                run = run + 1 if rows[k][8] == rows[k - 1][8] else 1
                frozen = max(frozen, run)
            m, y = rows[i][2], rows[i][4]
            first_lab = m == OAK_LAB and lab_events == 0
            if frozen >= FROZEN_WINDOW:
                cls, why = "stall", "frozen_window"
            elif prior >= REPEAT_LIMIT:
                cls, why = "stall", "repeat_loop"
            elif (m == PALLET and y <= 1) or first_lab:
                cls, why = "scripted", "oak_intercept" if m == PALLET else "oak_lab_arrival"
            else:
                cls, why = "navigable", "advancing"
            if m == OAK_LAB:
                lab_events += 1
            starts_at[key] = prior + 1
            events.append(dict(start=rows[i][0], length=length, map=m, x=rows[i][3], y=y, cls=cls, reason=why,
                               longest_frozen=frozen, ended_by_episode_end=j == n))
        i = j
    ow, run, prev = [], 0, None
    for r in rows:
        pos = r[2:5]
        if not r[5] and pos == prev:
            run += 1
        else:
            if run + 1 >= OVERWORLD_STALL:
                ow.append(run + 1)
            run = 0
        prev = pos
    if run + 1 >= OVERWORLD_STALL:
        ow.append(run + 1)
    by = {c: [e for e in events if e["cls"] == c] for c in ("scripted", "navigable", "stall")}
    return dict(events=events, n_long=len(events), counts={c: len(v) for c, v in by.items()},
                decisions={c: int(sum(e["length"] for e in v)) for c, v in by.items()},
                overworld_stall_runs=ow, overworld_stall_decisions=int(sum(ow)),
                genuine_stall_decisions=int(sum(e["length"] for e in by["stall"]) + sum(ow)),
                any_genuine_stall=bool(by["stall"] or ow), decisions_total=len(rows))
