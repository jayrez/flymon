"""Durable stream metadata (E30): episodes, all-time milestones, restarts, watchdog events.

SQLite, one writer (the supervisor). This is stream bookkeeping only: nothing here is neural
learning and nothing is ever read by the controller.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from flymon.stream_eval import STREAM_MILESTONES

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY, seed INTEGER NOT NULL, started REAL NOT NULL, ended REAL,
  decisions INTEGER, fitness REAL, furthest TEXT, end_reason TEXT, watchdog INTEGER DEFAULT 0, wall_s REAL,
  controller_sha256 TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS milestones (name TEXT PRIMARY KEY, first_ts REAL, episode INTEGER, seed INTEGER);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts REAL, kind TEXT, detail TEXT);
"""


class StreamStore:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    # ---------------------------------------------------------------- meta counters
    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        self.db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, json.dumps(value)))

    def add(self, key, delta):
        with self.db:
            self.set(key, self.get(key, 0) + delta)

    def next_seed(self, base):
        i = self.get("next_seed_index", 0)
        self.set("next_seed_index", i + 1)
        return base + i

    # ---------------------------------------------------------------- episodes / milestones / events
    def start_episode(self, seed, controller_sha256):
        cur = self.db.execute("INSERT INTO episodes(seed, started, controller_sha256) VALUES(?,?,?)",
                              (seed, time.time(), controller_sha256))
        return cur.lastrowid

    def end_episode(self, eid, decisions, fitness, furthest, reason, watchdog, wall_s, summary=None):
        with self.db:
            self.db.execute("UPDATE episodes SET ended=?, decisions=?, fitness=?, furthest=?, end_reason=?, watchdog=?, "
                            "wall_s=?, summary=? WHERE id=?", (time.time(), decisions, fitness, furthest, reason,
                                                               int(bool(watchdog)), wall_s, json.dumps(summary or {}), eid))
            self.set("all_time_runtime_s", self.get("all_time_runtime_s", 0.0) + float(wall_s or 0.0))
            if fitness is not None and fitness > self.get("best_fitness", float("-inf")):
                self.set("best_fitness", fitness)
                self.set("best_episode", dict(id=eid, seed=self.db.execute("SELECT seed FROM episodes WHERE id=?", (eid,)).fetchone()[0]))

    def milestone(self, name, eid, seed):
        """Record an all-time first; returns True if new."""
        if name not in STREAM_MILESTONES:
            raise ValueError(name)
        cur = self.db.execute("INSERT OR IGNORE INTO milestones(name, first_ts, episode, seed) VALUES(?,?,?,?)",
                              (name, time.time(), eid, seed))
        return cur.rowcount == 1

    def event(self, kind, detail=None):
        self.db.execute("INSERT INTO events(ts, kind, detail) VALUES(?,?,?)", (time.time(), kind, json.dumps(detail or {})))

    # ---------------------------------------------------------------- summary
    def furthest_all_time(self):
        names = [r[0] for r in self.db.execute("SELECT name FROM milestones")]
        return max(names, key=STREAM_MILESTONES.index) if names else None

    def stats(self):
        q = lambda s, *a: self.db.execute(s, a).fetchone()[0]
        return dict(episodes_all_time=q("SELECT COUNT(*) FROM episodes"),
                    episodes_completed=q("SELECT COUNT(*) FROM episodes WHERE end_reason='budget'"),
                    all_time_runtime_s=self.get("all_time_runtime_s", 0.0),
                    furthest_milestone_all_time=self.furthest_all_time(),
                    milestone_firsts={r[0]: dict(ts=r[1], episode=r[2], seed=r[3]) for r in
                                      self.db.execute("SELECT name, first_ts, episode, seed FROM milestones")},
                    best_fitness=self.get("best_fitness"), best_episode=self.get("best_episode"),
                    crashes=q("SELECT COUNT(*) FROM events WHERE kind='worker_crash'"),
                    restarts=q("SELECT COUNT(*) FROM events WHERE kind='worker_restart'"),
                    watchdog_interventions=q("SELECT COUNT(*) FROM events WHERE kind LIKE 'watchdog_%'"))

    def close(self):
        self.db.close()
