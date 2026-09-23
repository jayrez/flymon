"""Evaluator-only telemetry, milestones and loop detection for gameplay episodes (E24+).

EVALUATOR ONLY. Everything here may read Pokémon Red RAM through PyBoy, and nothing here
may ever be passed to a controller. Runners keep the objects in this module on the
evaluation side of the loop; `ControllerInputGuard` enforces that controllers receive only
neural rates.

RAM addresses (Pokémon Red, verified empirically from `states/bedroom.state`):
  wCurMap 0xD35E, wYCoord 0xD361, wXCoord 0xD362, wIsInBattle 0xD057,
  rWY 0xFF4A (window layer Y; < 144 while a text box or menu window is shown).
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

ADDR = {"map": 0xD35E, "y": 0xD361, "x": 0xD362, "battle": 0xD057, "wy": 0xFF4A}
MAP_NAMES = {38: "RedsHouse2F", 37: "RedsHouse1F", 0: "PalletTown", 12: "Route1",
             39: "BluesHouse", 40: "OaksLab"}
SPAWN_MAP = 38
DIRECTIONS = ("LEFT", "RIGHT", "UP", "DOWN")
MILESTONES = (
    ("M0", "produces nontrivial actions (>= 10 non-NONE decisions, >= 2 action types)"),
    ("M1", "moves from the spawn tile"),
    ("M2", "exits the starting bedroom (map != RedsHouse2F)"),
    ("M3", "exits the house (reaches PalletTown)"),
    ("M4", "navigates Pallet Town (>= 10 unique PalletTown tiles)"),
    ("M5", "enters another map (not bedroom, house 1F or PalletTown)"),
    ("M6", "triggers a text / menu window (interaction)"),
    ("M7", "reaches Route 1"),
    ("M8", "first battle"),
)


def read_telemetry(game) -> dict:
    """EVALUATOR ONLY: RAM snapshot. Never pass the result to a controller."""
    mem = game._pyboy.memory
    t = {k: int(mem[a]) for k, a in ADDR.items()}
    t["window"] = t["wy"] < 144
    return t


class ControllerInputGuard:
    """Rejects any controller input that is not a neural-rate key known in advance."""

    FORBIDDEN = ("map", "x", "y", "battle", "wy", "window", "tile", "milestone", "telemetry", "ram")

    def __init__(self, allowed_keys):
        self.allowed = frozenset(allowed_keys)

    def check(self, rates: dict) -> dict:
        extra = set(rates) - self.allowed
        bad = [k for k in rates if any(f == k.lower() or k.lower().startswith(f + "_") for f in self.FORBIDDEN)]
        if extra or bad:
            raise RuntimeError(f"controller input contains non-neural keys: {sorted(extra | set(bad))}")
        return rates


@dataclass
class MilestoneTracker:
    spawn: tuple | None = None
    reached: dict = field(default_factory=dict)
    pallet_tiles: set = field(default_factory=set)
    non_none: int = 0
    action_types: set = field(default_factory=set)

    def update(self, decision: int, action, tel: dict) -> list:
        new = []
        pos = (tel["map"], tel["x"], tel["y"])
        if self.spawn is None:
            self.spawn = pos
        if action is not None:
            self.non_none += 1; self.action_types.add(action)
        if tel["map"] == 0:
            self.pallet_tiles.add((tel["x"], tel["y"]))
        checks = {
            "M0": self.non_none >= 10 and len(self.action_types) >= 2,
            "M1": pos != self.spawn,
            "M2": tel["map"] != SPAWN_MAP,
            "M3": tel["map"] == 0,
            "M4": len(self.pallet_tiles) >= 10,
            "M5": tel["map"] not in (SPAWN_MAP, 37, 0),
            "M6": tel["window"],
            "M7": tel["map"] == 12,
            "M8": tel["battle"] != 0,
        }
        for m, ok in checks.items():
            if ok and m not in self.reached:
                self.reached[m] = decision; new.append(m)
        return new

    def best(self):
        order = [m for m, _ in MILESTONES]
        return max((order.index(m) for m in self.reached), default=-1)

    def best_name(self):
        b = self.best()
        return None if b < 0 else MILESTONES[b][0]


class LoopDetector:
    """Online behavioural-loop flags from actions and evaluator telemetry.

    wall_bump: a directional action repeated in the same direction with no tile or map change
               and no window shown (the first press in a new direction only turns the player);
    lr_oscillation / ud_oscillation: >= 3 direction reversals on one axis within the last 8
               directional decisions with net displacement <= 1 tile on that axis;
    menu_loop: text/menu window shown for >= 20 consecutive decisions;
    a_spam: >= 3 A presses within the last 20 decisions;
    inactivity: part of a run of >= 20 consecutive NONE decisions (flagged retroactively
               in `finalise`)."""

    def __init__(self):
        self.prev = None; self.prev_action = None
        self.dirs = deque(maxlen=8); self.a_hist = deque(maxlen=20)
        self.window_run = 0; self.none_run = 0
        self.flags = []
        self.actions = []

    def update(self, action, tel):
        pos = (tel["map"], tel["x"], tel["y"])
        f = dict(wall_bump=False, lr_oscillation=False, ud_oscillation=False,
                 menu_loop=False, a_spam=False, inactivity=False)
        if action in DIRECTIONS and self.prev is not None:
            if pos == self.prev and not tel["window"] and action == self.prev_action:
                f["wall_bump"] = True
        if action in DIRECTIONS:
            self.dirs.append((action, pos))
        for axis, pair, k in (("lr", ("LEFT", "RIGHT"), 1), ("ud", ("UP", "DOWN"), 2)):
            seq = [(a, p) for a, p in self.dirs if a in pair]
            rev = sum(1 for (a, _), (b, _) in zip(seq, seq[1:]) if a != b)
            if rev >= 3 and len({p[0] for _, p in seq}) == 1 and \
                    abs(seq[-1][1][k] - seq[0][1][k]) <= 1:
                f[f"{axis}_oscillation"] = True
        self.window_run = self.window_run + 1 if tel["window"] else 0
        f["menu_loop"] = self.window_run >= 20
        self.a_hist.append(action == "A")
        f["a_spam"] = sum(self.a_hist) >= 3
        self.none_run = self.none_run + 1 if action is None else 0
        f["inactivity"] = self.none_run >= 20
        self.flags.append(f)
        self.actions.append(action)
        self.prev, self.prev_action = pos, action
        return f

    def finalise(self):
        """Mark every decision inside a >= 20-decision NONE run as inactivity."""
        i, n = 0, len(self.actions)
        while i < n:
            if self.actions[i] is None:
                j = i
                while j < n and self.actions[j] is None:
                    j += 1
                if j - i >= 20:
                    for k in range(i, j):
                        self.flags[k]["inactivity"] = True
                i = j
            else:
                i += 1
        return self.flags


def entropy(counter: Counter) -> float:
    n = sum(counter.values())
    if not n:
        return 0.0
    p = np.array([v / n for v in counter.values() if v])
    return float(-(p * np.log2(p)).sum())


def episode_metrics(rows: list, flags: list, milestones: MilestoneTracker, minutes: float) -> dict:
    """Aggregate per-episode metrics from compact decision rows (evaluator side)."""
    acts = Counter(r["action"] or "NONE" for r in rows)
    n = len(rows)
    moves, map_changes, dir_changes, move_dirs, revisits = 0, 0, 0, Counter(), 0
    tiles, maps = set(), set()
    prev = None; prev_dir = None; last_dir = None
    for r in rows:
        pos = (r["map"], r["x"], r["y"])
        if r["action"] in DIRECTIONS:
            if prev_dir is not None and r["action"] != prev_dir:
                dir_changes += 1
            prev_dir = last_dir = r["action"]
        if prev is not None and pos != prev:
            moves += 1
            revisits += pos in tiles
            if pos[0] != prev[0]:
                map_changes += 1
            elif last_dir is not None:
                # a step spans 16 frames but a decision only 12, so the tile change often lands
                # on the following (NONE) decision: attribute it to the last directional press
                move_dirs[last_dir] += 1
        tiles.add(pos); maps.add(r["map"])
        prev = pos
    loop_any = [any(f.values()) for f in flags]
    per_flag = {k: float(np.mean([f[k] for f in flags])) if flags else 0.0 for k in
                ("wall_bump", "lr_oscillation", "ud_oscillation", "menu_loop", "a_spam", "inactivity")}
    directional = sum(acts[d] for d in DIRECTIONS)
    return dict(
        decisions=n, total_actions=n - acts["NONE"], actions_per_minute=(n - acts["NONE"]) / minutes,
        action_counts=dict(acts), directional_actions=directional,
        a_presses=acts["A"], b_presses=acts["B"], start_presses=acts["START"], select_presses=acts["SELECT"],
        idle_fraction=acts["NONE"] / n if n else 0.0,
        successful_moves=moves, map_changes=map_changes,
        blocked_moves=int(sum(f["wall_bump"] for f in flags)),
        direction_changes=dir_changes, locomotion_entropy_bits=entropy(move_dirs),
        action_entropy_bits=entropy(acts),
        unique_tiles=len(tiles), unique_maps=len(maps), maps_visited=sorted(MAP_NAMES.get(m, str(m)) for m in maps),
        revisit_rate=revisits / moves if moves else 0.0,
        loop_fraction=float(np.mean(loop_any)) if loop_any else 0.0, loop_flag_fractions=per_flag,
        milestones={m: milestones.reached.get(m) for m, _ in MILESTONES},
        best_milestone=milestones.best_name(), best_milestone_index=milestones.best())
