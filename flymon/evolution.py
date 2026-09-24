"""Evolvable downstream gameplay controller (Experiment 25).

The frozen E23 T4 sensory model (`flymon.frozen_t4`) is **not** part of any genome. A genome
is a linear softmax decoder from preregistered neural features to seven Game Boy actions:

    logits = W x + b,   p = softmax(logits / T),   T = exp(log_temp)

Feature sets (architecture):
  A  "dn"     DN / population spike rates (E14 populations), z-scored against the per-episode
              no-vision baseline, plus their one-decision deltas;
  B  "t4"     pooled frozen-T4 activity: 4 subtypes x 2 eyes and a 2 x 3 screen-region grid of
              receptive-field centres, plus one-decision deltas;
  C  "t4dn"   A + B.
Every architecture also sees a one-hot of its own previous action (controller state, not game
state). Controllers never see RAM, coordinates, map, dialogue, battle, milestones or fitness.

Fitness is computed by the evaluator from RAM telemetry after actions (`FitnessTracker`).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from flymon import gameplay_eval as ev

ACTIONS = ("NONE", "UP", "DOWN", "LEFT", "RIGHT", "A", "B")
DISABLED_ACTIONS = ("START", "SELECT")
ARCHITECTURES = ("dn", "t4", "t4dn")
T4_SCALE = 1.0 / 0.016453          # median positive E23 calibration response (E24 gain derivation)
DN_KEYS = ("DNa02_L", "DNa02_R", "DNg100_L", "DNg100_R", "MDN_L", "MDN_R", "DNp01_L", "DNp01_R")
DN_SD_FLOOR_HZ = 1.0
FEATURE_CLIP = 10.0
T4_SUBTYPES = ("T4a", "T4b", "T4c", "T4d")


def t4_feature_names():
    base = [f"{s}_{e}" for s in T4_SUBTYPES for e in ("L", "R")]
    base += [f"region_{h}{v}" for h in ("left", "right") for v in ("top", "mid", "bot")]
    return base + [f"d_{n}" for n in base]


def dn_feature_names():
    base = list(DN_KEYS) + ["P20_rms_z"]
    return base + [f"d_{n}" for n in base]


def feature_names(arch):
    names = {"dn": dn_feature_names(), "t4": t4_feature_names(),
             "t4dn": t4_feature_names() + dn_feature_names()}[arch]
    return names + [f"prev_{a}" for a in ACTIONS]


def n_features(arch):
    return len(feature_names(arch))


# ------------------------------------------------------------------ feature extraction
class T4Pooling:
    """Fixed pooling of the frozen T4 population (anatomy only: subtype, eye, RF centre)."""

    def __init__(self, subtype, side, rf_uv):
        subtype = np.asarray(subtype); side = np.asarray(side); rf_uv = np.asarray(rf_uv, float)
        groups = []
        for s in T4_SUBTYPES:
            for e in ("L", "R"):
                groups.append((subtype == s) & (side == e))
        for h in (0, 1):
            for v in (0, 1, 2):
                hm = (rf_uv[:, 0] >= 0.5) == bool(h)
                vm = np.floor(np.clip(rf_uv[:, 1], 0, 0.9999) * 3) == v
                groups.append(hm & vm)
        self.masks = [np.flatnonzero(g) for g in groups]
        if any(len(m) == 0 for m in self.masks):
            raise ValueError("empty T4 pooling group")

    def pool(self, response):
        if response is None:
            return np.zeros(len(self.masks))
        r = np.asarray(response, float)
        return np.array([r[m].mean() for m in self.masks]) * T4_SCALE


class FeatureBuilder:
    """Builds the controller input vector. Only neural quantities and the previous action."""

    def __init__(self, arch, pooling=None):
        if arch not in ARCHITECTURES:
            raise ValueError(arch)
        self.arch, self.pooling = arch, pooling
        self.prev_t4 = None; self.prev_dn = None; self.prev_action = "NONE"
        self.names = feature_names(arch)

    def dn_vector(self, rates, baseline_mean, baseline_sd, p20_rms_z):
        z = [(rates[k] - baseline_mean[k]) / max(baseline_sd[k], DN_SD_FLOOR_HZ) for k in DN_KEYS]
        return np.array(z + [p20_rms_z], float)

    def build(self, t4_response=None, dn=None):
        parts = []
        if self.arch in ("t4", "t4dn"):
            cur = self.pooling.pool(t4_response)
            prev = self.prev_t4 if self.prev_t4 is not None else cur
            parts += [cur, cur - prev]; self.prev_t4 = cur
        if self.arch in ("dn", "t4dn"):
            cur = np.asarray(dn, float)
            prev = self.prev_dn if self.prev_dn is not None else cur
            parts += [cur, cur - prev]; self.prev_dn = cur
        onehot = np.zeros(len(ACTIONS)); onehot[ACTIONS.index(self.prev_action)] = 1.0
        x = np.clip(np.concatenate(parts + [onehot]), -FEATURE_CLIP, FEATURE_CLIP)
        assert len(x) == len(self.names)
        return x

    def observe_action(self, action):
        self.prev_action = action or "NONE"


# ------------------------------------------------------------------ genome
@dataclass
class Genome:
    arch: str
    W: np.ndarray        # (n_actions, n_features)
    b: np.ndarray        # (n_actions,)
    log_temp: float

    def flat(self):
        return np.concatenate([self.W.ravel(), self.b, [self.log_temp]]).astype(np.float64)

    def sha256(self):
        return hashlib.sha256(self.arch.encode() + self.flat().tobytes()).hexdigest()

    def to_json(self):
        return dict(arch=self.arch, W=self.W.tolist(), b=self.b.tolist(), log_temp=float(self.log_temp),
                    sha256=self.sha256())

    @classmethod
    def from_json(cls, d):
        g = cls(d["arch"], np.asarray(d["W"], float), np.asarray(d["b"], float), float(d["log_temp"]))
        if "sha256" in d and d["sha256"] != g.sha256():
            raise ValueError("genome hash mismatch")
        return g

    @classmethod
    def random(cls, arch, rng, w_sd=0.1, b_sd=0.1):
        return cls(arch, rng.normal(0, w_sd, (len(ACTIONS), n_features(arch))),
                   rng.normal(0, b_sd, len(ACTIONS)), 0.0)

    def mutate(self, rng, w_sd=0.1, b_sd=0.2, t_sd=0.1):
        return Genome(self.arch, self.W + rng.normal(0, w_sd, self.W.shape),
                      self.b + rng.normal(0, b_sd, self.b.shape),
                      float(np.clip(self.log_temp + rng.normal(0, t_sd), -3.0, 1.6)))

    def temperature(self):
        return float(np.exp(np.clip(self.log_temp, -3.0, 1.6)))


class Policy:
    """Stochastic linear-softmax decoder; its RNG is seeded per episode."""

    def __init__(self, genome: Genome, rng):
        self.g, self.rng = genome, rng

    def probabilities(self, x):
        z = (self.g.W @ x + self.g.b) / self.g.temperature()
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def act(self, x):
        p = self.probabilities(x)
        a = ACTIONS[int(self.rng.choice(len(ACTIONS), p=p))]
        return None if a == "NONE" else a


# ------------------------------------------------------------------ evolution step
def next_population(ranked, rng, size, elite_frac=0.125, immigrant_frac=0.10, arch=None):
    """ranked: genomes sorted best-first. Elites kept; mutated offspring; random immigrants."""
    n_elite = max(1, int(np.ceil(elite_frac * size)))
    n_imm = int(round(immigrant_frac * size))
    elites = ranked[:n_elite]
    out = list(elites)
    k = 0
    while len(out) < size - n_imm:
        out.append(elites[k % n_elite].mutate(rng)); k += 1
    arch = arch or elites[0].arch
    while len(out) < size:
        out.append(Genome.random(arch, rng))
    return out, n_elite


def rank(genomes, fitness):
    """Deterministic ranking: fitness descending, ties broken by genome hash."""
    order = sorted(range(len(genomes)), key=lambda i: (-fitness[i], genomes[i].sha256()))
    return [genomes[i] for i in order], order


# ------------------------------------------------------------------ fitness
MILESTONE_BONUS = {"M1": 100, "M2": 1000, "M3": 3000, "M4": 1000, "M5": 3000, "M7": 5000, "M8": 5000}
TILE_POINTS, TILE_CAP_PER_MAP, MAP_POINTS = 5, 40, 50
CYCLE_POINTS, CYCLE_CAP, CYCLE_MAX_LEN = 100, 3, 30
LOCK_LEN, LOCK_PENALTY, LOCK_CAP = 30, 200, 3
LOOP_PENALTY, IDLE_PENALTY, REVISIT_PENALTY = 300, 300, 100


class FitnessTracker:
    """Evaluator-side fitness from RAM telemetry (never visible to the controller).

    Progression dominates: the largest possible in-house exploration + interaction score
    (2 maps x 40 tiles x 5 + 50 + 3 x 100 = 750) is below the M3 (leave house) bonus of 3000,
    and bedroom-only maxima (40 x 5 + 300 = 500) are below the M2 bonus of 1000."""

    def __init__(self, spawn):
        self.ms = ev.MilestoneTracker(); self.ms.spawn = spawn
        self.loops = ev.LoopDetector()
        self.tiles = {}; self.window_run = 0; self.cycles = 0; self.locks = 0
        self.moves = 0; self.revisits = 0; self.prev = None; self.actions = []

    def update(self, decision, action, tel):
        self.ms.update(decision, action, tel)
        self.loops.update(action, tel)
        self.actions.append(action or "NONE")
        pos = (tel["map"], tel["x"], tel["y"])
        seen = self.tiles.setdefault(tel["map"], set())
        if self.prev is not None and pos != self.prev:
            self.moves += 1
            self.revisits += (tel["x"], tel["y"]) in seen
        seen.add((tel["x"], tel["y"]))
        if tel["window"]:
            self.window_run += 1
            if self.window_run == LOCK_LEN:
                self.locks += 1
        else:
            if 0 < self.window_run <= CYCLE_MAX_LEN:
                self.cycles += 1
            self.window_run = 0
        self.prev = pos

    def result(self):
        flags = self.loops.finalise()
        n = max(1, len(flags))
        idle = sum(f["inactivity"] for f in flags) / n
        loop = sum(any(v for k, v in f.items() if k != "inactivity") for f in flags) / n
        revisit = self.revisits / self.moves if self.moves else 0.0
        milestone = sum(MILESTONE_BONUS.get(m, 0) for m in self.ms.reached)
        explore = TILE_POINTS * sum(min(len(t), TILE_CAP_PER_MAP) for t in self.tiles.values()) \
            + MAP_POINTS * (len(self.tiles) - 1)
        interact = CYCLE_POINTS * min(self.cycles, CYCLE_CAP)
        penalty = LOOP_PENALTY * loop + IDLE_PENALTY * idle + LOCK_PENALTY * min(self.locks, LOCK_CAP) \
            + REVISIT_PENALTY * revisit
        counts = {a: self.actions.count(a) for a in ACTIONS}
        return dict(fitness=float(milestone + explore + interact - penalty),
                    terms=dict(milestone=milestone, exploration=explore, interaction=interact, penalty=float(penalty)),
                    milestones={m: self.ms.reached.get(m) for m, _ in ev.MILESTONES},
                    unique_tiles=int(sum(len(t) for t in self.tiles.values())), unique_maps=len(self.tiles),
                    completed_window_cycles=self.cycles, dialogue_locks=self.locks,
                    idle_fraction=float(idle), loop_fraction_nonidle=float(loop), revisit_rate=float(revisit),
                    successful_moves=self.moves, action_counts=counts, decisions=len(self.actions))


def config_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
