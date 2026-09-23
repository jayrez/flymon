"""Dopamine-gated mushroom-body plasticity on the audited E26 edge set (Phases 3-5).

Scope: the 418 existing FlyBrain edges KCg-d -> MBON01 (appetitive-candidate compartment,
gated by PAM01) and KCg-d -> MBON11 (aversive-candidate compartment, gated by PPL101) found by
the E26 circuit audit (edge-set SHA-256 65dbc6264c04a2fa...). Nothing else is plastic.

Design
------
* The frozen FlyBrain matrix is never modified. Learned changes live in a separate
  `PlasticState` (one delta per audited edge). `PlasticBrain.step` replicates
  `flybrain.FlyBrain.step` (0.1.0, batch 1, no eye drive) operation by operation and adds

      I_plastic[post] = gain * sum_e delta_e * spiked(pre_e)                (same stage as W x)

  With all deltas zero it is bit-identical to FlyBrain's own step (tested).
* Reinforcement bridge: an external request (appetitive / aversive magnitude) becomes voltage
  injection into the audited DAN population (PAM01 or PPL101) for a fixed number of steps.
  The learning gate is the DAN population's *observed simulated spiking*, not the request.
  The DANs' ordinary fast excitatory synapses stay intact (the acute confound is measured).
* Eligibility per KCg-d presynaptic cell: e_i <- lambda e_i + s_i(t), clipped at e_max.
* Three-factor rule per edge (pre i, post j, compartment c):

      d(delta_ij) = sign_c * eta * D_c(t) * e_i(t) * w0_ij,   delta clipped to [lo_ij, hi_ij]

  D_c(t) = max(0, f_c(t) - theta_c), f_c = fraction of compartment-c DANs spiking this step.
  Default sign -1 (depression, literature default); bounds [-w0, 0] (w_eff in [0, w0]).
  Scaling by w0 makes eta a relative rate. Only the gating DAN of an edge's own compartment
  can change it (compartment specificity), unless `cross_compartment` is set for a control.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

AUDITED_EDGE_HASH = "65dbc6264c04a2fa"
COMPARTMENTS = {"appetitive": dict(mbon="MBON01", dan="PAM01"),
                "aversive": dict(mbon="MBON11", dan="PPL101")}
KC_TYPE = "KCg-d"


# ------------------------------------------------------------------ audited edge set
@dataclass(frozen=True)
class EdgeSet:
    pre_idx: np.ndarray        # FlyBrain index of KCg-d presynaptic cell
    post_idx: np.ndarray       # FlyBrain index of MBON
    pre_body: np.ndarray
    post_body: np.ndarray
    w0: np.ndarray             # baseline FlyBrain weight (signed, input-normalised)
    compartment: np.ndarray    # "appetitive" / "aversive"
    sha256: str

    @property
    def n(self):
        return len(self.w0)


def audited_edge_set(cell_type, ids, W):
    """Reproduce the audit's 418-edge set from FlyBrain metadata and W (rows = post)."""
    ct = np.asarray(cell_type).astype(str)
    kc = np.flatnonzero(ct == KC_TYPE)
    rows = []
    for comp, spec in COMPARTMENTS.items():
        mb = np.flatnonzero(ct == spec["mbon"])
        sub = W[mb][:, kc].tocoo()
        for r, c, v in zip(sub.row, sub.col, sub.data):
            rows.append((int(ids[kc[c]]), int(ids[mb[r]]), spec["mbon"], float(v), int(kc[c]), int(mb[r]), comp))
    rows.sort()
    h = hashlib.sha256(json.dumps([(a, b) for a, b, *_ in rows]).encode()).hexdigest()
    ro = lambda a, dt: np.asarray(a, dt)
    es = EdgeSet(pre_idx=ro([r[4] for r in rows], np.int64), post_idx=ro([r[5] for r in rows], np.int64),
                 pre_body=ro([r[0] for r in rows], np.int64), post_body=ro([r[1] for r in rows], np.int64),
                 w0=ro([r[3] for r in rows], np.float64), compartment=ro([r[6] for r in rows], object), sha256=h)
    for a in (es.pre_idx, es.post_idx, es.pre_body, es.post_body, es.w0, es.compartment):
        a.setflags(write=False)
    return es


def population(cell_type, name):
    return np.flatnonzero(np.asarray(cell_type).astype(str) == name)


# ------------------------------------------------------------------ configuration / state
@dataclass(frozen=True)
class PlasticityConfig:
    eta: float = 0.0                 # relative learning rate (0 = plasticity off)
    tau_elig_steps: float = 25.0     # eligibility time constant in brain steps (20 ms each)
    e_max: float = 5.0               # eligibility clip
    theta_appetitive: float = 0.0    # DAN gate thresholds (fraction of population spiking),
    theta_aversive: float = 0.0      #   fixed by the preregistered baseline-percentile rule
    sign: float = -1.0               # -1 depression (default), +1 potentiation diagnostic
    k_max: float = 1.0               # upper bound w_eff <= k_max * w0 (1.0 = depression-only)
    cross_compartment: bool = False  # control only: every DAN channel gates every edge
    shuffle_eligibility_seed: int | None = None   # control only: permute KC eligibility

    def to_json(self):
        return asdict(self)


@dataclass
class PlasticState:
    edge_hash: str
    delta: np.ndarray
    meta: dict = field(default_factory=dict)

    @classmethod
    def zeros(cls, edges: EdgeSet):
        return cls(edges.sha256, np.zeros(edges.n))

    def effective(self, edges):
        return edges.w0 + self.delta

    def save(self, path, config: PlasticityConfig, provenance: dict):
        Path(path).write_text(json.dumps(dict(edge_hash=self.edge_hash, delta=self.delta.tolist(),
                                              config=config.to_json(), provenance=provenance, meta=self.meta)))

    @classmethod
    def load(cls, path, edges: EdgeSet):
        d = json.loads(Path(path).read_text())
        if d["edge_hash"] != edges.sha256:
            raise ValueError("plastic state edge-set hash does not match the audited edge set")
        delta = np.asarray(d["delta"], float)
        if delta.shape != (edges.n,):
            raise ValueError("plastic state size mismatch")
        return cls(d["edge_hash"], delta, d.get("meta", {}))


# ------------------------------------------------------------------ learning rule (CPU, tiny)
class Learner:
    """Eligibility traces on KCg-d cells and the three-factor update on the 418 edges."""

    def __init__(self, edges: EdgeSet, config: PlasticityConfig, kc_idx, dan_idx: dict):
        self.e, self.c = edges, config
        self.kc_idx = np.asarray(kc_idx)                       # the KCg-d population
        self.kc_slot = {int(k): i for i, k in enumerate(self.kc_idx)}
        self.edge_kc = np.array([self.kc_slot[int(p)] for p in edges.pre_idx])
        self.dan_idx = {c: np.asarray(v) for c, v in dan_idx.items()}
        self.lam = float(np.exp(-1.0 / config.tau_elig_steps)) if config.tau_elig_steps > 0 else 0.0
        self.elig = np.zeros(len(self.kc_idx))
        self.lo = -edges.w0.copy()                                   # w_eff >= 0
        self.hi = (config.k_max - 1.0) * edges.w0                   # w_eff <= k_max w0
        self.perm = (np.random.default_rng(config.shuffle_eligibility_seed).permutation(len(self.kc_idx))
                     if config.shuffle_eligibility_seed is not None else None)
        comp = edges.compartment
        self.masks = {c: (comp == c) for c in COMPARTMENTS}

    def reset_traces(self):
        self.elig[:] = 0.0

    def dan_signal(self, fractions: dict, theta: dict):
        """Per-channel gate D_c = max(0, observed fraction of channel DANs spiking - theta_c)."""
        return {c: max(0.0, float(fractions.get(c, 0.0)) - float(theta.get(c, 0.0))) for c in self.dan_idx}

    def step(self, kc_spikes, D, state: PlasticState, learning=True):
        """kc_spikes: 0/1 per KCg-d cell for this step. Returns summed |update| this step."""
        self.elig = np.minimum(self.lam * self.elig + kc_spikes, self.c.e_max)
        if not learning or self.c.eta == 0.0:
            return 0.0
        el = self.elig if self.perm is None else self.elig[self.perm]
        e_edge = el[self.edge_kc]
        upd = np.zeros(self.e.n)
        for c, mask in self.masks.items():
            gate = sum(D.values()) if self.c.cross_compartment else D.get(c, 0.0)
            if gate > 0:
                upd[mask] = self.c.sign * self.c.eta * gate * e_edge[mask] * self.e.w0[mask]
        if not upd.any():
            return 0.0
        new = np.clip(state.delta + upd, self.lo, self.hi)
        moved = float(np.abs(new - state.delta).sum())
        state.delta = new
        return moved


# ------------------------------------------------------------------ brain step with plastic current
class PlasticBrain:
    """FlyBrain step with (a) vector voltage injection and (b) the plastic-delta current.

    Replicates flybrain.FlyBrain.step (0.1.0, batch 1, no eye drive). The frozen matrix is
    only read; the plastic contribution is added at the same stage as W x."""

    def __init__(self, brain, edges: EdgeSet):
        if brain.batch != 1:
            raise ValueError("batch 1 only")
        self.b, self.e = brain, edges
        xp = brain.xp
        self.pre = xp.asarray(edges.pre_idx); self.post = xp.asarray(edges.post_idx)
        self.post_u, inv = np.unique(edges.post_idx, return_inverse=True)
        self.post_u_dev = xp.asarray(self.post_u); self.inv = xp.asarray(inv)
        self._mask = xp.zeros(brain.n, xp.float32)

    def prepare(self, pairs):
        xp = self.b.xp
        if not pairs:
            return None
        idx = np.concatenate([np.asarray(i, np.int64) for i, _ in pairs])
        vals = np.concatenate([np.full(len(i), np.float32(a), np.float32) for i, a in pairs])
        return xp.asarray(idx), xp.asarray(vals)

    def step(self, prepared, delta):
        b, xp = self.b, self.b.xp
        current = b.synaptic_input(b.fired) * b.gain
        if delta is not None and np.any(delta):
            self._mask[:] = 0
            if b.fired.size:
                self._mask[b.fired] = 1.0
            contrib = xp.asarray(delta, xp.float32) * self._mask[self.pre]
            per_post = xp.bincount(self.inv, weights=contrib, minlength=len(self.post_u)).astype(xp.float32)
            current[self.post_u_dev, 0] += per_post * xp.float32(b.gain)
        b.v *= b.decay
        b.v += current + b.tonic
        b.v += (b.rng.random((b.n, 1)) < b.noise_hz * b.dt) * np.float32(b.noise_amp)
        if prepared is not None:
            idx, vals = prepared
            b.v[idx, 0] += vals
        if b.refractory_steps:
            b.v[(b.steps - b.last_spike) <= b.refractory_steps] = 0.0
        fired = xp.flatnonzero(b.v >= 1.0)
        b.v.ravel()[fired] = 0.0
        if b.refractory_steps:
            b.last_spike.ravel()[fired] = b.steps
        b.fired = fired
        b.steps += 1
        return fired if xp is np else fired.get()
