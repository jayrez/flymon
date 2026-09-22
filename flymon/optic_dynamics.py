"""Opt-in optic-lobe dynamics adapter for Experiment 17.

The reference FlyBrain path is never modified: `RepairedFlyBrain` with an empty
`OpticDynamicsConfig` reproduces `FlyBrain.step` exactly (same RNG draws, same
order of operations), so Experiments 1-16 keep their historical behaviour. All
Experiment-17 interventions are explicit, population-scoped and biologically
motivated:

* `tonic_add`   - extra maintained depolarisation for named cell types. Lamina
  monopolar cells (L1/L2/L3) are graded, tonically depolarised neurons that
  photoreceptor histamine *inhibits*; in MaleCNS every R1-6 -> L1/L2/L3 weight is
  negative, so in a purely spiking model with no maintained drive the lamina is
  pinned at zero and the whole downstream pathway loses its dynamic range. A
  tonic term restores a baseline that inhibition can modulate bidirectionally.
* `tau_scale`   - population-specific membrane time constants (e.g. a slow,
  sign-inverting Mi9/Mi4/CT1 arm).
* `delay_steps` - population-specific axonal/synaptic transmission delay, the
  temporal offset a Hassenstein-Reichardt / Barlow-Levick style motion detector
  needs. A presynaptic spike from a delayed population is delivered `n` steps
  later; the weight matrix itself is untouched because the delay is applied to
  the presynaptic spike vector, not to W.
* `eye_gain_scale` - multiplies the photoreceptor drive only.

No game state, RAM, reward, action label or controller code is involved.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass(frozen=True)
class OpticDynamicsConfig:
    name: str = "reference"
    eye_gain_scale: float = 1.0
    tonic_add: dict = field(default_factory=dict)    # cell type -> added tonic current
    tau_scale: dict = field(default_factory=dict)    # cell type -> membrane tau multiplier
    delay_steps: dict = field(default_factory=dict)  # cell type -> presynaptic delay (steps)

    def is_reference(self) -> bool:
        return (self.eye_gain_scale == 1.0 and not self.tonic_add
                and not self.tau_scale and not self.delay_steps)

    def to_json(self) -> dict:
        return asdict(self)


def expand_types(cell_type: np.ndarray, spec: dict) -> dict:
    """Map a {cell type or prefix*: value} spec to {value: neuron index array}."""
    ct = cell_type.astype(str)
    out = {}
    for key, value in spec.items():
        mask = np.char.startswith(ct, key[:-1]) if key.endswith("*") else (ct == key)
        idx = np.flatnonzero(mask)
        if len(idx):
            out.setdefault(float(value), []).append(idx)
    return {v: np.concatenate(parts) for v, parts in out.items()}


def build_repaired(flybrain_cls, config: OpticDynamicsConfig, **kwargs):
    """Construct a RepairedFlyBrain subclass instance bound to the given FlyBrain."""

    class RepairedFlyBrain(flybrain_cls):
        def __init__(self, cfg, **kw):
            self._cfg = cfg
            super().__init__(**kw)

        def _setup_dynamics(self):
            xp = self.xp
            cfg = self._cfg
            ct = self.cell_type
            # per-neuron decay from population tau multipliers
            tau_mult = np.ones(self.n, np.float32)
            for value, idx in expand_types(ct, cfg.tau_scale).items():
                tau_mult[idx] = value
            decay = np.exp(-self.dt / (self.tau * tau_mult)).astype(np.float32)
            self._decay_vec = xp.asarray(decay).reshape(-1, 1)
            self._uniform_decay = bool(np.allclose(decay, float(self.decay)))
            # per-neuron tonic
            tonic = np.full(self.n, float(self.tonic), np.float32)
            for value, idx in expand_types(ct, cfg.tonic_add).items():
                tonic[idx] += value
            self._tonic_vec = xp.asarray(tonic).reshape(-1, 1)
            self._uniform_tonic = bool(np.allclose(tonic, float(self.tonic)))
            # presynaptic delays
            self._delay_mask = None
            self._max_delay = 0
            if cfg.delay_steps:
                delay = np.zeros(self.n, np.int32)
                for value, idx in expand_types(ct, cfg.delay_steps).items():
                    delay[idx] = int(value)
                self._max_delay = int(delay.max())
                if self._max_delay > 0:
                    # one boolean mask per distinct positive delay
                    self._delay_mask = {int(d): xp.asarray(delay == d)
                                        for d in sorted(set(delay.tolist())) if d > 0}
            self._spike_history = []

        def reset(self, seed=None):
            super().reset(seed)
            # dynamics state is (re)built and temporal buffers cleared on every reset
            if not hasattr(self, "_cfg"):
                return
            self._setup_dynamics()
            self._spike_history = []

        def _effective_spikes(self, fired):
            """Spike vector with delayed populations replaced by their past spikes."""
            xp = self.xp
            s = xp.zeros((self.n, self.batch), xp.float32)
            if len(fired):
                s.ravel()[fired] = 1.0
            if self._delay_mask:
                self._spike_history.append(s.copy())
                if len(self._spike_history) > self._max_delay + 1:
                    self._spike_history.pop(0)
                eff = s.copy()
                for d, mask in self._delay_mask.items():
                    past = self._spike_history[-(d + 1)] if len(self._spike_history) > d else None
                    eff[mask] = past[mask] if past is not None else 0.0
                return eff
            return s

        def _current_from(self, spikes):
            if self.device == "cuda":
                if self.batch == 1:
                    return (self._W @ spikes[:, 0])[:, None]
                return self._W @ spikes
            from flybrain.brain import _propagate
            cols = np.flatnonzero(spikes.ravel() > 0)
            rows, c = np.divmod(cols, self.batch)
            return np.column_stack([_propagate(self.indptr, self.indices, self.weights,
                                               rows[c == b], self.n) for b in range(self.batch)])

        def step(self, eye_drive=None, inject=()):
            cfg = self._cfg
            if cfg.is_reference():
                return super().step(eye_drive=eye_drive, inject=inject)
            xp, B = self.xp, self.batch
            spikes = self._effective_spikes(self.fired)
            current = self._current_from(spikes) * self.gain
            self.v *= self._decay_vec
            self.v += current + self._tonic_vec
            # identical RNG consumption pattern to the reference model
            self.v += (self.rng.random((self.n, B)) < self.noise_hz * self.dt) * np.float32(self.noise_amp)
            if eye_drive is not None:
                drive = xp.asarray(eye_drive, dtype=xp.float32)
                self.v[self._visual] += (drive[:, None] if drive.ndim == 1 else drive) * \
                    (self.eye_gain * cfg.eye_gain_scale)
            for idx, amount in inject:
                self.v[xp.asarray(idx)] += self._amount(amount)
            if self.refractory_steps:
                self.v[(self.steps - self.last_spike) <= self.refractory_steps] = 0.0
            fired = xp.flatnonzero(self.v >= 1.0)
            self.v.ravel()[fired] = 0.0
            if self.refractory_steps:
                self.last_spike.ravel()[fired] = self.steps
            self.fired = fired
            self.steps += 1
            flat = fired if xp is np else fired.get()
            if B == 1:
                return flat
            rows, cols = np.divmod(flat, B)
            order = np.argsort(cols, kind="stable")
            return np.split(rows[order], np.cumsum(np.bincount(cols, minlength=B))[:-1])

    return RepairedFlyBrain(config, **kwargs)


# ---- Experiment-17 preregistered candidate configurations ----
# Lamina tonic level is chosen on calibration seeds only (see preregistration).
def candidates(lamina_tonic: float, delay_targets=("Mi9", "Mi4", "CT1")):
    slow = {t: 3.0 for t in delay_targets}
    lam = {"L1": lamina_tonic, "L2": lamina_tonic, "L3": lamina_tonic, "L5": lamina_tonic}
    return [
        OpticDynamicsConfig(name="A_reference"),
        OpticDynamicsConfig(name="C_lamina_tonic", tonic_add=lam),
        OpticDynamicsConfig(name="D_tonic_slow_inhibition", tonic_add=lam, tau_scale=slow),
        OpticDynamicsConfig(name="E_tonic_delay1", tonic_add=lam,
                            delay_steps={t: 1 for t in delay_targets}),
        OpticDynamicsConfig(name="F_tonic_delay2", tonic_add=lam,
                            delay_steps={t: 2 for t in delay_targets}),
        OpticDynamicsConfig(name="G_tonic_slow_delay2", tonic_add=lam, tau_scale=slow,
                            delay_steps={t: 2 for t in delay_targets}),
    ]
