"""Live runtime for the recovered E25 controller (E30).

Two sides, never mixed:

* **Controller side** (`FrozenController`, `RandomController`): receives *only* game framebuffers
  through `observe_frames(frames)` and returns a decision from `decide()` (no arguments). It holds
  no emulator handle, reads no RAM and receives no evaluator, watchdog or milestone state. The
  E25 genome is frozen: its SHA-256 is verified at load and re-checked after every decision.
* **Environment / evaluator side** (`Episode`): owns the emulator, applies actions with the E24/E25
  timing (8 frames held + 4 released per decision), feeds the new frames to the controller and
  only then reads RAM telemetry for the evaluator (`flymon.stream_eval`).

The arithmetic is identical to E25 `run_evolution.run_task` (tested action-for-action against the
committed E25 held-out logs).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flymon import evolution as evo

ROOT = Path(__file__).resolve().parents[1]
HOLD, RELEASE = 8, 4                      # E24/E25 decision timing (frames)
FRAMES_PER_DECISION = HOLD + RELEASE
START_WAIT_MAX = 60
SHUFFLE_BASE = 25_000_000
CHAMPION_SHA256 = "9fd9795d01af51133ff26b349d186417e5611b452e722dafb4a8187fc1ad31df"
CHAMPION_PATH = ROOT / "results" / "e30-stream-readiness" / "champion-genome.json"
T4_NATIVE_SHA256_PREFIX = "87829806e398f66b"
VISION_MODES = ("normal", "none", "shuffled")


def load_champion(path=CHAMPION_PATH, expected=CHAMPION_SHA256):
    g = evo.Genome.from_json(json.loads(Path(path).read_text()))
    if g.sha256() != expected:
        raise ValueError("champion genome hash mismatch")
    return g


class SensoryContext:
    """Frozen E23 T4 pathway, loaded once and shared by controllers (read-only)."""

    def __init__(self):
        import run_fixed_background_experiment as E22
        from flymon.fast_io import FastColumnSampler
        from flymon.frozen_t4 import FrozenT4Readout
        self.E22 = E22
        self.ctx, self.const = E22.contexts()
        self.native = FrozenT4Readout(self.ctx, self.const)
        side = self.ctx["meta"]["side"][self.native.brain_index].astype(str)
        rf = np.concatenate([self.ctx["rf_uv"][s] for s in evo.T4_SUBTYPES])
        self.pooling = evo.T4Pooling(self.native.subtype, side, rf)
        self.sampler = FastColumnSampler(self.native._records, self.native._colindex, self.native._uv)
        self._shuffled = {}

    def readout(self, vision, seed):
        from flymon.frozen_t4 import FrozenT4Readout
        if vision == "none":
            return None
        if vision == "normal":
            return self.native
        if vision != "shuffled":
            raise ValueError(vision)
        if seed not in self._shuffled:
            proj = self.E22.permuted_projections(self.ctx, 1, SHUFFLE_BASE + seed)[0]
            self._shuffled = {seed: FrozenT4Readout(self.ctx, self.const, proj=proj,
                                                    geometry_label=f"column-shuffle-{SHUFFLE_BASE + seed}")}
        return self._shuffled[seed]


@dataclass
class Decision:
    action: str | None                       # None == NONE
    probabilities: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    t4_summary: dict = field(default_factory=dict)


def _check_frames(frames):
    for f in frames:
        if not (isinstance(f, np.ndarray) and f.shape == (144, 160, 4) and f.dtype == np.uint8):
            raise TypeError("controller accepts only 144x160 RGBA uint8 framebuffers")


class FrozenController:
    """E25 T4 linear-softmax decoder. Inputs: framebuffers only. Frozen: no learning."""

    kind = "genome"

    def __init__(self, genome: evo.Genome, sensory: SensoryContext, vision="normal"):
        if genome.arch != "t4":
            raise ValueError("the live controller is the E25 T4-only architecture")
        self.genome, self.sensory, self.vision = genome, sensory, vision
        self.sha256 = genome.sha256()
        self._W0 = genome.flat().copy()

    def reset(self, seed):
        self.readout = self.sensory.readout(self.vision, seed)
        if self.readout is not None:
            self.readout.reset()
        self.fb = evo.FeatureBuilder("t4", self.sensory.pooling)
        self.policy = evo.Policy(self.genome, np.random.default_rng([seed, 2501]))
        self.buf = []

    def sensory_sha256(self):
        return self.readout.sha256() if self.readout is not None else None

    def observe_frames(self, frames):
        _check_frames(frames)
        if self.readout is not None:
            s = self.sensory.sampler
            self.buf += [self.readout.update_from_luminance(s.luminance(f)) for f in frames]

    def decide(self) -> Decision:
        t4_mean = np.mean(self.buf, axis=0) if self.buf else None
        self.buf = []
        x = self.fb.build(t4_response=t4_mean)
        p = self.policy.probabilities(x)
        action = self.policy.act(x)
        self.fb.observe_action(action)
        if not np.array_equal(self.genome.flat(), self._W0):          # frozen during production
            raise RuntimeError("controller parameters changed at runtime")
        pooled = x[:len(evo.t4_feature_names()) // 2]
        return Decision(action=action, probabilities={a: float(v) for a, v in zip(evo.ACTIONS, p)},
                        features={n: float(v) for n, v in zip(self.fb.names, x)},
                        t4_summary={s: float(np.mean(pooled[2 * i:2 * i + 2])) for i, s in enumerate(evo.T4_SUBTYPES)})


class RandomController:
    """E25 baselines: uniform over the 7 actions, or the E24 C0 matched-rate law."""

    def __init__(self, kind):
        if kind not in ("random_uniform", "random_matched"):
            raise ValueError(kind)
        self.kind, self.vision, self.sha256 = kind, "none", None
        if kind == "random_matched":
            law = json.loads((ROOT / "results/experiment-24-generation-zero/c0-action-law.json").read_text())["law"]
            self.labels = list(law); p = np.array([law[k] for k in self.labels]); self.probs = p / p.sum()

    def reset(self, seed):
        self.rng = np.random.default_rng([seed, 2501])

    def sensory_sha256(self):
        return None

    def observe_frames(self, frames):
        _check_frames(frames)

    def decide(self) -> Decision:
        if self.kind == "random_uniform":
            a = evo.ACTIONS[int(self.rng.integers(0, len(evo.ACTIONS)))]
        else:
            a = self.labels[int(self.rng.choice(len(self.labels), p=self.probs))]
        return Decision(action=None if a == "NONE" else a)


class Episode:
    """Environment + evaluator side of one episode. The controller never sees `game` or RAM."""

    def __init__(self, game, controller, seed, budget, state_path=None, evaluator_factory=None,
                 realtime=False, fps=60.0):
        import run_generation_zero as G
        self.game, self.controller, self.seed, self.budget = game, controller, seed, budget
        self.state_path = state_path or G.STATE
        self.evaluator_factory = evaluator_factory
        self.realtime, self.fps = realtime, fps
        self.decision = 0

    def start(self):
        from flymon import gameplay_eval as ev
        g = self.game
        g.load_state(self.state_path); g.tick(1)
        wait = int(np.random.default_rng([self.seed, 2500]).integers(0, START_WAIT_MAX))
        frames = [g.framebuffer()]
        for _ in range(wait):
            g.tick(1); frames.append(g.framebuffer())
        self.start_wait = wait
        self.controller.reset(self.seed)
        self.controller.observe_frames(frames)
        tel0 = ev.read_telemetry(g)                               # evaluator only
        self.evaluator = self.evaluator_factory(tel0) if self.evaluator_factory else None
        self.last_frame = frames[-1]
        self._t_next = time.perf_counter()
        return self

    @property
    def done(self):
        return self.decision >= self.budget

    def step(self):
        from flymon import gameplay_eval as ev
        g = self.game
        dec = self.controller.decide()
        if dec.action in ("START", "SELECT"):
            raise RuntimeError("START/SELECT are disabled")
        if dec.action is not None:
            g.press(dec.action.lower())
        frames = []
        for k in range(FRAMES_PER_DECISION):
            if k == HOLD and dec.action is not None:
                g.release(dec.action.lower())
            g.tick(1); frames.append(g.framebuffer())
        self.controller.observe_frames(frames)
        tel = ev.read_telemetry(g)                                # evaluator only, after the action
        self.last_frame = frames[-1]
        ev_out = self.evaluator.update(self.decision, dec.action, tel, frames[-1]) if self.evaluator else None
        self.decision += 1
        if self.realtime:                                         # pace to the Game Boy frame rate
            self._t_next += FRAMES_PER_DECISION / self.fps
            dt = self._t_next - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                self._t_next = time.perf_counter()
        return dec, tel, ev_out

    def run(self):
        self.start()
        while not self.done:
            self.step()
        return self.evaluator.result() if self.evaluator else None


def frame_sha1(frame):
    return hashlib.sha1(np.ascontiguousarray(frame).tobytes()).hexdigest()
