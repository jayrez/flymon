"""Deterministic Experiment 14 recording and frozen-controller checks."""
from __future__ import annotations

import gzip
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from flymon.exploration import (
    ExplorationConfig, FrozenExplorationController, FrozenInterfaceController)
from flymon.interaction import (
    FrozenVectorBaseline, PopulationEventConfig, PopulationEventController)
from flymon.locomotion import PopulationBaseline
from flymon.progression import (
    FFmpegVideoRecorder, ProgressionTracker, StreamingMetricsWriter,
    atomic_write_json, checkpoint_decisions, sha256_file, validate_metrics)
from flymon.steering import SteeringBaseline


class NonClosingBytesIO(io.BytesIO):
    def close(self):
        self.flush()


class FakeProcess:
    def __init__(self, output: Path):
        self.stdin = NonClosingBytesIO()
        self.stderr = NonClosingBytesIO()
        self.output = output
        self.terminated = False
        output.write_bytes(b"mock-mp4")

    def wait(self):
        return 0

    def terminate(self):
        self.terminated = True


class ProgressionTests(unittest.TestCase):
    def test_checkpoint_predicate_and_final(self):
        self.assertEqual(
            checkpoint_decisions(1500), [0, 250, 500, 750, 1000, 1250, 1500])
        self.assertEqual(checkpoint_decisions(601), [0, 250, 500, 601])

    def test_progression_tracker_counts_only_hashes(self):
        tracker = ProgressionTracker()
        a, b = "a" * 64, "b" * 64
        states = [tracker.update(value) for value in (a, a, b, b, a)]
        self.assertEqual(states[-1]["unique_coarse_states_so_far"], 2)
        self.assertEqual(states[-1]["coarse_transition_count_so_far"], 2)
        self.assertEqual(states[1]["repeated_state_run_length"], 2)
        self.assertEqual(states[-1]["longest_repeated_state_run_so_far"], 2)

    @staticmethod
    def row(decision, action=None, condition="live_full"):
        return {
            "decision": decision, "condition": condition, "seed": 921,
            "action": action, "controller": {"selected_action": action},
        }

    def test_streaming_metrics_one_row_per_decision_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.jsonl.gz"
            with StreamingMetricsWriter(path) as writer:
                for decision in range(3):
                    writer.write(self.row(decision, "LEFT" if decision == 1 else None))
            result = validate_metrics(
                path, horizon=3, condition="live_full", seed=921,
                allowed_actions={None, "LEFT", "RIGHT", "UP", "A"})
            self.assertEqual(result["row_count"], 3)
            with gzip.open(path, "rt") as handle:
                self.assertEqual(len(handle.readlines()), 3)

    def test_metrics_writer_rejects_missing_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl.gz"
            with self.assertRaises(ValueError):
                with StreamingMetricsWriter(path) as writer:
                    writer.write(self.row(1))

    def test_a_disabled_and_down_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            a_path = Path(directory) / "a.jsonl.gz"
            with StreamingMetricsWriter(a_path) as writer:
                writer.write(self.row(0, "A", "live_no_a"))
            with self.assertRaises(RuntimeError):
                validate_metrics(
                    a_path, horizon=1, condition="live_no_a", seed=921,
                    allowed_actions={None, "LEFT", "RIGHT", "UP", "A"},
                    forbid_a=True)
            down_path = Path(directory) / "down.jsonl.gz"
            with StreamingMetricsWriter(down_path) as writer:
                writer.write(self.row(0, "DOWN"))
            with self.assertRaises(RuntimeError):
                validate_metrics(
                    down_path, horizon=1, condition="live_full", seed=921,
                    allowed_actions={None, "LEFT", "RIGHT", "UP", "A", "DOWN"})

    def test_video_recorder_streams_rgb_without_raw_files(self):
        created = []
        def factory(command, **_):
            process = FakeProcess(Path(command[-1]))
            created.append((command, process))
            return process
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.mp4"
            with FFmpegVideoRecorder(
                    path, fps=5, popen_factory=factory, ffmpeg="ffmpeg") as recorder:
                recorder.write(np.zeros((144, 160, 3), np.uint8))
            command, process = created[0]
            self.assertIn("pipe:0", command)
            self.assertEqual(len(process.stdin.getvalue()), 144 * 160 * 3)
            self.assertTrue(path.is_file())
            self.assertFalse(list(Path(directory).glob("*.partial.mp4")))
            self.assertFalse(list(Path(directory).glob("*.raw")))

    def test_video_recorder_rejects_non_rgb_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            created = []
            def factory(command, **_):
                process = FakeProcess(Path(command[-1])); created.append(process); return process
            with self.assertRaises(ValueError):
                with FFmpegVideoRecorder(
                        Path(directory) / "trial.mp4", popen_factory=factory,
                        ffmpeg="ffmpeg") as recorder:
                    recorder.write(np.zeros((144, 160, 4), np.uint8))

    def test_atomic_manifest_and_file_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            atomic_write_json(path, {"size": 17})
            self.assertEqual(json.loads(path.read_text()), {"size": 17})
            self.assertEqual(len(sha256_file(path)), 64)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

        manifest = json.loads(Path(
            "results/experiment-14-progression/trial-manifest.json").read_text())
        self.assertEqual(len(manifest["trials"]), 41)
        for trial in manifest["trials"]:
            self.assertEqual(len(trial["metrics_sha256"]), 64)
            self.assertGreater(trial["metrics_compressed_size"], 0)
            self.assertEqual(len(trial["video_sha256"]), 64)
            self.assertGreater(trial["video_size"], 0)
            metrics = Path(trial["metrics_path"])
            self.assertEqual(metrics.stat().st_size, trial["metrics_compressed_size"])
            self.assertEqual(sha256_file(metrics), trial["metrics_sha256"])

    @staticmethod
    def controllers():
        steering = SteeringBaseline(0, 0, 1, 1, 0, 1, 10)
        locomotion = PopulationBaseline(0, 1, 0, 1, 10, (0,), (0,))
        config = ExplorationConfig(2, 1, "L", 2, 1, 2, 1, 1)
        direction = FrozenExplorationController(
            steering, locomotion, config, enable_down=False)
        event = PopulationEventController(
            FrozenVectorBaseline((0., 0.), (0., 0.), 10),
            PopulationEventConfig("norm", "rising", 1.5, 1., 1, 1, .5, 5))
        return FrozenInterfaceController(direction, None, event)

    def test_recording_does_not_change_controller_output(self):
        rates = {
            "DNa02_L": 4., "DNa02_R": 0., "DNg100": 0., "MDN": 99.,
            "event_population_rates_hz": (0., 0.),
        }
        expected = self.controllers().decode(rates)
        tracker = ProgressionTracker()
        tracker.update("c" * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl.gz"
            with StreamingMetricsWriter(path) as writer:
                writer.write(self.row(0, expected[0]))
        observed = self.controllers().decode(rates)
        self.assertEqual(expected, observed)
        self.assertNotEqual(observed[0], "DOWN")

    def test_no_per_decision_png_or_raw_frame_writer(self):
        source = Path("run_progression_experiment.py").read_text().lower()
        self.assertNotIn("raw_frames", source)
        self.assertNotIn("decision_frames", source)
        self.assertNotIn(".npy", source)
        self.assertIn("if boundary in checkpoints", source)
        self.assertIn("previous_result = initial", source)

    def test_controller_surface_has_no_framebuffer_or_ram(self):
        for cls in (
                FrozenExplorationController, FrozenInterfaceController,
                PopulationEventController):
            self.assertEqual(list(inspect.signature(cls.decode).parameters), ["self", "rates"])
            source = inspect.getsource(cls.decode).lower()
            for forbidden in (
                    "framebuffer", "frame_hash", "coarse_state", "read_memory",
                    "player_position", "map_id", "reward"):
                self.assertNotIn(forbidden, source)
        runner = Path("run_progression_experiment.py").read_text().lower()
        self.assertNotIn("read_memory", runner)
        self.assertNotIn("memory[", runner)


if __name__ == "__main__":
    unittest.main()
