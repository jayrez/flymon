"""Streaming audit artifacts for Experiment 14.

This module only observes frames and serialized decision records. It has no
controller reference and cannot influence an action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import numpy as np
from PIL import Image


FRAME_WIDTH = 160
FRAME_HEIGHT = 144
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 288


def checkpoint_decisions(horizon: int, interval: int = 250) -> list[int]:
    """Return initial, fixed-interval, and final checkpoint boundaries."""
    if horizon < 1 or interval < 1:
        raise ValueError("horizon and checkpoint interval must be positive")
    values = list(range(0, horizon + 1, interval))
    if values[-1] != horizon:
        values.append(horizon)
    return values


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, destination)


def save_checkpoint(frame: np.ndarray, path: str | Path) -> None:
    values = np.asarray(frame)
    if values.dtype != np.uint8 or values.shape not in {
            (FRAME_HEIGHT, FRAME_WIDTH, 3), (FRAME_HEIGHT, FRAME_WIDTH, 4)}:
        raise ValueError(f"unexpected checkpoint framebuffer {values.shape} {values.dtype}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values, "RGB" if values.shape[-1] == 3 else "RGBA").save(destination)


class StreamingMetricsWriter:
    """Write one gzip JSONL row at a time and publish only on clean close."""
    def __init__(self, path: str | Path, compresslevel: int = 6):
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + ".partial")
        self.compresslevel = compresslevel
        self.handle = None
        self.rows = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(
            self.partial, "wt", encoding="utf-8", newline="\n",
            compresslevel=self.compresslevel)
        return self

    def write(self, row: dict) -> None:
        if self.handle is None:
            raise RuntimeError("metrics writer is not open")
        if int(row.get("decision", -1)) != self.rows:
            raise ValueError(
                f"metric decision must be contiguous: {row.get('decision')} != {self.rows}")
        self.handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False))
        self.handle.write("\n")
        self.rows += 1

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if exc_type is None:
            os.replace(self.partial, self.path)
        return False


def _video_partial(path: Path) -> Path:
    return path.with_name(path.stem + ".partial" + path.suffix)


class FFmpegVideoRecorder:
    """Stream in-memory RGB24 frames directly to a compact H.264 MP4."""
    def __init__(self, path: str | Path, fps: float = 5.0, *,
                 popen_factory=subprocess.Popen, ffmpeg: str | None = None):
        self.path = Path(path)
        self.partial = _video_partial(self.path)
        self.fps = float(fps)
        self.popen_factory = popen_factory
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg")
        self.process = None
        self.frames = 0
        if self.fps <= 0:
            raise ValueError("video fps must be positive")

    @staticmethod
    def check_available() -> None:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            raise RuntimeError("Experiment 14 requires ffmpeg and ffprobe")
        check = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"], check=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if check.returncode or "libx264" not in check.stdout:
            raise RuntimeError("FFmpeg libx264 encoder is unavailable")

    def command(self) -> list[str]:
        if self.ffmpeg is None:
            raise RuntimeError("ffmpeg executable is unavailable")
        return [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "-r", f"{self.fps:g}", "-i", "pipe:0",
            "-vf", f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:flags=neighbor",
            "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "32", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(self.partial),
        ]

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.process = self.popen_factory(
            self.command(), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin pipe was not created")
        return self

    def write(self, rgb: np.ndarray) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("video recorder is not open")
        frame = np.asarray(rgb)
        if frame.dtype != np.uint8 or frame.shape != (
                FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise ValueError(f"expected RGB24 144x160 frame, got {frame.shape} {frame.dtype}")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def __exit__(self, exc_type, exc, traceback):
        if self.process is None:
            return False
        process = self.process
        self.process = None
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if exc_type is not None:
            process.terminate()
        return_code = process.wait()
        stderr = process.stderr.read().decode("utf-8", errors="replace") \
            if process.stderr is not None else ""
        if exc_type is None:
            if return_code:
                raise RuntimeError(f"FFmpeg failed ({return_code}): {stderr.strip()}")
            if not self.partial.is_file():
                raise RuntimeError("FFmpeg did not create its MP4 output")
            os.replace(self.partial, self.path)
        return False


def _ratio(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def probe_video(path: str | Path, expected_frames: int | None = None) -> dict:
    source = Path(path)
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is unavailable")
    command = [
        ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames,nb_read_frames:format=duration,size",
        "-of", "json", str(source),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"FFprobe failed: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    if len(payload.get("streams", [])) != 1:
        raise RuntimeError("expected exactly one video stream")
    stream = payload["streams"][0]
    count_value = stream.get("nb_read_frames", stream.get("nb_frames"))
    frames = None if count_value in {None, "N/A"} else int(count_value)
    metadata = {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": _ratio(stream.get("r_frame_rate")),
        "duration_seconds": float(payload["format"]["duration"]),
        "encoded_frame_count": frames,
        "file_size": source.stat().st_size,
        "sha256": sha256_file(source),
        "verified": False,
    }
    valid = (metadata["codec"] == "h264" and metadata["width"] == VIDEO_WIDTH
             and metadata["height"] == VIDEO_HEIGHT and metadata["fps"] == 5.0)
    if expected_frames is not None:
        valid = valid and frames == expected_frames
    if not valid:
        raise RuntimeError(f"invalid Experiment 14 MP4 metadata: {metadata}")
    metadata["verified"] = True
    return metadata


@dataclass
class ProgressionTracker:
    """Hash-only online counters; these values cannot affect the controller."""
    unique: set[str] = field(default_factory=set)
    previous: str | None = None
    current_run: int = 0
    longest_run: int = 0
    transitions: int = 0

    def update(self, coarse_state_hash: str) -> dict:
        if not isinstance(coarse_state_hash, str) or len(coarse_state_hash) != 64:
            raise ValueError("coarse state must be a SHA-256 hex string")
        self.unique.add(coarse_state_hash)
        if self.previous is None:
            self.current_run = 1
        elif coarse_state_hash == self.previous:
            self.current_run += 1
        else:
            self.transitions += 1
            self.current_run = 1
        self.previous = coarse_state_hash
        self.longest_run = max(self.longest_run, self.current_run)
        return {
            "unique_coarse_states_so_far": len(self.unique),
            "coarse_transition_count_so_far": self.transitions,
            "repeated_state_run_length": self.current_run,
            "longest_repeated_state_run_so_far": self.longest_run,
        }


def validate_metrics(path: str | Path, *, horizon: int, condition: str,
                     seed: int, allowed_actions: Iterable[str | None],
                     forbid_a: bool = False) -> dict:
    allowed = set(allowed_actions)
    rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("decision") != rows:
                raise RuntimeError(f"noncontiguous metric decision at row {rows}")
            if row.get("condition") != condition or int(row.get("seed")) != seed:
                raise RuntimeError(f"metric trial identity mismatch at row {rows}")
            action = row.get("action")
            if action not in allowed:
                raise RuntimeError(f"forbidden action in metric row: {action}")
            if action == "DOWN":
                raise RuntimeError("DOWN found in Experiment 14 metrics")
            if forbid_a and action == "A":
                raise RuntimeError("A found in A-disabled metrics")
            if row.get("controller", {}).get("selected_action") != action:
                raise RuntimeError(f"controller/action mismatch at row {rows}")
            rows += 1
    if rows != horizon:
        raise RuntimeError(f"metric row count {rows} != {horizon}")
    return {
        "passed": True,
        "row_count": rows,
        "decisions_contiguous": True,
        "condition_correct": True,
        "seed_correct": True,
        "actions_valid": True,
        "a_forbidden": bool(forbid_a),
        "down_absent": True,
    }
