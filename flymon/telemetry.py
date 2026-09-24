"""Read-only stream telemetry (E30): message schema, hub and a local HTTP/SSE server for OBS.

Data flows one way only: simulation worker -> supervisor -> TelemetryHub -> browser overlay.
The server exposes GET endpoints only (no POST/PUT), and the worker never reads from it, so the
overlay cannot influence the controller. Frames travel as JPEG on /frame.jpg and /stream.mjpg,
never inside JSON.
"""
from __future__ import annotations

import io
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCHEMA_VERSION = 1
OVERLAY_DIR = Path(__file__).resolve().parents[1] / "overlay"

DECISION_FIELDS = {
    "type": str, "schema": int, "ts": float, "episode": int, "seed": int, "decision": int, "episode_runtime_s": float,
    "uptime_s": float, "decisions_per_second": float, "speed_ratio": float, "action": str, "recent_actions": list,
    "controller": dict, "t4": dict, "visual_pathway": dict, "game": dict, "milestone": dict, "unique_tiles": int,
    "watchdog": dict, "frame_sha1": str,
}
CONTROLLER_KEYS = ("top_action", "probabilities", "features", "sha256", "frozen", "kind")


def validate(msg):
    """Check a decision message against the documented schema (raises ValueError)."""
    if msg.get("type") != "decision":
        return msg
    for k, t in DECISION_FIELDS.items():
        if k not in msg:
            raise ValueError(f"missing field {k}")
        if t is float and isinstance(msg[k], int):
            continue
        if not isinstance(msg[k], t):
            raise ValueError(f"field {k} has type {type(msg[k]).__name__}, expected {t.__name__}")
    for k in CONTROLLER_KEYS:
        if k not in msg["controller"]:
            raise ValueError(f"controller.{k} missing")
    json.dumps(msg)
    return msg


def schema_document():
    return dict(
        schema=SCHEMA_VERSION, transport=dict(sse="/events (text/event-stream, one JSON object per 'data:' line)",
                                              snapshot="/state (latest decision message + stream stats)",
                                              frame="/frame.jpg (latest frame, JPEG, 4x) and /stream.mjpg (multipart JPEG)",
                                              provenance="/provenance", schema="/schema",
                                              pages="/overlay (broadcast, transparent) and /dashboard (research)",
                                              direction="GET only; one-way worker -> overlay"),
        decision_message={k: t.__name__ for k, t in DECISION_FIELDS.items()},
        field_notes=dict(
            action="NONE | UP | DOWN | LEFT | RIGHT | A | B (controller output)",
            controller="top_action, probabilities (7 softmax outputs), features (35 controller inputs), genome sha256, "
                       "frozen=true, kind",
            t4="mean pooled frozen-T4 activity per subtype (T4a-d), what the controller reads",
            visual_pathway="status, mode (normal|none|shuffled), sensory_sha256",
            game="EVALUATOR/OVERLAY ONLY (RAM): map, map_name, x, y, window, battle, dialogue window run length",
            milestone="EVALUATOR/OVERLAY ONLY: current, furthest_episode, furthest_all_time",
            watchdog="healthy, restarts, interventions, last (supervisor view)",
            speed_ratio="emulated game frames per wall second / 60"),
        other_messages=dict(episode_end="type, episode, seed, decisions, fitness, furthest_milestone, end_reason, "
                                        "watchdog, dialogue summary", stats="type, stream stats from the SQLite store",
                            watchdog="type, kind, detail"))


class TelemetryHub:
    def __init__(self, provenance=None):
        self.lock = threading.Lock()
        self.latest = {}; self.stats = {}; self.frame = None; self.recent_events = []
        self.subs = set(); self.provenance = provenance or {}
        self.frame_cv = threading.Condition()

    def publish(self, msg):
        validate(msg)
        with self.lock:
            if msg["type"] == "decision":
                self.latest = msg
            elif msg["type"] == "stats":
                self.stats = msg
            else:
                self.recent_events = (self.recent_events + [msg])[-50:]
            subs = list(self.subs)
        data = json.dumps(msg, default=float)
        for q in subs:
            try:
                q.put_nowait(data)
            except queue.Full:
                pass

    def publish_frame(self, jpeg_bytes):
        with self.frame_cv:
            self.frame = jpeg_bytes
            self.frame_cv.notify_all()

    def subscribe(self):
        q = queue.Queue(maxsize=256)
        with self.lock:
            self.subs.add(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.subs.discard(q)

    def snapshot(self):
        with self.lock:
            return dict(latest=self.latest, stats=self.stats, recent_events=self.recent_events[-10:])


def make_handler(hub: TelemetryHub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store"); self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            try:
                if p in ("/", "/overlay", "/dashboard"):
                    name = "dashboard.html" if p == "/dashboard" else "overlay.html"
                    return self._send(200, (OVERLAY_DIR / name).read_bytes(), "text/html; charset=utf-8")
                if p.startswith("/static/"):
                    f = OVERLAY_DIR / Path(p[len("/static/"):]).name
                    ctype = "text/css" if f.suffix == ".css" else "application/javascript"
                    return self._send(200, f.read_bytes(), ctype)
                if p == "/state":
                    return self._send(200, json.dumps(hub.snapshot(), default=float).encode(), "application/json")
                if p == "/provenance":
                    return self._send(200, json.dumps(hub.provenance).encode(), "application/json")
                if p == "/schema":
                    return self._send(200, json.dumps(schema_document()).encode(), "application/json")
                if p == "/frame.jpg":
                    if hub.frame is None:
                        return self._send(204, b"", "image/jpeg")
                    return self._send(200, hub.frame, "image/jpeg")
                if p == "/events":
                    return self._sse()
                if p == "/stream.mjpg":
                    return self._mjpeg()
                return self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                return None

        def _sse(self):
            q = hub.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            try:
                snap = hub.snapshot()
                for m in [snap["stats"], snap["latest"]]:
                    if m:
                        self.wfile.write(f"data: {json.dumps(m, default=float)}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        data = q.get(timeout=15)
                        self.wfile.write(f"data: {data}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                hub.unsubscribe(q)

        def _mjpeg(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            try:
                last = None
                while True:
                    with hub.frame_cv:
                        hub.frame_cv.wait_for(lambda: hub.frame is not None and hub.frame is not last, timeout=5)
                        f = hub.frame
                    if f is None or f is last:
                        continue
                    last = f
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(f)).encode()
                                     + b"\r\n\r\n" + f + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def do_POST(self):
            self._send(405, b"read-only telemetry", "text/plain")

        do_PUT = do_DELETE = do_PATCH = do_POST

    return Handler


class TelemetryServer:
    def __init__(self, hub, host="127.0.0.1", port=8765):
        self.httpd = ThreadingHTTPServer((host, port), make_handler(hub))
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def port(self):
        return self.httpd.server_address[1]

    def start(self):
        self.thread.start(); return self

    def stop(self):
        self.httpd.shutdown(); self.httpd.server_close()


def encode_jpeg(frame, scale=4, quality=85):
    from PIL import Image
    im = Image.fromarray(frame[..., :3]).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
