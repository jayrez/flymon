"""Benchmark the published MaleCNS v1.0 FlyBrain model on CUDA and CPU."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import threading
import time
import resource

import numpy as np
from scipy import sparse
from flybrain import FlyBrain
import flybrain

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("FLY_DATA", str(ROOT / "data"))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
CSV_PATH = RESULTS / "flybrain-p40-benchmark.csv"
JSON_PATH = RESULTS / "flybrain-p40-benchmark.json"
GPU_QUERY = "utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw,clocks_event_reasons.active"
FIELDS = ["device", "batch_size", "total_step_time_ms", "ms_per_batch_step", "ms_per_fly_per_step",
          "batch_steps_per_second", "effective_brain_steps_per_second", "realtime_50hz_equivalents",
          "VRAM_used_MB", "VRAM_delta_MB", "GPU_utilization_percent", "CPU_core_equivalents",
          "spikes_per_step", "repeat_count", "steps_per_repeat"]


def gpu_sample():
    output = subprocess.check_output([
        "nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"
    ], text=True, timeout=10).strip().splitlines()[0]
    parts = [x.strip() for x in output.split(",")]
    return dict(zip(["gpu_util", "memory_mb", "temp_c", "sm_mhz", "power_w", "throttle"],
                    [float(x) if i < 5 else x for i, x in enumerate(parts)]))


class Monitor:
    def __init__(self, interval=0.25):
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.samples.append(dict(time=time.time(), **gpu_sample()))
            except (subprocess.SubprocessError, ValueError, IndexError):
                pass
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()


def synchronize(device):
    if device == "cuda":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()


def check_state(brain):
    if brain.device == "cuda":
        import cupy as cp
        ok = bool(cp.isfinite(brain.v).all().get())
    else:
        ok = bool(np.isfinite(brain.v).all())
    if not ok:
        raise RuntimeError("FlyBrain state contains NaN or Inf")


def step_count(brain):
    fired = brain.step()
    return len(fired) if brain.batch == 1 else sum(map(len, fired))


def run_repeat(brain, steps, sample_gpu=True):
    synchronize(brain.device)
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    monitor = Monitor() if sample_gpu else None
    spikes = 0
    if monitor:
        with monitor:
            for _ in range(steps):
                spikes += step_count(brain)
            synchronize(brain.device)
    else:
        for _ in range(steps):
            spikes += step_count(brain)
        synchronize(brain.device)
    elapsed = time.perf_counter() - start
    cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = (cpu_after.ru_utime + cpu_after.ru_stime
                   - cpu_before.ru_utime - cpu_before.ru_stime)
    check_state(brain)
    return dict(elapsed=elapsed, spikes=spikes,
                cpu_cores=cpu_seconds / elapsed,
                samples=monitor.samples if monitor else [])


def version_info():
    import cupy as cp
    import numba
    import scipy
    return dict(python=platform.python_version(), platform=platform.platform(),
                flybrain=flybrain.__version__ if hasattr(flybrain, "__version__") else "0.1.0",
                cupy=cp.__version__, numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                gpu_name=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                gpu_compute_capability=cp.cuda.Device(0).compute_capability)


def validate_data():
    data = Path(os.environ["FLY_DATA"])
    with np.load(data / "brain.npz") as meta:
        neurons = len(meta["cell_type"])
    weights = sparse.load_npz(data / "weights.npz")
    connections = weights.nnz
    if (neurons, connections) != (166700, 25582938):
        raise RuntimeError(f"Unexpected data: {neurons} neurons, {connections} connections")
    return dict(neurons=neurons, connections=connections, source="MaleCNS v1.0 prebuilt FlyBrain data")


def benchmark_batch(batch, device, steps, repeats, warmup):
    baseline = gpu_sample()["memory_mb"] if device == "cuda" else 0.0
    brain = FlyBrain(data=ROOT / "data", device=device, batch=batch)
    if brain.device != device or brain.n != 166700 or (device == "cuda" and brain._W.nnz != 25582938):
        raise RuntimeError("Brain device or connectome size mismatch")
    warm_spikes = sum(step_count(brain) for _ in range(warmup))
    synchronize(device)
    check_state(brain)
    if warm_spikes <= 0:
        raise RuntimeError("No spikes during warm-up")
    runs = [run_repeat(brain, steps, sample_gpu=(device == "cuda")) for _ in range(repeats)]
    times = [run["elapsed"] for run in runs]
    median_s = statistics.median(times)
    samples = [sample for run in runs for sample in run["samples"]]
    peak_memory = max([baseline] + [s["memory_mb"] for s in samples])
    row = dict(device=device, batch_size=batch,
               total_step_time_ms=median_s * 1000,
               ms_per_batch_step=median_s * 1000 / steps,
               ms_per_fly_per_step=median_s * 1000 / steps / batch,
               batch_steps_per_second=steps / median_s,
               effective_brain_steps_per_second=batch * steps / median_s,
               realtime_50hz_equivalents=batch * steps / median_s / 50,
               VRAM_used_MB=peak_memory if device == "cuda" else None,
               VRAM_delta_MB=peak_memory - baseline if device == "cuda" else None,
               GPU_utilization_percent=statistics.mean(s["gpu_util"] for s in samples) if samples else None,
               CPU_core_equivalents=statistics.median(run["cpu_cores"] for run in runs),
               spikes_per_step=statistics.median(run["spikes"] / steps for run in runs),
               repeat_count=repeats, steps_per_repeat=steps)
    detail = dict(row=row, repeats=[{k: v for k, v in run.items() if k != "samples"} for run in runs],
                  gpu_samples=samples, warmup_steps=warmup, warmup_spikes=warm_spikes)
    del brain
    if device == "cuda":
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    return detail


def save(state):
    JSON_PATH.write_text(json.dumps(state, indent=2) + "\n")
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        for detail in state["benchmarks"]:
            writer.writerow(detail["row"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["sweep", "sustained", "cpu", "all"])
    parser.add_argument("--sustained-batch", type=int)
    parser.add_argument("--batches", type=int, nargs="+", choices=[1, 2, 4, 8, 16, 32, 64],
                        help="GPU batches to run; completed batches are skipped")
    args = parser.parse_args()
    state = json.loads(JSON_PATH.read_text()) if JSON_PATH.exists() else {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": version_info(), "data": validate_data(), "benchmarks": [], "sustained": None}
    if args.phase in ("sweep", "all"):
        for batch in (args.batches or (1, 2, 4, 8, 16, 32, 64)):
            if any(d["row"]["device"] == "cuda" and d["row"]["batch_size"] == batch
                   for d in state["benchmarks"]):
                print(f"CUDA batch {batch}: already saved; skipping", flush=True)
                continue
            print(f"CUDA batch {batch}: loading, warming, measuring", flush=True)
            try:
                detail = benchmark_batch(batch, "cuda", 1000, 3, 100)
            except Exception as exc:
                state.setdefault("failures", []).append(dict(batch=batch, error=repr(exc)))
                save(state)
                raise
            state["benchmarks"].append(detail)
            save(state)
            print(f"  {detail['row']['ms_per_batch_step']:.3f} ms/step; "
                  f"{detail['row']['effective_brain_steps_per_second']:.1f} brain steps/s", flush=True)
    if args.phase in ("cpu", "all"):
        for batch in (1, 4):
            print(f"CPU batch {batch}: loading, warming, measuring", flush=True)
            detail = benchmark_batch(batch, "cpu", 1000, 3, 100)
            state["benchmarks"].append(detail)
            save(state)
            print(f"  {detail['row']['ms_per_batch_step']:.3f} ms/step", flush=True)
    if args.phase in ("sustained", "all"):
        gpu_rows = [d["row"] for d in state["benchmarks"] if d["row"]["device"] == "cuda"]
        if not gpu_rows:
            raise RuntimeError("Run sweep before sustained test")
        batch = args.sustained_batch or max(gpu_rows, key=lambda r: r["effective_brain_steps_per_second"])["batch_size"]
        print(f"Sustained CUDA batch {batch}: 300 seconds", flush=True)
        brain = FlyBrain(data=ROOT / "data", device="cuda", batch=batch)
        for _ in range(100):
            step_count(brain)
        synchronize("cuda")
        windows = []
        with Monitor(interval=1.0) as mon:
            start = time.perf_counter()
            while time.perf_counter() - start < 300:
                window_start = time.perf_counter()
                steps = 0
                spikes = 0
                while time.perf_counter() - window_start < 30:
                    spikes += step_count(brain)
                    steps += 1
                synchronize("cuda")
                elapsed = time.perf_counter() - window_start
                windows.append(dict(steps=steps, seconds=elapsed, batch_steps_per_second=steps / elapsed,
                                    spikes=spikes))
                print(f"  {len(windows) * 30}s: {steps / elapsed:.1f} batch steps/s", flush=True)
        check_state(brain)
        state["sustained"] = dict(batch_size=batch, duration_seconds=time.perf_counter() - start,
                                  windows=windows, gpu_samples=mon.samples)
        save(state)
    print(f"Saved {CSV_PATH} and {JSON_PATH}", flush=True)


if __name__ == "__main__":
    main()
