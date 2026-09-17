# FlyBrain MaleCNS v1.0 benchmark — Tesla P40

Run on 17 September 2026. The public `flybrain` 0.1.0 CUDA implementation loaded the verified prebuilt MaleCNS v1.0 connectome: **166,700 neurons and 25,582,938 connections**. It ran on CuPy/CUDA (`device="cuda"`), advanced neural state, produced spikes, and passed finite-state checks after every measurement. The idle GPU showed 0% utilization; it rose during simulation.

## CUDA batch sweep

Each row is the median of **3 × 1,000 measured steps**, after **100 warm-up steps**. CUDA was synchronized before and after each timing. The public `step()` return includes spike transfer to NumPy and, for batches, CPU sorting. GPU utilization is the mean of `nvidia-smi` samples during measurement. VRAM is the peak device allocation reported by `nvidia-smi`.

| Batch | ms/batch step | ms/fly* | Batch steps/s | Brain steps/s | 50 Hz equivalents | VRAM MiB | GPU util |
| ----: | ------------: | ------: | ------------: | ------------: | -----------------: | -------: | -------: |
| 1 | 1.931 | 1.931 | 517.9 | 517.9 | 10.36 | 359 | 58.8% |
| 2 | 4.281 | 2.141 | 233.6 | 467.2 | 9.34 | 373 | 50.8% |
| 4 | 7.790 | 1.947 | 128.4 | 513.5 | 10.27 | 387 | 48.9% |
| 8 | 25.415 | 3.177 | 39.3 | 314.8 | 6.30 | 401 | 66.6% |
| 16 | 79.660 | 4.979 | 12.6 | 200.9 | 4.02 | 435 | 77.8% |
| 32 | 162.059 | 5.064 | 6.2 | 197.5 | 3.95 | 495 | 76.6% |
| 64 | 344.018 | 5.375 | 2.9 | 186.0 | 3.72 | 695 | 71.9% |

*`ms/fly` is batch step time divided by batch size, a throughput-derived cost. Actual latency per fly is the `ms/batch step` column. `50 Hz equivalents` is aggregate brain steps/s divided by 50; it does not promise that many independent workers with game overhead. `VRAM_delta_MB` is present in CSV/JSON and is relative to memory immediately before each batch load. Since batches 1–32 ran in one Python process and batch 64 in a fresh process, deltas include different amounts of CUDA context memory; use absolute VRAM for cross-batch comparison.

## Hardware and software

- GPU: Tesla P40, compute capability 6.1, 24,576 MiB VRAM; driver 580.178.04; driver-reported CUDA maximum 13.0. Idle usage before test: 0 MiB, 0%; no GPU consumers.
- CPU: Intel Xeon E5-2670 0 @ 2.60 GHz, 24 vCPUs under KVM; RAM: 62 GiB; Linux kernel 6.8.0-139-generic.
- Python 3.12.3 in a local virtual environment; flybrain 0.1.0, CuPy 14.2.0 with CUDA 12.9 runtime, NumPy 2.5.3, SciPy 1.18.1, Numba 0.67.0.
- No host GPU configuration, drivers, clocks, or power limits were changed.

## Sustained load

Batch 1 ran for 300.0 seconds after 100 warm-up steps. The ten 30-second windows ranged from **498.8 to 510.4 batch steps/s**; first and last were 503.8 and 506.8. No NaN/Inf or runtime errors occurred. Sampled GPU utilization was 51–63% (median 59%); VRAM stayed at 359 MiB. Temperature rose to 45°C (median 43°C), SM clock was usually 1531 MHz (minimum 1442 MHz), power peaked at 115.9 W, and the active clock-event flag was zero in all 290 samples. There was no observed thermal throttling or throughput degradation.

## CPU comparison

Using the same data and model, CPU batch 1 took **9.714 ms/step** and CPU batch 4 took **41.604 ms/step** (each median of 3 × 1,000 steps). CUDA was about **5.0×** faster for batch 1 and **5.3×** for batch 4. CPU mode used about 6.6–7.3 core equivalents; the CUDA Python process used about one core equivalent.

## Practical batch and RedFly recommendations

**Best useful batch: 1.** It delivered the highest measured aggregate throughput (517.9 brain steps/s). Batch 4 was close (513.5 brain steps/s) and kept batch latency below 20 ms (7.790 ms), so it is a workable upper bound for four lockstep 50 Hz brains. Batch 8 took 25.415 ms per step and therefore missed a 20 ms cadence; batches 16–64 became progressively slower. The P40 had abundant VRAM headroom even at 64 (695 MiB); memory capacity is not the practical constraint for this API.

**One persistent Pokémon-playing fly:** reserve one CUDA FlyBrain instance, approximately 359 MiB VRAM and 1.9–2.0 ms per 20 ms model step under this idle-host test. Budget one CPU core for Python orchestration plus separate resources for the game and encoder. Use a real scheduler to verify end-to-end latency when those parts exist.

**Parallel training environments:** begin with **2–4 total MaleCNS brains** on this GPU, and start with 2 if a live fly also shares it. Batch 4 is the largest measured batch that stays under a 20 ms neural step, but its aggregate throughput is about the same as batch 1. More environments may help only when game stepping and data collection overlap useful GPU work; test that later. A FlyBrain batch entry is only neural state: PyBoy, sensory encoding, action decoding, resets, and episode handling add CPU/GPU work and synchronization, so this result is not a Pokémon-worker count.

## Bottleneck assessment

Profiling the unmodified public `step()` path points to **spike extraction and CPU postprocessing**, with GPU work synchronized inside CuPy operations. At batch 8, 200 profiled steps spent 3.16 s in `cupy.flatnonzero`/`nonzero` and 0.89 s in NumPy `argsort`, out of 5.01 s total. At batch 64, 50 steps spent 12.30 s in `flatnonzero`/`nonzero` and 3.47 s in NumPy `argsort`, out of 17.80 s. The CuPy sparse-matrix call accounted for only 0.04 s in that profile, though asynchronous GPU work may be charged to a later synchronization point. Direct `.get()` time was 0.29 s. GPU utilization rose with batch size while VRAM stayed low, so neither VRAM capacity nor host-to-device transfer volume alone explains the scaling loss. The measured limit is this implementation’s large spike selection plus CPU sorting/synchronization path; a kernel-level profile would separate GPU selection work from synchronization more precisely.

## Next benchmark

Once PyBoy is added, measure complete observation → sensory encoding → FlyBrain step → readout → action → PyBoy step latency, CPU use, frame throughput, and simultaneous episode scaling. Compare one live fly with 2–4 training environments sharing the P40. No game or training integration was built in this benchmark.

Raw measurements and GPU telemetry: [`flybrain-p40-benchmark.json`](flybrain-p40-benchmark.json); flat table: [`flybrain-p40-benchmark.csv`](flybrain-p40-benchmark.csv).
