# Tesla P40 FlyBrain benchmark

This directory benchmarks [fly.ai](https://github.com/alextitonis/fly.ai) `flybrain` 0.1.0 with its verified prebuilt MaleCNS v1.0 data. The Python environment and downloaded data stay in `.venv/` and `data/`, respectively. Neither is committed.

## Setup

```sh
python3 -m venv --without-pip .venv
python3 -m pip --python .venv install pip
.venv/bin/python -m pip install "flybrain[gpu]"
FLY_DATA="$PWD/data" .venv/bin/flybrain download
FLY_DATA="$PWD/data" .venv/bin/flybrain info
```

On this machine the system `ensurepip` component was missing. Setup used `python3 -m venv --without-pip .venv` followed by `python3 -m pip --python .venv install pip` to bootstrap pip inside the environment without changing system packages.

## Run

From this directory:

```sh
.venv/bin/python benchmark_flybrain.py sweep
.venv/bin/python benchmark_flybrain.py sustained --sustained-batch 1
.venv/bin/python benchmark_flybrain.py cpu
```

The script validates 166,700 neurons and 25,582,938 connections before timing. Each sweep row warms 100 steps and takes the median of three 1,000-step runs. It synchronizes CUDA around timing and samples GPU utilization, VRAM, temperature, clocks, power and active clock event reasons via `nvidia-smi`. `step()` includes FlyBrain's spike transfer to NumPy, so timings describe the public API, not only GPU kernels. The 50 Hz equivalents are a reference measure, not a requirement for a future game controller.

Results are written after each completed batch to `results/flybrain-p40-benchmark.csv` and `.json`. To resume a specific batch, pass `--batches 64`; completed batches are skipped. The sustained test runs for approximately 300 seconds in 30-second windows. CPU comparison measures batches 1 and 4 with the same model.
