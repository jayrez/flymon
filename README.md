# MaleCNS on the Tesla P40

`malecns` is an R client for the published male fly central nervous system connectome. It fetches metadata and neuron skeletons from neuPrint. It is not a fly simulation or a Pokemon controller. Its queries and R parsing use the network and CPU; the P40 is available for a separate model you may build from the data.

## Run

Put a neuPrint access token in `.env` as `NEUPRINT_TOKEN=...`. The file is ignored by Git. Then run:

```sh
docker compose build
docker compose run --rm flybrain
```

The benchmark reports wall clock time to fetch PN metadata and three neuron skeletons from `male-cns:v1.0`. Results depend on the neuPrint service and network. It prints counts and timings, not the token. If the image is already built, skip the build.

Check CUDA and benchmark a simple GPU operation separately:

```sh
docker compose run --rm flybrain python3 benchmark_gpu.py
```

This GPU result shows whether PyTorch can compute on the P40. It does not measure `malecns` or predict Pokemon performance. For the Pokemon project, the next step is to define a game observation and action interface and a model/controller that maps observations to actions. The connectome can inform that model, but the package does not provide one.

Upstream package: https://github.com/natverse/malecns
