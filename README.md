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

## Fly Brain Plays Pokémon: live controller architecture (E30)

A frozen fly-derived visual pathway turns each Pokémon frame into neural activity. A controller that was
evolved outside the fly model reads that activity and presses buttons. Nothing learns while it plays.

```
Game frame (PyBoy framebuffer, 160×144)
   ↓
Frozen T4 visual pathway (E23: retina sampling → Mi1/Mi4/Tm → T4, MaleCNS v1.0 weights; SHA-256 87829806…)
   ↓
Neural feature vector (14 pooled T4 signals + their changes + the controller's previous action = 35 inputs)
   ↓
Externally evolved controller (E25 linear softmax decoder, 253 parameters, frozen; SHA-256 9fd9795d…)
   ↓
Pokémon button (NONE, UP, DOWN, LEFT, RIGHT, A, B)

- - - - - - - - - - - - - -  controller boundary  - - - - - - - - - - - - - -
Game RAM → evaluator (fitness, milestones, dialogue/stall labels) → overlay, stats, watchdog
           RAM never enters the controller (tested in test_live_runtime.py)
```

What each component is:

- **Visual pathway:** the frozen T4 optic-lobe model. The whole-brain FlyBrain spiking simulation is not in
  this loop.
- **Controller:** trained by evolutionary search (E25) before the stream.
- **What this is not:** biological learning. E26–E29 investigated mushroom-body learning and stopped after
  E29.

Run:

```sh
POKEMON_ROM=/path/to/pokemon-red.gb python run_stream.py --mode broadcast   # OBS browser source http://127.0.0.1:8765/overlay
POKEMON_ROM=/path/to/pokemon-red.gb python run_stream.py --mode research    # dashboard          http://127.0.0.1:8765/dashboard
```

A supervisor restarts the simulation if it crashes or freezes, and records every intervention. It never
presses buttons. Stream statistics persist in `runtime/stream/stream-state.sqlite`. Details:
`results/e30-stream-readiness/analysis.md`.
