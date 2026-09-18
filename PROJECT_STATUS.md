# Flymon project status

Updated 18 September 2026. Flymon currently measures how visual information from Pokémon Red's framebuffer propagates through the real MaleCNS v1.0 FlyBrain simulation. It does not control the game. No reinforcement learning, rewards, action decoder, or Pokémon RAM sensory input has been added.

## Runtime and data

- `flybrain` 0.1.0 runs MaleCNS v1.0 on the Tesla P40 through CUDA/CuPy. The connectome contains **166,700 neurons** and **25,582,938 connections**. The installed environment uses Python 3.12 and PyBoy 2.7.0.
- The [P40 benchmark](redfly-benchmark/results/SUMMARY.md) measured **1.931 ms per step** for one brain, or **517.9 brain steps/s**. Batch 4 was comparable in aggregate throughput; larger batches slowed down. GPU memory capacity was not the practical limit in this API. The five-minute single-brain run showed no sustained slowdown or thermal throttling.
- PyBoy runs headless, yields `(144, 160, 4)` RGBA frames, and supports scripted buttons. The [canonical bedroom state](states/bedroom.state) provides the verified bedroom starting frame. The Pokémon Red ROM is supplied externally through `POKEMON_ROM` and is excluded from Git.

## Visual experiments

All experiments use five conceptual screens (Intro, Title, New Game menu, Oak dialogue, and Bedroom), a no-vision baseline, 12 matched seeds, and 200 MaleCNS steps per trial. The descending-neuron (DN) population is selected from MaleCNS metadata, with **1,314 DNs**. Each experiment's report describes its own controls and exact encoder.

| Experiment | Visual path or readout | Verdict | DN within | DN between | Between/within | Key finding |
| --- | --- | --- | ---: | ---: | ---: | --- |
| [1](results/experiment-01-photoreceptors/analysis.md) | Frame → 1-D panorama → 6,006 photoreceptors | FAIL | 178.8 | 168.4 | 0.942 | Raw receptor drive changes activity, but DN vectors do not separate screens beyond trial variation. |
| [2](results/experiment-02-feature-detectors/analysis.md) | Static features → biological projection cells | WEAK | 163.5 | 167.8 | 1.026 | Four screens produced identical injections; Title accounted for most separation. |
| [3](results/experiment-03-spatiotemporal/analysis.md) | 18×20 spatial grid and frame change → 460 LC10a/LPLC2 cells | WEAK | 177.0 | 169.3 | 0.957 | Five encoder sequences and projection responses differ. Some identity reaches DNs, but all-DN totals remain noisy. |
| [4](results/experiment-04-temporal-dn/analysis.md) | Compare 200-step DN totals with temporal DN bins and DN subsets | FAIL for temporal-readout hypothesis | 177.0* | 169.3* | 0.957* | Temporal binning did not materially improve the all-DN classifier; a small DN subset carried strong screen information in aggregate counts. |

*Experiment 4 reuses Experiment 3's sensory conditions and aggregate DN measurements; these are the aggregate-control distances, not a new readout result.*

Experiment 3's **40.0%** five-screen nearest-centroid result (20% chance) already used leave-one-seed-out validation; Experiment 4 verified there was no seed leakage. Its all-DN aggregate classifier scored **40.0%**, while grouped, nested temporal-bin choice scored **41.7%**. Training-fold-selected DNs using **aggregate** counts scored **98.3% (59/60)** with a conservative permutation **p = 0.001**; selected temporal bins scored **90.0%**. The selected aggregate result remained **97.9%** on four non-Title screens. Static input performed as well as or better than the changing-frame condition on the strongest readouts. These are offline diagnostics on five fixed screen sequences, not evidence of game-playing ability or generalization to unseen frames.

The main current finding is that visual-state information reaches a small DN subpopulation, while population-wide stochastic activity masks it in simple distance and nearest-centroid analyses. Experiment 4 did **not** support the idea that 200-step aggregation was the main loss mechanism. The selected DN result needs independent validation on new seeds, new instances of each visual state, and more game states before it can guide any behavior work.

## Reproducing and navigating the work

- The [CUDA benchmark workspace](redfly-benchmark/README.md) contains its script and raw measurements.
- Experiment code is in `run_visual_experiment.py`, `run_feature_experiment.py`, `run_spatiotemporal_experiment.py`, and `run_temporal_dn_experiment.py`; analysis scripts and trial data sit beside their corresponding reports under `results/`. Experiment 4 can replay the saved Experiment 3 injection vectors without the ROM.
- For experiments that capture game frames, set `POKEMON_ROM` to a locally owned ROM path. Keep the ROM outside this repository. The optional `compose.gui.yml` also requires `POKEMON_ROM_DIR` to point to its host directory. `webtop-config/` contains local GUI runtime files and stays ignored.
- The `captures/vision/`, `captures/experiment-02/`, `captures/experiment-03/`, and `captures/experiment-04/` directories contain the visual encoding checks and analysis plots. The exact replayed Experiment 4 DN bins are in `results/experiment-04-temporal-dn/dn-bins-5.npz`.

The next useful sensory validation is to repeat the training-fold DN-subset analysis on independently captured screen instances and new stochastic seeds, then measure whether those DNs still discriminate state changes over time. No game controller has been built.
