# Experiment 2 — framebuffer feature detectors → MaleCNS

**Result: WEAK.** Across five screens, between-screen DN distance was only **167.8** versus **163.5** within-screen distance (ratio **1.026**). Four screens produced *exactly the same injected voltages* and exactly the same per-seed DN vectors. Title alone selected the opposite side. The ratio above 1 does not demonstrate useful five-screen discrimination. The pathway strengthened visual-versus-baseline DN response, but its coarse, capped input lost most screen identity.

## Upstream implementation review

- [`flybrain.eyes.FeatureDetectors`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/eyes.py) already maps `(dx, size)` object geometry to LPLC2 onset (`loom`) and LC10a tracking (`chase`) using `brain.cells(type, side)`, and returns `(indices, voltage)` pairs. It also has LC4 threat and LPLC1 projectile channels; our static frames contain no validated threat/projectile semantics, so those channels were not driven.
- [`FlyBrain.step`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/brain.py) accepts this list as `inject` independently of `eye_drive`. Every Experiment 2 step used `eye_drive=None`; the no-vision control used `inject=()`.
- [`reservoir.Trace`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/reservoir.py) provides decaying features, but exact returned spike indices were counted to preserve Experiment 1's per-DN endpoint. No readout or action decoder was fitted.
- Fixed assumptions and target mapping were recorded in [design.md](design.md) before running the experiment. This adapter is a coarse object proxy, not opponent recognition or true measured looming.

## Feature encoding and target neurons

Each PyBoy RGBA framebuffer was converted to Rec.709 grayscale, as in Experiment 1. Pixels with luminance below **0.5** formed an 8-connected mask. The largest component with at least **16 pixels** supplied centre `dx = mean_x − 79.5` and size `sqrt(component area)`. These fixed values were applied to all screens without identity-specific tuning. A zero-size object primed the upstream detector without brain stimulation; the real object then generated one LPLC2 appearance transient and sustained LC10a drive for the 200-step static presentation. This appearance is only a looming-like proxy.

| Type | Upstream channel | Left cells | Right cells | Selected total | Timing of injection |
| --- | --- | ---: | ---: | ---: | --- |
| LPLC2 | loom | 94 | 91 | 185 | first step only |
| LC10a | chase | 135 | 140 | 275 | every step |
| LC4 | threat | 71 | 55 | 0 used | no threat evidence |
| LPLC1 | shot | 68 | 66 | 0 used | no projectile evidence |

**460 unique LPLC2/LC10a neurons** were selected by metadata across both sides; each image stimulated only one side: **229 left-side cells** or **231 right-side cells** across its onset and steady phases. All selected cells had superclass `visual_projection`. The same **1,314** `descending_neuron` cells, **166,700**-neuron/ **25,582,938**-connection MaleCNS v1.0 brain, 12 seeds (`101..112`), 200 steps, CUDA device, reset procedure, and default model parameters from Experiment 1 were retained.

| Screen | Component pixels | Bounding box `(left,top,right,bottom)` | Side | LPLC2 onset | LC10a steady |
| --- | ---: | --- | :---: | ---: | ---: |
| Intro | 7,262 | `(0, 58, 160, 144)` | L | 0.800 | 0.800 |
| Title | 1,872 | `(17, 10, 144, 59)` | R | 0.800 | 0.800 |
| New Game menu | 840 | `(1, 1, 119, 47)` | L | 0.800 | 0.800 |
| Oak dialogue | 1,080 | `(1, 97, 159, 143)` | L | 0.800 | 0.800 |
| Bedroom | 10,966 | `(0, 0, 160, 144)` | L | 0.800 | 0.800 |

The upstream **0.8 voltage cap was reached in both used channels for every screen**. Intro, menu, dialogue, and bedroom all selected the left side; consequently, their injection sequences were identical. This is the main information loss, despite different source images and component geometries.

## Experimental conditions and debug artifacts

Experiment 2 used fresh PyBoy captures and required every framebuffer to match the corresponding Experiment 1 PNG **byte for byte** before running. Bedroom was loaded from canonical [`states/bedroom.state`](../../states/bedroom.state), SHA-256 `305212ea2a9e15b4dac7cc6ed5c1873ff7436f82e488bd8b0e0d59492b0dca95`. The sixth condition was no-vision baseline with no external injection. No Pokémon RAM was used as agent input, and the ROM stayed outside the repository.

| Screen | Framebuffer | Grayscale | Dark mask | Selected region | LPLC2 projection | LC10a projection |
| --- | --- | --- | --- | --- | --- | --- |
| Intro | [PNG](../../captures/experiment-02/intro-original.png) | [PNG](../../captures/experiment-02/intro-grayscale.png) | [PNG](../../captures/experiment-02/intro-dark-mask.png) | [PNG](../../captures/experiment-02/intro-selected-component.png) | [PNG](../../captures/experiment-02/intro-LPLC2-projection.png) | [PNG](../../captures/experiment-02/intro-LC10a-projection.png) |
| Title | [PNG](../../captures/experiment-02/title-original.png) | [PNG](../../captures/experiment-02/title-grayscale.png) | [PNG](../../captures/experiment-02/title-dark-mask.png) | [PNG](../../captures/experiment-02/title-selected-component.png) | [PNG](../../captures/experiment-02/title-LPLC2-projection.png) | [PNG](../../captures/experiment-02/title-LC10a-projection.png) |
| New Game menu | [PNG](../../captures/experiment-02/new_game_menu-original.png) | [PNG](../../captures/experiment-02/new_game_menu-grayscale.png) | [PNG](../../captures/experiment-02/new_game_menu-dark-mask.png) | [PNG](../../captures/experiment-02/new_game_menu-selected-component.png) | [PNG](../../captures/experiment-02/new_game_menu-LPLC2-projection.png) | [PNG](../../captures/experiment-02/new_game_menu-LC10a-projection.png) |
| Oak dialogue | [PNG](../../captures/experiment-02/oak_dialogue-original.png) | [PNG](../../captures/experiment-02/oak_dialogue-grayscale.png) | [PNG](../../captures/experiment-02/oak_dialogue-dark-mask.png) | [PNG](../../captures/experiment-02/oak_dialogue-selected-component.png) | [PNG](../../captures/experiment-02/oak_dialogue-LPLC2-projection.png) | [PNG](../../captures/experiment-02/oak_dialogue-LC10a-projection.png) |
| Bedroom | [PNG](../../captures/experiment-02/bedroom-original.png) | [PNG](../../captures/experiment-02/bedroom-grayscale.png) | [PNG](../../captures/experiment-02/bedroom-dark-mask.png) | [PNG](../../captures/experiment-02/bedroom-selected-component.png) | [PNG](../../captures/experiment-02/bedroom-LPLC2-projection.png) | [PNG](../../captures/experiment-02/bedroom-LC10a-projection.png) |

The projection maps show the driven half-screen and selected object box. LC4/LPLC1 maps were not produced because those channels were never injected.

## Neural response

Each value is mean ± sample SD across 12 trials, counting spikes over 200 steps. Target-class counts include both sides, including spontaneous spikes in the unstimulated side. Every trial passed the finite-state check.

| Condition | CNS spikes | Feature-projection spikes | LPLC2 spikes | LC10a spikes | DN spikes |
| --- | ---: | ---: | ---: | ---: | ---: |
| No vision | 2,372,397 ± 30,048 | 1,118 ± 67 | 592 ± 25 | 526 ± 72 | 12,170 ± 571 |
| Intro | 2,410,218 ± 30,781 | 16,770 ± 80 | 612 ± 28 | 16,158 ± 66 | 12,585 ± 601 |
| Title | 2,413,514 ± 32,148 | 16,986 ± 139 | 615 ± 21 | 16,371 ± 137 | 12,718 ± 558 |
| New Game menu | 2,410,218 ± 30,781 | 16,770 ± 80 | 612 ± 28 | 16,158 ± 66 | 12,585 ± 601 |
| Oak dialogue | 2,410,218 ± 30,781 | 16,770 ± 80 | 612 ± 28 | 16,158 ± 66 | 12,585 ± 601 |
| Bedroom | 2,410,218 ± 30,781 | 16,770 ± 80 | 612 ± 28 | 16,158 ± 66 | 12,585 ± 601 |

Feature-target and DN activity clearly rose above spontaneous baseline. Intro, menu, dialogue, and bedroom have identical spike counts and per-DN vectors for matching seeds; their apparent across-seed differences are stochastic only.

## Within-screen variance and between-screen separation

Each response vector contains firing counts for the same 1,314 DNs. Cosine and Euclidean values below compare **mean vectors** across seeds; trial distances compare individual runs. The diagonal of the trial-distance matrix is median **within-screen** distance between different seeds, and off-diagonal cells are median **between-screen** distance across all trial pairs.

### Cosine similarity of mean DN vectors

| | Intro | Title | New Game menu | Oak dialogue | Bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 1.00000 | 0.99036 | 1.00000 | 1.00000 | 1.00000 |
| Title | 0.99036 | 1.00000 | 0.99036 | 0.99036 | 0.99036 |
| New Game menu | 1.00000 | 0.99036 | 1.00000 | 1.00000 | 1.00000 |
| Oak dialogue | 1.00000 | 0.99036 | 1.00000 | 1.00000 | 1.00000 |
| Bedroom | 1.00000 | 0.99036 | 1.00000 | 1.00000 | 1.00000 |

### Euclidean distance of mean DN vectors

| | Intro | Title | New Game menu | Oak dialogue | Bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 0.0 | 85.8 | 0.0 | 0.0 | 0.0 |
| Title | 85.8 | 0.0 | 85.8 | 85.8 | 85.8 |
| New Game menu | 0.0 | 85.8 | 0.0 | 0.0 | 0.0 |
| Oak dialogue | 0.0 | 85.8 | 0.0 | 0.0 | 0.0 |
| Bedroom | 0.0 | 85.8 | 0.0 | 0.0 | 0.0 |

### Trial-to-trial DN Euclidean distance

| | Intro | Title | New Game menu | Oak dialogue | Bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 163.9 | 179.0 | 160.3 | 160.3 | 160.3 |
| Title | 179.0 | 162.0 | 179.0 | 179.0 | 179.0 |
| New Game menu | 160.3 | 179.0 | 163.9 | 160.3 | 160.3 |
| Oak dialogue | 160.3 | 179.0 | 160.3 | 163.9 | 160.3 |
| Bedroom | 160.3 | 179.0 | 160.3 | 160.3 | 163.9 |

Aggregate within-screen distance: **163.5**. Aggregate between-screen distance: **167.8**. Between/within ratio: **1.026**. The modest positive aggregate separation comes from Title's right-side injection versus the four left-side screens. All six pairings among the four left-side screens have mean-vector distance **0** and matched-seed trial distance **0**.

### No-vision baseline

| Image vs no vision | Mean-vector cosine | Mean-vector Euclidean | Cross-trial median | Matched-seed median | Mean DN total difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 0.99440 | 65.3 | 171.4 | 87.9 | +414.4 |
| Title | 0.99407 | 68.2 | 169.9 | 88.5 | +548.1 |
| New Game menu | 0.99440 | 65.3 | 171.4 | 87.9 | +414.4 |
| Oak dialogue | 0.99440 | 65.3 | 171.4 | 87.9 | +414.4 |
| Bedroom | 0.99440 | 65.3 | 171.4 | 87.9 | +414.4 |

The no-vision DN vectors matched Experiment 1 **exactly for each seed**, confirming that the baseline model and reset/noise procedure were unchanged. Feature injection moves DN activity away from baseline more than the photoreceptor drive did, but it does not preserve most screen identities.

## Performance

Frame preprocessing and component extraction were measured once per static frame; the step/projection/aggregation values are medians across 12 trials, per neural step. Timings are approximate and include public API overhead.

| Screen | Preprocess/frame ms | Feature extraction/frame ms | Feature-to-neuron projection/step ms | MaleCNS step ms | DN/target aggregation ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 0.960 | 1.175 | 0.378 | 2.429 | 0.365 |
| Title | 0.624 | 0.852 | 0.380 | 2.438 | 0.366 |
| New Game menu | 0.537 | 0.686 | 0.370 | 2.391 | 0.362 |
| Oak dialogue | 0.559 | 0.701 | 0.370 | 2.367 | 0.355 |
| Bedroom | 0.492 | 0.825 | 0.378 | 2.417 | 0.350 |

Experiment 2 median visual step was **2.417 ms**, versus **2.369 ms** for Experiment 1 and the earlier approximately **1.93 ms** bare CUDA step. `FeatureDetectors.inject` added about 0.37–0.38 ms per step in Python. The aggregation measurement includes the two target classes and is therefore broader than Experiment 1's receptor/DN aggregation. These measurements were taken in separate runs, so small step-time differences should not be overinterpreted.

## Direct comparison with Experiment 1

Experiment 1 files were copied unchanged to [`experiment-01-photoreceptors`](../experiment-01-photoreceptors/) and remain available at their original paths. Both experiments used the exact same frames and seed protocol.

| Metric | Experiment 1: photoreceptors | Experiment 2: feature detectors |
| --- | ---: | ---: |
| Within-screen DN distance | 178.8 | 163.5 |
| Between-screen DN distance | 168.4 | 167.8 |
| Between/within ratio | 0.942 | 1.026 |
| Mean pairwise screen cosine | 0.99961 | 0.99615 |
| Mean image-vs-baseline cosine | 0.99964 | 0.99434 |
| Mean image-vs-baseline DN Euclidean | 16.6 | 65.9 |
| Mean total DN spikes per visual trial | 12,217.4 | 12,611.4 |
| Median visual brain step | 2.369 ms | 2.417 ms |
| Result | FAIL | WEAK |

The feature route increased DN response to visual input and raised the aggregate ratio from 0.942 to 1.026. **It did not materially improve five-screen discrimination:** four conditions collapsed into one identical injection. The ratio alone would hide this failure mode.

## Conclusion and recommended next experiment

**WEAK.** One screen (Title) differs through a left/right side switch; the other four are indistinguishable for fixed seeds. The limitation is the static-frame, largest-component-to-one-object adapter plus upstream amplitude clipping, not a lack of activity in the biological projection classes. This experiment does not justify RL or an action decoder.

A next sensory experiment should use **temporal sequences** and a **retinotopic or multi-object mapping** that preserves more than one left/right bit, while keeping projection amplitudes within the upstream unsaturated range using a preregistered normalization. It should repeat the same baseline and variance analysis. That experiment was not implemented here.

Raw trials, per-DN vectors, target-class counts, exact frame hashes, and timings: [`trials.json`](trials.json). Reproduce from the repository root with `POKEMON_ROM=/local/path/pokemon-red.gb redfly-benchmark/.venv/bin/python run_feature_experiment.py`. The ROM is external.
