# Experiment 3 — 2-D spatial + temporal projection into MaleCNS

**Verdict: WEAK.** All five injected sequences are distinct and visual-projection neurons separate them strongly. DN trial distances do not separate beyond stochastic variation: between **169.3**, within **177.0**, ratio **0.957**. Held-out five-way DN classification is **40.0%** versus **20%** chance, but it equals Experiment 2's aggregate accuracy and has concentrated confusions. Temporal sequences did not outperform the static control.

## Experiment 2 collapse diagnosis

Experiment 2 reduced each full frame to the largest dark component `(dx, sqrt(area))`. Upstream [`FeatureDetectors`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/eyes.py) then selected only L/R side and capped both active channels at 0.8. Intro, New Game menu, Oak dialogue and bedroom all selected L; their 460-element onset hashes are `d3777aff4d9c80dd7b7fad2f75c8fae1e2e1fcdf0ecdb20de86d66ce5d53999c`, and steady hashes are `d97869e217b46f2ae5818022fc70de7073c92a8da3fc744eda94137d9865098d`. Pairwise distance among those four is exactly **0** in both phases. Their onset vector min/max/mean/SD/nonzero is **0/0.8/0.3983/0.4000/229**; steady is **0/0.8/0.2348/0.3643/135**. Title's opposite-side onset and steady hashes differ, with distances **17.16** and **13.27**. The inputs had distinct images and component geometries, so the collapse occurred in single-object spatial compression, L/R aggregation, and detector saturation, before the CNS. See the preregistered [design and diagnosis](design.md).

| Experiment 2 screen | Phase | SHA-256 prefix | Min/max | Mean/SD | Nonzero of 460 |
| --- | --- | --- | --- | --- | ---: |
| Intro | onset | `d3777aff4d9c80dd…` | 0.00/0.80 | 0.3983/0.4000 | 229 |
| Intro | steady | `d97869e217b46f2a…` | 0.00/0.80 | 0.2348/0.3643 | 135 |
| Title | onset | `ae9e13cb4f6c9c04…` | 0.00/0.80 | 0.4017/0.4000 | 231 |
| Title | steady | `12902db61c51b58e…` | 0.00/0.80 | 0.2435/0.3681 | 140 |
| New Game menu | onset | `d3777aff4d9c80dd…` | 0.00/0.80 | 0.3983/0.4000 | 229 |
| New Game menu | steady | `d97869e217b46f2a…` | 0.00/0.80 | 0.2348/0.3643 | 135 |
| Oak dialogue | onset | `d3777aff4d9c80dd…` | 0.00/0.80 | 0.3983/0.4000 | 229 |
| Oak dialogue | steady | `d97869e217b46f2a…` | 0.00/0.80 | 0.2348/0.3643 | 135 |
| Bedroom | onset | `d3777aff4d9c80dd…` | 0.00/0.80 | 0.3983/0.4000 | 229 |
| Bedroom | steady | `d97869e217b46f2a…` | 0.00/0.80 | 0.2348/0.3643 | 135 |

Pairwise exact distances for all onset and steady vectors are in [the collapse diagnostic](experiment-02-collapse.json). All six pairings among the four left-side screens are zero; each Title pairing is 17.16 onset and 13.27 steady. Full hashes are retained in that JSON.

## Spatial and temporal encoder

Each actual 144×160 RGBA PyBoy frame is converted to Rec.709 luminance, then mean-pooled in nonoverlapping 8×8 blocks to an **18×20 two-dimensional grid**. Each cell retains x and y. LC10a receives `0.55 × (1 − luminance)` as a dark-object/contrast proxy; LPLC2 receives `0.75 × abs(current − previous luminance)` as a change/looming-like proxy. First-frame change is zero. Voltage is rounded to 0.05 increments and capped at 0.8; observed gain limits prevent detector saturation. These are explicit engineering approximations, not a learned model or a claim that static game pixels are natural fly prey or true looming.

[`Eyes`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/eyes.py) supplies the biological luminance-plus-change idea, but accepts only a 1-D panorama. Upstream `FeatureDetectors` chooses biological classes but offers only one L/R object value. [`FlyBrain.step`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/brain.py) accepts the per-group voltage injections used here. `reservoir.Trace` was not used because exact firing-count vectors are the endpoint. MaleCNS metadata provides no projection-neuron receptive-field coordinates or visual elevation. Soma positions were not treated as receptive fields. We evenly assigned distinct grid samples within each half-screen to metadata-selected cells in stable neuron order. This preserves source x/y in a **surrogate mapping**, not biological retinotopy. The exact 460 indices and grid coordinates are in `trials.json`.

| Class/side | Neurons | Grid source | Feature |
| --- | ---: | --- | --- |
| LPLC2 L | 94 | L half of 18×20 | absolute change |
| LPLC2 R | 91 | R half of 18×20 | absolute change |
| LC10a L | 135 | L half of 18×20 | darkness |
| LC10a R | 140 | R half of 18×20 | darkness |

**460 unique visual-projection neurons** receive potential drive; **1,314 metadata-selected descending neurons** are measured. LC4/LPLC1 were excluded because no validated threat/projectile signal was extracted. The MaleCNS v1.0 dataset has 166,700 neurons and 25,582,938 connections.

Ten frames were captured four PyBoy ticks apart, then presented for 20 MaleCNS steps each: **200 steps per trial**, matching Experiments 1/2. The initial checkpoint matches the Experiment 1 framebuffer pixel-for-pixel. Bedroom begins from canonical `states/bedroom.state` (SHA-256 `305212ea2a9e15b4dac7cc6ed5c1873ff7436f82e488bd8b0e0d59492b0dca95`) and holds Right for eight game frames between captured frames 3 and 5. Other conditions use no buttons during capture. Capture happens once; its exact sequence is reused for all 12 seeds (`101..112`). The static control repeats each condition's initial grid for every window; no-vision injects nothing and uses `eye_drive=None`. No Pokémon RAM or ROM copy is used.

## Encoder separation and debug artifacts

The exact quantized injected 460-cell vector for every frame is stored in `trials.json`, with SHA-256, min/max/mean/SD and nonzero count. This was checked **before** CNS trials. There are **5 unique condition sequences**. Repeated trials replay identical captured sequences, so encoder within-condition distance is **0**; its between/within ratio is **undefined**, rather than infinite or an estimated stochastic separation. Mean between-condition encoder distance is **15.41** (Euclidean over 10×460 values).

### Pairwise encoder Euclidean distance

| | Intro | Title | New Game menu | Oak dialogue | Bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 0.0 | 18.3 | 19.6 | 19.0 | 17.2 |
| Title | 18.3 | 0.0 | 7.9 | 8.4 | 19.0 |
| New Game menu | 19.6 | 7.9 | 0.0 | 6.1 | 20.3 |
| Oak dialogue | 19.0 | 8.4 | 6.1 | 0.0 | 18.2 |
| Bedroom | 17.2 | 19.0 | 20.3 | 18.2 | 0.0 |

| Condition | Unique framebuffer frames | Unique injected windows | Dynamic–static encoder distance | Vector min/max | Mean/SD | Nonzero entries over 10 windows | Debug strip |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| Intro | 7 | 8 | 4.41 | 0.00/0.55 | 0.173/0.246 | 1698 | [contact sheet](../../captures/experiment-03/intro-contact-sheet.png) |
| Title | 1 | 1 | 0.00 | 0.00/0.45 | 0.056/0.110 | 1100 | [contact sheet](../../captures/experiment-03/title-contact-sheet.png) |
| New Game menu | 1 | 1 | 0.00 | 0.00/0.20 | 0.015/0.051 | 380 | [contact sheet](../../captures/experiment-03/new_game_menu-contact-sheet.png) |
| Oak dialogue | 5 | 4 | 0.48 | 0.00/0.35 | 0.028/0.069 | 718 | [contact sheet](../../captures/experiment-03/oak_dialogue-contact-sheet.png) |
| Bedroom | 5 | 6 | 6.38 | 0.00/0.55 | 0.185/0.237 | 2854 | [contact sheet](../../captures/experiment-03/bedroom-contact-sheet.png) |

Each condition's [capture directory](../../captures/experiment-03/) contains all ten original frames, downsampled grids, temporal-difference maps, darkness feature maps and projected-drive images. In projected-drive images, red is LC10a darkness and blue is LPLC2 change; the contact sheet stacks original, grid, change and projection by time. Title and New Game menu did not animate within the 36-elapsed-game-frame capture, so their dynamic and static inputs are identical.

## Neural response and information-loss stage

Means ± sample SD across 12 seeds, counting spikes across 200 steps. Every trial passed a finite-state and spike-accounting check. The no-vision DN vector matched Experiment 1 exactly for each seed. Per-window CNS/visual-projection/DN counts and all per-neuron DN/visual-projection vectors are in `trials.json`.

| Condition | CNS spikes | Visual-projection spikes | DN spikes |
| --- | ---: | ---: | ---: |
| No vision | 2,372,397 ± 30,048 | 1,118 ± 67 | 12,170 ± 571 |
| Intro temporal | 2,406,158 ± 30,241 | 16,404 ± 82 | 12,419 ± 690 |
| Title temporal | 2,383,861 ± 30,066 | 5,592 ± 44 | 12,306 ± 587 |
| New Game menu temporal | 2,372,983 ± 29,907 | 2,104 ± 33 | 12,132 ± 679 |
| Oak dialogue temporal | 2,382,101 ± 28,877 | 4,695 ± 52 | 12,307 ± 513 |
| Bedroom temporal | 2,409,936 ± 27,067 | 17,782 ± 64 | 12,623 ± 331 |
| Intro static | 2,406,361 ± 28,790 | 16,074 ± 46 | 12,480 ± 522 |
| Title static | 2,383,861 ± 30,066 | 5,592 ± 44 | 12,306 ± 587 |
| New Game menu static | 2,372,983 ± 29,907 | 2,104 ± 33 | 12,132 ± 679 |
| Oak dialogue static | 2,377,862 ± 31,915 | 4,640 ± 47 | 12,246 ± 778 |
| Bedroom static | 2,407,450 ± 28,591 | 16,967 ± 68 | 12,496 ± 367 |

Visual-projection firing-count vectors strongly distinguish all conditions: mean within-screen Euclidean **38.6**, mean between-screen **954.5**, ratio **24.76**, and held-out nearest-centroid accuracy **100.0%**. Thus the encoder distinction survives through the stimulated biological projection cells.

### Projection-neuron cross-trial Euclidean distance

| | Intro | Title | New Game menu | Oak dialogue | Bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 43.6 | 1119.8 | 1249.6 | 1050.6 | 933.2 |
| Title | 1119.8 | 35.2 | 474.8 | 528.5 | 1242.3 |
| New Game menu | 1249.6 | 474.8 | 30.7 | 521.4 | 1369.4 |
| Oak dialogue | 1050.6 | 528.5 | 521.4 | 36.6 | 1054.9 |
| Bedroom | 933.2 | 1242.3 | 1369.4 | 1054.9 | 46.6 |

DN vectors are less separated: mean within-screen **177.0**, between-screen **169.3**, ratio **0.957**. Pairwise mean-vector cosines remain close to one; total DN spike differences alone cannot establish identity separation.

### DN cross-trial Euclidean distance (diagonal = within-screen)

| | Intro | Title | New Game menu | Oak dialogue | Bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 189.2 | 165.3 | 176.7 | 179.1 | 167.6 |
| Title | 165.3 | 168.9 | 166.6 | 168.5 | 161.8 |
| New Game menu | 176.7 | 166.6 | 189.9 | 173.1 | 169.9 |
| Oak dialogue | 179.1 | 168.5 | 173.1 | 175.1 | 164.3 |
| Bedroom | 167.6 | 161.8 | 169.9 | 164.3 | 161.7 |

### DN mean-vector cosine

| | Intro | Title | New Game menu | Oak dialogue | Bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 1.00000 | 0.99861 | 0.99747 | 0.99698 | 0.99861 |
| Title | 0.99861 | 1.00000 | 0.99927 | 0.99930 | 0.99880 |
| New Game menu | 0.99747 | 0.99927 | 1.00000 | 0.99920 | 0.99776 |
| Oak dialogue | 0.99698 | 0.99930 | 0.99920 | 1.00000 | 0.99831 |
| Bedroom | 0.99861 | 0.99880 | 0.99776 | 0.99831 | 1.00000 |

| Pair | Mean-vector Euclidean | Mean DN total difference (first − second) |
| --- | ---: | ---: |
| Intro / Title | 32.2 | +112.3 |
| Intro / New Game menu | 43.4 | +286.7 |
| Intro / Oak dialogue | 47.6 | +112.0 |
| Intro / Bedroom | 34.7 | -204.2 |
| Title / New Game menu | 24.1 | +174.3 |
| Title / Oak dialogue | 22.9 | -0.3 |
| Title / Bedroom | 32.0 | -316.6 |
| New Game menu / Oak dialogue | 25.8 | -174.7 |
| New Game menu / Bedroom | 44.6 | -490.9 |
| Oak dialogue / Bedroom | 36.9 | -316.2 |

The stage classification is **D: some distinctions survive into descending neurons**, but weakly and mainly as a few condition clusters. Stage A is excluded by five distinct encoder outputs; B is excluded by 100% projection-vector held-out classification. Stage C as a complete DN collapse is excluded by significant held-out DN classification, although stochastic DN distance still exceeds aggregate between-screen distance. This is **not** a PASS.

## Static control and no-vision baseline

Static DN within/between distance is **176.5/169.0** (ratio **0.958**), with **41.7%** held-out accuracy. It is as good as or slightly better than temporal input, so this experiment does **not** show a temporal benefit. Dynamic–static encoder distance is 0 for Title and menu, 0.48 for dialogue, 4.41 for Intro and 6.38 for bedroom.

| Condition | DN mean-vector distance from no vision | Matched-seed DN distance from no vision | DN mean-vector dynamic–static distance |
| --- | ---: | ---: | ---: |
| Intro | 49.4 | 74.2 | 14.7 |
| Title | 29.7 | 57.6 | 0.0 |
| New Game menu | 17.4 | 53.1 | 0.0 |
| Oak dialogue | 26.6 | 55.4 | 20.9 |
| Bedroom | 45.9 | 73.4 | 14.6 |

## Held-out classification diagnostic

Leave-one-**seed**-out nearest centroid uses 11 seeds per condition for training and predicts the five held-out trials for the twelfth seed, repeated over all 12 seeds. No policy or action decoder is fitted. Five-way chance is **20%**. Observed DN accuracy is **40.0% (24/60)**; a 1,000-shuffle label permutation within each seed gives p = **0.0010**, null mean 20.1%, null 95th percentile 26.7%. This shows recoverable DN information, but modest accuracy and confusions prevent a PASS.

| Actual \ predicted | Intro | Title | New Game menu | Oak dialogue | Bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 3 | 0 | 0 | 0 | 9 |
| Title | 1 | 2 | 1 | 5 | 3 |
| New Game menu | 0 | 0 | 3 | 7 | 2 |
| Oak dialogue | 0 | 0 | 3 | 7 | 2 |
| Bedroom | 3 | 0 | 0 | 0 | 9 |

For the four screens merged by Experiment 2 (excluding Title), held-out DN accuracy rises from **25.0%** (chance 25%; Experiment 2) to **45.8%**. Full five-way accuracy remains **40.0%** in both experiments. This is partial recovery of screen information, not robust five-screen discrimination.

## Performance

Frame capture includes PyBoy ticks plus framebuffer copy. Spatial preprocessing includes grayscale and grid pooling; temporal feature calculation and projection encoding were measured separately per captured frame. Injection-group construction is amortized per 20-step window. `FlyBrain.step` includes grouped voltage application and CUDA synchronization; DN aggregation also counts the 460 projection cells. Medians are shown; runs were not simultaneous with previous experiments.

| Condition | Capture/frame ms | Spatial/frame ms | Change/frame ms | Encode/frame ms | Group/window ms | MaleCNS step ms | Aggregation/step ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Intro | 0.617 | 0.764 | 0.026 | 0.569 | 1.520 | 5.807 | 0.273 |
| Title | 0.390 | 0.859 | 0.021 | 0.300 | 1.315 | 5.312 | 0.263 |
| New Game menu | 0.996 | 0.559 | 0.020 | 0.160 | 0.558 | 2.998 | 0.243 |
| Oak dialogue | 1.257 | 0.568 | 0.012 | 0.260 | 1.057 | 4.524 | 0.261 |
| Bedroom | 0.523 | 0.572 | 0.014 | 0.113 | 1.477 | 5.457 | 0.271 |

The visual step ranged about **3.0–5.8 ms**, above the **2.37 ms** Experiment 1 and **2.42 ms** Experiment 2 visual steps and the earlier bare CUDA **~1.93 ms**. The 0.05-V grouping limits injection calls but still makes this surrogate much costlier than the prior one-value-per-side detector. No optimization was attempted.

## Comparison with Experiments 1 and 2

| Metric | Exp. 1 photoreceptors | Exp. 2 feature detector | Exp. 3 spatial + temporal |
| --- | ---: | ---: | ---: |
| Within-screen DN distance | 178.8 | 163.5 | 177.0 |
| Between-screen DN distance | 168.4 | 167.8 | 169.3 |
| Between/within ratio | 0.942 | 1.026 | 0.957 |
| Held-out five-way DN accuracy | 23.3% | 40.0% | 40.0% |
| Distinct direct feature injections | 5 receptor drives | 2 projection patterns | 5 projection sequences |
| Verdict | FAIL | WEAK | **WEAK** |

Experiment 3 materially repairs the **encoder/projection collapse** and improves four-screen DN recoverability, but does **not** materially improve aggregate five-screen DN accuracy or the primary between/within distance criterion. Its ratio (0.957) remains below one. Most measured improvement comes from spatial coding, as the static control matches temporal performance. The previous experiment files and results were not modified.

## Conclusion and recommended Experiment 4

**WEAK.** Case **D**, with limited downstream survival: strong distinct encoder and projection signals, a statistically detectable but low-accuracy DN identity signal, and no aggregate DN distance advantage over stochastic variation. Title/menu had no temporal changes in the chosen window. These observations do not justify RL or action decoding.

Experiment 4 should isolate **temporal contribution** with longer, deliberately motion-rich but fixed visual sequences and a matched static control. It should also test an independently justified retinotopic mapping if MaleCNS visual receptive-field metadata or connectome-derived optic-column correspondence becomes available; this surrogate neuron order must not be treated as anatomy. Keep held-out classification and both projection/DN variance gates.

Reproduce with `POKEMON_ROM=/local/path/pokemon-red.gb redfly-benchmark/.venv/bin/python run_spatiotemporal_experiment.py` followed by `redfly-benchmark/.venv/bin/python analyze_spatiotemporal_experiment.py`. The ROM stays external.
