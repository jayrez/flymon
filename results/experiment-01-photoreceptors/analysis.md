# Pokémon framebuffer → MaleCNS visual response

**Conclusion: FAIL.** The five Pokémon screens, including the canonical bedroom state, stimulate visual neurons but do not produce descending-neuron responses that separate beyond normal trial-to-trial noise. This conclusion applies to this static-frame encoder and the upstream point-neuron model.

## Visual encoding and assumptions

- Headless PyBoy 2.7.0 supplied `uint8 (144, 160, 4)` framebuffer images. The experiment code used only the framebuffer as agent input; it did not read Pokémon RAM, use rewards, or fit an action decoder. The ROM stayed outside this repository.
- Intro, Title, New Game menu, and Oak dialogue came from timed intro frames and simple Start/A presses. **Bedroom** came only from [`states/bedroom.state`](../states/bedroom.state), loaded with PyBoy `load_state()` and rendered for one tick. The rendered bedroom frame matched [`states/bedroom-test.png`](../states/bedroom-test.png) pixel for pixel. No battle screen was reached.
- The saved bedroom state SHA-256 is `305212ea2a9e15b4dac7cc6ed5c1873ff7436f82e488bd8b0e0d59492b0dca95`. All paths in project code are repository-relative or supplied through `POKEMON_ROM`; no container mount path is required by Python code.
- Alpha was discarded. RGB was divided by 255 and converted to Rec.709 luminance using `(0.2126, 0.7152, 0.0722)`. Vertical averaging produced a 160-sample horizontal panorama. Absolute brightness was retained in `[0,1]`, and left-to-right orientation was preserved.
- Linear interpolation from azimuth `-1..+1` projected the panorama onto the receptor-aligned `brain.azimuth` array. The drive was `clip(0.45 × luminance + 1.6 × abs(luminance − previous), 0, 1)`. This matches the blend in upstream `flybrain.eyes.Eyes`; its Blob renderer cannot directly accept measured pixels, so `PanoramaEyes` provides the framebuffer adapter. FlyBrain applied its normal `eye_gain=0.62` in `brain.step(eye_drive=...)`.
- Every trial began from a common previous luminance of `0.9`, received one onset-drive step, and then 199 steps of unchanged steady drive. This is a static-screen identity experiment with one onset transient, not a test of game motion. The default `sensory_input=True` setting was retained.
- The CUDA brain was reset for each of 12 seeds (`101..112`) in every condition, including no-vision `eye_drive=None`. Each trial ran 200 steps (`dt=20 ms`, 4 simulated seconds). Spike indices returned by `brain.step()` supplied exact CNS, receptor, and per-DN counts. Upstream `reservoir.Trace` provides decaying features; direct counts were chosen for this inspectable first experiment.

## Neural populations

**6,006** unique photoreceptors were selected from `brain.visual`. `brain.azimuth` has one entry per receptor and spans `-1..+1`. **1,314** descending neurons were selected by `brain.cells(["descending_neuron"])`; every selected cell had `superclass == "descending_neuron"`. The real MaleCNS v1.0 CUDA model contained **166,700 neurons and 25,582,938 connections**.

## Debug images

Each condition has an original screenshot, grayscale image, panorama strip, and receptor-drive visualization. Receptor drive is ordered by azimuth, with onset above steady drive.

| Screen | Original | Grayscale | Panorama | Receptor drive |
| --- | --- | --- | --- | --- |
| Intro | [PNG](../captures/vision/intro-original.png) | [PNG](../captures/vision/intro-grayscale.png) | [PNG](../captures/vision/intro-panorama.png) | [PNG](../captures/vision/intro-receptor-drive.png) |
| Title | [PNG](../captures/vision/title-original.png) | [PNG](../captures/vision/title-grayscale.png) | [PNG](../captures/vision/title-panorama.png) | [PNG](../captures/vision/title-receptor-drive.png) |
| New Game menu | [PNG](../captures/vision/new_game_menu-original.png) | [PNG](../captures/vision/new_game_menu-grayscale.png) | [PNG](../captures/vision/new_game_menu-panorama.png) | [PNG](../captures/vision/new_game_menu-receptor-drive.png) |
| Oak dialogue | [PNG](../captures/vision/oak_dialogue-original.png) | [PNG](../captures/vision/oak_dialogue-grayscale.png) | [PNG](../captures/vision/oak_dialogue-panorama.png) | [PNG](../captures/vision/oak_dialogue-receptor-drive.png) |
| Bedroom | [PNG](../captures/vision/bedroom-original.png) | [PNG](../captures/vision/bedroom-grayscale.png) | [PNG](../captures/vision/bedroom-panorama.png) | [PNG](../captures/vision/bedroom-receptor-drive.png) |

## Neural response

Values are mean ± sample SD across 12 trials, totals over 200 steps. Neural state passed the finite-value check after every trial.

| Condition | CNS spikes | Visual-neuron spikes | DN spikes |
| --- | ---: | ---: | ---: |
| No vision | 2,372,397 ± 30,048 | 14,942 ± 387 | 12,170 ± 571 |
| Intro | 2,501,701 ± 30,223 | 166,815 ± 116 | 12,173 ± 682 |
| Title | 2,584,589 ± 30,335 | 254,625 ± 98 | 12,177 ± 665 |
| New Game menu | 2,618,358 ± 27,607 | 291,308 ± 130 | 12,270 ± 602 |
| Oak dialogue | 2,619,073 ± 31,367 | 291,464 ± 149 | 12,196 ± 692 |
| Bedroom | 2,497,474 ± 28,666 | 158,564 ± 234 | 12,271 ± 595 |

Visual-neuron spikes rose sharply from the no-vision baseline, so the framebuffer drive reached the receptors. Descending-neuron totals remained near baseline and varied substantially between seeds.

## Similarity and stochastic variance

Each DN response is a 1,314-element firing-count vector. Cosine and Euclidean matrices compare **mean vectors** across the 12 trials. Euclidean units are spike counts across 200 steps.

### Cosine similarity of mean DN vectors

| | Intro | Title | New Game menu | Oak dialogue | Bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 1.00000 | 0.99971 | 0.99957 | 0.99966 | 0.99952 |
| Title | 0.99971 | 1.00000 | 0.99969 | 0.99970 | 0.99962 |
| New Game menu | 0.99957 | 0.99969 | 1.00000 | 0.99952 | 0.99971 |
| Oak dialogue | 0.99966 | 0.99970 | 0.99952 | 1.00000 | 0.99940 |
| Bedroom | 0.99952 | 0.99962 | 0.99971 | 0.99940 | 1.00000 |

### Euclidean distance of mean DN vectors

| | Intro | Title | New Game menu | Oak dialogue | Bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 0.0 | 14.7 | 18.6 | 15.7 | 20.3 |
| Title | 14.7 | 0.0 | 16.0 | 14.9 | 18.1 |
| New Game menu | 18.6 | 16.0 | 0.0 | 19.5 | 15.0 |
| Oak dialogue | 15.7 | 14.9 | 19.5 | 0.0 | 22.2 |
| Bedroom | 20.3 | 18.1 | 15.0 | 22.2 | 0.0 |

The trial-distance matrix compares individual runs. Diagonal cells are median distances between different seeds for the **same** screen; off-diagonal cells are medians across trials of **different** screens.

### Trial-to-trial Euclidean distance

| | Intro | Title | New Game menu | Oak dialogue | Bedroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 190.5 | 175.3 | 167.7 | 178.9 | 165.8 |
| Title | 175.3 | 182.7 | 165.1 | 170.2 | 165.4 |
| New Game menu | 167.7 | 165.1 | 167.1 | 170.5 | 159.4 |
| Oak dialogue | 178.9 | 170.2 | 170.5 | 188.2 | 165.7 |
| Bedroom | 165.8 | 165.4 | 159.4 | 165.7 | 165.3 |

Across all five screens, average within-screen median distance was **178.8**; cross-screen median distance was **168.4**, a ratio of **0.94**. Different screens were no farther apart than repeated runs of the same screen. Mean-vector screen distances averaged **17.5**; cosine similarity ranged **0.99940–0.99971**. Same-seed cross-condition effects were present but much smaller than the between-seed spread.

### No-vision baseline

| Image vs no vision | Mean-vector cosine | Mean-vector Euclidean | Cross-trial median | Matched-seed median | Mean DN total difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 0.99972 | 14.5 | 165.4 | 50.7 | +2.7 |
| Title | 0.99974 | 13.9 | 166.9 | 50.2 | +6.3 |
| New Game menu | 0.99958 | 18.4 | 158.5 | 48.8 | +100.2 |
| Oak dialogue | 0.99962 | 16.7 | 166.0 | 57.9 | +25.7 |
| Bedroom | 0.99955 | 19.7 | 160.1 | 54.7 | +100.9 |

All visual conditions remained close to the no-vision DN baseline despite large receptor changes. The bedroom DN total was about 101 spikes above baseline on average, versus a matched-seed difference SD of about 205 spikes.

## Performance

Frame preprocessing and projection were measured once per static capture; these values are approximate. Neural step and DN aggregation values are medians across 12 trials, per step. `brain.step()` returns NumPy spike indices and therefore includes CUDA synchronization and transfer.

| Screen | Preprocess/frame ms | Projection/call ms | MaleCNS step ms | DN aggregation ms |
| --- | ---: | ---: | ---: | ---: |
| Intro | 0.943 | 0.601 | 2.369 | 0.188 |
| Title | 0.494 | 0.520 | 2.392 | 0.197 |
| New Game menu | 0.517 | 0.522 | 2.340 | 0.192 |
| Oak dialogue | 0.534 | 0.536 | 2.544 | 0.202 |
| Bedroom | 0.468 | 0.515 | 2.333 | 0.184 |

No-vision median step time was **1.993 ms**. Visual steps were around 2.3–2.5 ms in this rerun. Encoding a newly captured framebuffer cost roughly 1 ms before the brain step. Static drive was reused for steps 2–200; a moving-game stream would need a separate end-to-end timing test.

## Conclusion and next sensory experiment

**FAIL — raw framebuffer-to-photoreceptor drive does not preserve enough screen identity in descending-neuron activity under this protocol.** This remains true after including the canonical bedroom state. Upstream notes that raw eye signals can fade through its simplified spiking lamina. The next experiment should use upstream `FeatureDetectors` to route inspectable visual features into **LPLC2, LC4, LPLC1, and LC10a**, then repeat this seeded baseline comparison. That second pathway has not been implemented.

Raw per-trial counts, per-DN vectors, performance samples, and comparisons: [`visual-response-trials.json`](visual-response-trials.json). Reproduce from the repository root with `POKEMON_ROM=/local/path/pokemon-red.gb redfly-benchmark/.venv/bin/python run_visual_experiment.py`. The ROM is never copied into the repository.

Upstream APIs inspected: [`brain.py`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/brain.py), [`eyes.py`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/eyes.py), [`reservoir.py`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/reservoir.py).
