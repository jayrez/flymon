# Pokémon framebuffer → MaleCNS visual response

**Conclusion: FAIL.** Raw photoreceptor stimulation strongly changes visual-neuron and total CNS activity, but the four Pokémon screens do **not** produce descending-neuron responses that separate beyond normal trial-to-trial noise. This is a negative result for this specific static-frame encoder and the upstream point-neuron model; it is not a claim that biological flies cannot distinguish the images.

## Method and visual encoding

- The ROM was read from `POKEMON_ROM` by headless PyBoy 2.7.0. Experiment code used no Pokémon RAM as agent input and used no reward signal or action decoder; observations came only from the framebuffer. The ROM was not copied into this repository. Simple timed intro frames and Start/A presses yielded Intro, Title, New Game menu, and Oak dialogue. No battle or bedroom screen was reached.
- Each PyBoy framebuffer was verified as `uint8 (144, 160, 4)`. Alpha was dropped; RGB was divided by 255 and converted to luminance with Rec.709 weights `(0.2126, 0.7152, 0.0722)`. Vertical mean of the 144 rows gave 160 horizontal samples. Brightness remained absolute in `[0,1]`; no per-image contrast normalization was applied. X=0 remained left.
- Linear interpolation mapped those 160 samples at azimuths `-1..+1` directly onto `brain.azimuth`, which is aligned with `brain.visual`. The encoder used `clip(0.45 × luminance + 1.6 × abs(luminance − previous), 0, 1)`, matching the blend in upstream `flybrain.eyes.Eyes`. Upstream `Eyes` renders synthetic `Blob` objects, so the small `PanoramaEyes` adapter accepts measured frame panoramas instead. FlyBrain then applied its normal `eye_gain=0.62` inside `brain.step(eye_drive=...)`.
- For a controlled onset, every trial began with a common uniform previous luminance of `0.9`. The onset drive was applied on step 1 and the unchanged steady drive on steps 2–200. This tests static screen identity plus one visual transient; it does not test frame-to-frame game motion. The default FlyBrain `sensory_input=True` was retained.
- Twelve seeds (`101..112`) were reused across all five conditions. Each trial reset the CUDA brain and ran 200 steps (`dt=20 ms`, 4 simulated seconds). A no-vision `eye_drive=None` condition used the same seeds. The returned spike indices provided exact total CNS, visual-neuron, and per-DN counts. Upstream `reservoir.Trace` supplies decaying features, but direct counts were used here to keep this first measurement inspectable; no readout was fitted.

## Receptors and descending neurons

**6,006** unique `brain.visual` receptors received drive; `brain.azimuth` also has length 6,006 and spans `-1..+1`. **1,314** descending neurons were selected with `brain.cells(["descending_neuron"])`; every selected neuron had `superclass == "descending_neuron"`. The loaded CUDA brain contained exactly **166,700 neurons and 25,582,938 connections**.

## Debug captures

Each screen has four inspectable PNGs. Receptor-drive images sort receptors by azimuth, with onset above steady drive. Original frames were visually checked.

| Screen | Original | Grayscale | Panorama | Receptor drive |
| --- | --- | --- | --- | --- |
| Intro | [PNG](../captures/vision/intro-original.png) | [PNG](../captures/vision/intro-grayscale.png) | [PNG](../captures/vision/intro-panorama.png) | [PNG](../captures/vision/intro-receptor-drive.png) |
| Title | [PNG](../captures/vision/title-original.png) | [PNG](../captures/vision/title-grayscale.png) | [PNG](../captures/vision/title-panorama.png) | [PNG](../captures/vision/title-receptor-drive.png) |
| New Game menu | [PNG](../captures/vision/new_game_menu-original.png) | [PNG](../captures/vision/new_game_menu-grayscale.png) | [PNG](../captures/vision/new_game_menu-panorama.png) | [PNG](../captures/vision/new_game_menu-receptor-drive.png) |
| Oak dialogue | [PNG](../captures/vision/oak_dialogue-original.png) | [PNG](../captures/vision/oak_dialogue-grayscale.png) | [PNG](../captures/vision/oak_dialogue-panorama.png) | [PNG](../captures/vision/oak_dialogue-receptor-drive.png) |

## Neural response

Values are mean ± sample SD across 12 trials, totals over 200 steps. Neural state was finite after every trial.

| Condition | CNS spikes | Visual-neuron spikes | DN spikes |
| --- | ---: | ---: | ---: |
| No vision | 2,372,397 ± 30,048 | 14,942 ± 387 | 12,170 ± 571 |
| Intro | 2,501,701 ± 30,223 | 166,815 ± 116 | 12,173 ± 682 |
| Title | 2,584,589 ± 30,335 | 254,625 ± 98 | 12,177 ± 665 |
| New Game menu | 2,618,358 ± 27,607 | 291,308 ± 130 | 12,270 ± 602 |
| Oak dialogue | 2,619,073 ± 31,367 | 291,464 ± 149 | 12,196 ± 692 |

The visual-neuron count rose from about 14,942 at baseline to 166,815–291,464 with images, proving that frame-derived drive reached the photoreceptors. DN totals stayed near 12,170 per trial, with 602–692 SD for the visual conditions. Relative to matched-seed baseline, mean DN total differences were +2.7 (Intro), +6.3 (Title), +100.2 (menu), and +25.7 (dialogue); the matched-seed SDs were 304, 215, 235, and 324 respectively.

## Similarity and stochastic variance

Each DN response is a 1,314-element firing-count vector. Cosine similarity and Euclidean distance below compare **mean DN vectors** across the 12 seeds. Euclidean units are spikes per DN summed across the 200 steps.

### Cosine similarity of mean DN vectors

| | Intro | Title | New Game menu | Oak dialogue |
| --- | ---: | ---: | ---: | ---: |
| Intro | 1.00000 | 0.99971 | 0.99957 | 0.99966 |
| Title | 0.99971 | 1.00000 | 0.99969 | 0.99970 |
| New Game menu | 0.99957 | 0.99969 | 1.00000 | 0.99952 |
| Oak dialogue | 0.99966 | 0.99970 | 0.99952 | 1.00000 |

### Euclidean distance of mean DN vectors

| | Intro | Title | New Game menu | Oak dialogue |
| --- | ---: | ---: | ---: | ---: |
| Intro | 0.0 | 14.7 | 18.6 | 15.7 |
| Title | 14.7 | 0.0 | 16.0 | 14.9 |
| New Game menu | 18.6 | 16.0 | 0.0 | 19.5 |
| Oak dialogue | 15.7 | 14.9 | 19.5 | 0.0 |

The next matrix compares **individual trials**. Diagonal cells are the median distances between two different seeds for the *same* screen; off-diagonal cells are median distances between all trial pairs for *different* screens.

### Trial-to-trial Euclidean distance

| | Intro | Title | New Game menu | Oak dialogue |
| --- | ---: | ---: | ---: | ---: |
| Intro | 190.5 | 175.3 | 167.7 | 178.9 |
| Title | 175.3 | 182.7 | 165.1 | 170.2 |
| New Game menu | 167.7 | 165.1 | 167.1 | 170.5 |
| Oak dialogue | 178.9 | 170.2 | 170.5 | 188.2 |

Across the four visual conditions, median within-screen distance averaged **182.1**, while cross-screen distance averaged **171.3** (ratio **0.94**). Different screens were no farther apart than repeated runs of the same screen. Mean-vector screen distances averaged **16.6**, only about 9% of within-screen trial distance. Mean-vector cosine similarity ranged from about **0.99952 to 0.99971**. Matching the same seed across different screens gave median distances around 48–53, confirming small image effects, but those effects were well below the normal between-seed spread.

### No-vision baseline

| Image vs no vision | Mean-vector cosine | Mean-vector Euclidean | Cross-trial median | Matched-seed median | Mean DN total difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intro | 0.99972 | 14.5 | 165.4 | 50.7 | +2.7 |
| Title | 0.99974 | 13.9 | 166.9 | 50.2 | +6.3 |
| New Game menu | 0.99958 | 18.4 | 158.5 | 48.8 | +100.2 |
| Oak dialogue | 0.99962 | 16.7 | 166.0 | 57.9 | +25.7 |

Baseline and image DN mean vectors were likewise almost identical (cosine >0.9995). The strong receptor response did not turn into a reliable condition-specific DN signature.

## Performance

Frame preprocessing and projection are one-time per captured static frame; the `step()` and DN aggregation timings are medians across 12 trials, per neural step. `step()` returns NumPy spike indices and therefore includes CUDA synchronization/transfer.

| Screen | Preprocess/frame ms | Projection/call ms | MaleCNS step ms | DN aggregation ms |
| --- | ---: | ---: | ---: | ---: |
| Intro | 0.984 | 0.592 | 2.165 | 0.167 |
| Title | 0.538 | 0.538 | 2.180 | 0.177 |
| New Game menu | 0.459 | 0.521 | 2.178 | 0.176 |
| Oak dialogue | 0.636 | 0.530 | 2.179 | 0.176 |

No-vision median MaleCNS step time was **1.917 ms**, close to the earlier ~1.93 ms P40 benchmark. Frame-derived visual input raised step time to about 2.17–2.18 ms. Preprocessing plus projection cost about 1 ms per newly encoded framebuffer; it is material beside one brain step, although the experiment reused static drives for 200 steps. DN aggregation cost about 0.17–0.18 ms per step.

## Conclusion and next experiment

**FAIL — the raw framebuffer → photoreceptor path does not preserve enough Pokémon screen information in descending-neuron activity for this model and protocol.** The receptor population responds, but image differences are smaller than normal stochastic DN variance. The upstream project also notes that raw eye signals fade through its simplified spiking lamina. The next sensory experiment should use upstream `FeatureDetectors` to route inspectable visual features into biological projection-neuron types **LPLC2, LC4, LPLC1, and LC10a**, then repeat the same baseline and seeded DN comparison. That pathway was not implemented here.

Raw per-trial counts, per-DN vectors, timing, and comparison values: [`visual-response-trials.json`](visual-response-trials.json). Reproduce with `POKEMON_ROM=/local/path/pokemon-red.gb redfly-benchmark/.venv/bin/python run_visual_experiment.py` from the repository root. The ROM remains outside the repository.

Upstream APIs inspected: [`brain.py`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/brain.py), [`eyes.py`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/eyes.py), [`reservoir.py`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/reservoir.py).
