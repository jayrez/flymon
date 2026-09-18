# Experiment 5 design — inferred 2-D R1–R6 retina

## Question and preregistered interpretation

Does preserving inferred 2-D MaleCNS R1–R6 geometry improve downstream Pokémon-screen discrimination over Experiment 1’s vertical-mean panorama and Experiment 3’s engineered LC10a/LPLC2 projection? PASS requires robust grouped-by-seed DN discrimination, material improvement over Experiment 1, and meaningful synthetic spatial calibration; WEAK indicates partial structure or small-subset dependence; FAIL indicates that useful screen information does not survive the current photoreceptor/lamina path.

## Retinal inference

The installed FlyBrain metadata is inspected by neuron ID and type. Only R1–R6 cells are eligible. For every receptor, all released outgoing MaleCNS contacts onto L1/L2/L3 cells with `assignedOlHex1/2` are accumulated by coordinate; the highest-weight coordinate is selected and confidence is selected weight divided by all eligible weight. No receptor IDs or coordinates are hardcoded. The axial embedding and viewports follow DoomFly’s MIT-licensed implementation: `x=h1-0.5*h2`, `y=sqrt(3)/2*h2`; each eye is normalized independently; left occupies u=0..0.6 and right is mirrored into u=0.4..1; v reverses y. The overlap is fixed before viewing classification results.

## Pixel and neural path

RGBA PyBoy frames are converted from sRGB to linear light and then Rec.709 luminance. Luminance is bilinearly sampled at inferred `(u,v)`. A full `len(brain.visual)` vector is constructed by visual-slot identity; mapped R1–R6 slots receive drive, while R7, R8, and unmapped slots receive 0. Two transforms are fixed: direct luminance, and Flymon defaults `clip(0.45*luminance + 1.6*absolute_change, 0, 1)` with a 0.9 initial background. No DoomFly voltage/current constants are used.

## Conditions and measurements

Synthetic 144×160 stimuli are black, white, left, right, vertical stripes, horizontal stripes, deterministic motion-left, and motion-right. Pokémon conditions are Intro, Title, New Game menu, Oak dialogue, and the canonical Bedroom state. Controls are `eye_drive=None` and uniform gray. Each trial runs 200 FlyBrain steps; dynamic stimuli have ten frames held for twenty steps each. Seeds 101–112 are shared across conditions but every classifier holds out all conditions for one seed.

Aggregate activity is recorded for R1–R6, L1/L2/L3/L5, T4, T5, the `visual_projection` superclass, all DNs, and the previously reported Experiment 4 DN types. Five-step bins are retained for every DN. The fixed DN types are an exploratory diagnostic only.

## Classification and statistics

Raw counts use Euclidean nearest-centroid classification. All-DN evaluation is grouped leave-one-seed-out. DN ranking uses the Experiment 4 training-only ANOVA-like between/within score. Population size is selected from all 1,314, top 500, 250, 100, and 50 by inner grouped validation for each outer fold. Test labels and spikes never enter ranking or size selection. The primary transform is the existing Flymon temporal blend, fixed before results. Label permutations shuffle within seed and repeat grouped folds; the selected-DN null repeats training-only ranking and size selection for 1,000 permutations.

No reward, reinforcement learning, action decoding, gameplay control, or Pokémon RAM sensory input is used.

## Reproduction

Install `pyarrow` alongside the existing project environment and place the official MaleCNS v1.0 annotation and connection Feather files under `redfly-benchmark/data/raw/`. Their SHA-256 values are recorded in `retina-mapping.json`. Keep the Pokémon ROM external and set `POKEMON_ROM`; then run:

```sh
redfly-benchmark/.venv/bin/python run_retinotopic_experiment.py
redfly-benchmark/.venv/bin/python analyze_retinotopic_experiment.py
```

The derived mapping is cached in `retina-mapping.json`; remove only that Experiment 5 cache to force independent coordinate derivation again.
