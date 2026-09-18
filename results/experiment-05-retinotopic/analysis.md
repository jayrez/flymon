# Experiment 5 — inferred 2-D R1–R6 retinal geometry

**Verdict: FAIL. Primary information-loss stage: D.** The primary encoder uses the existing Flymon temporal blend on bilinearly sampled, linear-light Rec.709 luminance. Grouped leave-one-seed-out all-DN accuracy was **20.0%** and nested training-selected DN accuracy was **16.7%**, versus 20% chance.

## Retinal mapping

`brain.visual` contains 6,006 cells: 3,377 R1–R6, 1,300 R7, and 1,329 R8. Experiment 5 maps only R1–R6. From local MaleCNS edges, every receptor's contacts onto column-annotated L1/L2/L3 cells were accumulated and the maximum-weight optic column selected. 3,335/3,377 mapped (1,107 left, 2,228 right); 42 were unmapped. Median confidence was 1.000; 15 were below 0.8.

The axial embedding is `x=h1-0.5*h2`, `y=sqrt(3)/2*h2`. Each eye is normalized independently. Left maps to u=0..0.6; right is horizontally mirrored into u=0.4..1; v reverses hex y so larger y is screen-up. The 0.4..0.6 overlap is engineered and was not tuned against Pokémon images. R7, R8, and unmapped visual cells receive zero drive.

## Calibration and stage diagnosis

| Pair | R1–R6 | Lamina | T4 | T5 | Visual projection | DNs | First > chance |
|---|---:|---:|---:|---:|---:|---:|---|
| left/right | 100.0% | 100.0% | 95.8% | 100.0% | 66.7% | 45.8% | r1_6 |
| vertical/horizontal | 100.0% | 100.0% | 75.0% | 100.0% | 50.0% | 50.0% | r1_6 |
| motion left/right | 100.0% | 58.3% | 45.8% | 45.8% | 50.0% | 54.2% | r1_6 |


## Pokémon results

| Transform | R1–R6 | Lamina | T4 | T5 | Visual projection | All DNs | Nested selected DNs | Fixed Experiment 4 DN types |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 100.0% | 100.0% | 41.7% | 71.7% | 25.0% | 18.3% | 16.7% | 26.7% |
| temporal | 100.0% | 100.0% | 43.3% | 76.7% | 26.7% | 20.0% | 16.7% | 21.7% |


Primary all-DN distance: within 164.1, between 158.5, ratio 0.966. The fixed-DN result is exploratory and separate from the nested classifier. Every primary outer fold reranked DNs using training spikes and selected among 1,314/500/250/100/50 using inner grouped validation.

## Permutation tests

| Result | Observed | Null mean | Null 95th | p |
|---|---:|---:|---:|---:|
| All DNs | 20.0% | 20.0% | 25.0% | 0.6424 |
| Nested selected DNs | 16.7% | 20.1% | 28.3% | 0.8482 |

## Comparison and performance

Experiment 1: all-DN classification was not reported; distance ratio 0.942. Experiment 3: all-DN 40.0%, nested selected-DN 98.3%, distance ratio 0.957. Experiment 5 values are reported above. Mean luminance conversion was 1.913 ms, bilinear sampling 0.332 ms, and eye-drive construction 0.447 ms. Per-trial neural and aggregation timing is retained in `trials.json`; offline classification took 419.99 s before report writing.

## No-vision and uniform-gray controls

| Pokémon screen | Mean-vector distance to no vision | Mean-vector distance to uniform gray |
|---|---:|---:|
| intro | 13.9 | 13.3 |
| title | 17.5 | 14.7 |
| new game menu | 15.2 | 13.6 |
| oak dialogue | 14.2 | 15.0 |
| bedroom | 15.0 | 13.7 |

These are DN-space presence controls. Both controls use identical seeds; uniform gray traverses the inferred retina while no vision passes `eye_drive=None`.

## Spike and timing audit

No requested neural population had zero total spikes in any synthetic or Pokémon condition (`zero_spike_conditions_by_population` in `classification.json` records the empty lists). Coordinate inference took **10.054 s** before caching. Mean frame luminance conversion took **1.913 ms**, bilinear sampling **0.332 ms**, and full eye-drive construction **0.447 ms**. Median MaleCNS step time was **2.676 ms** and median stage/DN aggregation overhead **1.155 ms per step**. Offline classification, including the recorded analysis phase but excluding report postprocessing, took **419.99 s**.

## Limitations

This is inferred connectome geometry projected onto an arbitrary display, not measured fly vision. Display projection is not calibrated to fly viewing angle; eye pose is unknown; viewport overlap is engineered; photoreceptor physiology is simplified; FlyBrain's spiking lamina may not preserve graded biological signals; and Pokémon pixels are not natural fly stimuli.

## Answer

**No.** DoomFly-style inferred 2-D geometry preserved Pokémon identity through R1–R6 and lamina and slightly raised the all-DN distance ratio from Experiment 1’s 0.942 to 0.966, but it did not produce downstream discrimination: all-DN accuracy was chance (20.0%) and nested selected-DN accuracy was below chance (16.7%, permutation p=0.8482). It was materially worse than Experiment 3’s 40.0% all-DN and 98.3% nested selected-DN results. The primary loss is stage D: visual projection neurons retain weak identity information, while the DN population collapses it. No gameplay controls, reward, reinforcement learning, action decoder, or Pokémon RAM sensory input was added.
