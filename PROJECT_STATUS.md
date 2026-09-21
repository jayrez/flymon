# Flymon project status

Updated 21 September 2026. Flymon investigates whether Pokémon Red visual
information can pass through the real MaleCNS v1.0 FlyBrain simulation and produce
biologically derived behaviour. A frozen descending-neuron controller now exists
and has been run in closed loop (Experiments 7–14), but the project remains a
scientific probe, not reinforcement learning. No reward learning, imitation
learning, action-reward decoder, Pokémon RAM sensory input, map coordinates, or
scripted gameplay policy is used anywhere.

## Runtime and data

- `flybrain` 0.1.0 runs MaleCNS v1.0 on the Tesla P40 through CUDA/CuPy. The
  connectome contains **166,700 neurons** and **25,582,938 connections**. The
  environment is Python 3.12 with CuPy and PyBoy; the brain files (`brain.npz`,
  `weights.npz`) live under `redfly-benchmark/data/`.
- PyBoy runs headless, yields `(144, 160, 4)` RGBA frames, and supports scripted
  buttons. The [canonical bedroom state](states/bedroom.state) is the verified
  starting frame. The ROM is supplied externally via `POKEMON_ROM` and is
  excluded from Git.

## Current system

```
Pokémon Red framebuffer → visual encoder → visual projection / photoreceptors
   → MaleCNS v1.0 (166,700 neurons) → descending neurons → frozen readouts → Game Boy actions
```

A frozen controller maps DN activity to four scientifically defensible controls
(LEFT/RIGHT/UP/A; DOWN disabled): LEFT/RIGHT = DNa02 baseline-relative steering
(Exp 8), UP = DNg100 baseline-relative locomotion (Exp 9), A = frozen Experiment-4
P20 RMS-z rising-edge event decoder (Exp 13). These mappings are frozen; Experiment
15 did not change them.

## Experiment arc

Full reports and exact encoders live under `results/experiment-NN-*/`. Summary:

| Exp | Question / path | Verdict |
| --- | --- | --- |
| 1–2 | Panorama photoreceptor / static feature drive → DNs | FAIL / WEAK |
| 3 | 18×20 spatiotemporal → 460 LC10a/LPLC2 cells → DNs | WEAK (all-DN 40 %) |
| 4 | Is temporal DN readout the missing piece? | FAIL for temporal hypothesis; **discovered** a selected-DN subset at 98.3 % (p=0.001) |
| 5 | Inferred biological R1–R6 retina → optic lobe → DNs | FAIL at DN stage; loss localised downstream of the optic lobe (stage D) |
| 6 | Does the selected-DN signal generalise to unseen image instances/seeds? | PASS (nested selected-DN 86.3 %), but largely luminance-statistic driven |
| 7–10 | Closed-loop interface, steering, locomotion, exploration calibration | controller components validated |
| 11–13 | Neural interaction interface; DNp01 and P20 A-channel event decoding | A channel validated as aggregate visual drive |
| 14 | Can the frozen controller autonomously progress from `bedroom.state`? | PASS (Category D); visual drive affects behaviour, but dynamic feedback not shown to uniquely cause progression |
| 15 | Can a connectivity-selected biological pathway carry screen info into DNs? | **FAIL**; bottleneck localised to T5 → visual-projection → DN transfer |

## Latest result (Experiment 15)

Driving the Experiment-6 dataset through the biological Experiment-5 retina
reproduces the degradation pattern on held-out image instances: R1–R6 97.0 %,
lamina 96.4 %, T4 50.5 %, **T5 80.9 %**, visual-projection 28.9 %, all-DN 20.6 %
(chance 20 %). A label-free connectome audit ranks DNs by anatomical visual input;
its top cells are the known looming/escape cluster (DNp01/03/04/11) and the
strongest T5 targets are LPLC2/LPLC1/VS — but no connectivity-selected DN or
visual-projection subset exceeds chance on these screens (best 22.0 %, p=0.11). The
decisive control: the *same* 20 P20 neurons score **87.5 %** via the engineered
LC10a/LPLC2 injection but **18.9 %** via the biological retina. The bottleneck is
therefore biological sensory transfer through the optic lobe, not the descending
neurons or the readout. DNg13 ranks 486/1314 anatomically and is not a
Pokémon-vision neuron. No closed-loop test was run — the DN-level gate was not met.

## Reproducing and navigating

- Experiment code is at the repository root (`run_*_experiment.py` / `analyze_*`),
  with trial data and reports beside each `results/experiment-NN-*/analysis.md`.
- Experiment 15: `run_pathway_audit.py` (label-free connectivity),
  `run_biological_pathway_experiment.py` (biological retina sweep),
  `analyze_biological_pathway_experiment.py` (held-out analysis);
  connectivity tracing lives in `flymon/pathway.py`.
- Tests: `test_generalization.py`, `test_interface.py`, `test_progression.py`,
  `test_pathway.py` (CPU only).
- Set `POKEMON_ROM` to a locally owned ROM for capture steps; keep it outside the
  repository.
