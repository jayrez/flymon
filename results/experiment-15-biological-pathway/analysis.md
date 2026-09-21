# Experiment 15 — Biological Visual Pathway Reconstruction

**Verdict: FAIL. Exact bottleneck: the T5 → visual-projection → descending-neuron
transfer (preregistered loss stage D/E).** A connectivity-selected biological
pathway does not preserve Pokémon screen information into descending neurons. The
best-case anatomically selected biological DN subset reached **22.0 %** held-out
accuracy (chance 20 %, permutation p = 0.108), versus **87.5 %** for the *same*
readout neurons driven by the engineered Experiment-3 pathway. No closed-loop
test was run: the DN-level gate was not met.

This is a clean negative result that localises the bottleneck precisely and rules
out several candidate causes.

## What was done

The Experiment-6 frozen dataset (5 screen families, 32 instances, 20 CNS seeds,
seeds 201–220) was driven through the **unchanged Experiment-5 biological retina**
(inferred 2-D R1–R6 geometry → eye-drive), and 200-step spike counts were recorded
at every stage. Descending-neuron and visual-projection subsets were selected
**from connectivity only** (`run_pathway_audit.py`, label-free restart diffusion of
the released weight matrix from the 3,335 mapped R1–R6 receptors), so no subset can
leak test labels. All accuracies are unseen-instance + unseen-seed nested
cross-validation using the Experiment-6 framework. The engineered pathway is the
Experiment-6 recorded DN counts on the identical fold plan.

## Stage-by-stage held-out screen accuracy (biological retina)

| Stage | Held-out acc | Between/within margin | First interpretation |
|---|---:|---:|---|
| R1–R6 | **97.0 %** | large | image enters intact |
| lamina (L1/L2/L3/L5) | **96.4 %** | large | preserved |
| T4 | 50.5 % | — | partial loss (motion channel) |
| T5 | **80.9 %** | 1.36 | still strongly informative |
| visual-projection (9,201) | 28.9 % | **0.02** | near-total collapse |
| all DN (1,314) | 20.6 % | **−0.35** | chance; centroids inseparable |

The collapse is sharp and localised: T5 retains 80.9 % with a positive class margin
(1.36), the broad visual-projection population falls to a margin of 0.02, and by the
descending neurons the between-class distance is actually smaller than the
within-class distance (margin −0.35). **The information is lost in the T5 →
visual-projection → DN transfer, not in the retina, lamina, or T5.** This reproduces
Experiment 5's stage-D diagnosis on held-out image instances.

## Did any biological subset rescue the DN stage? No.

| Biological DN readout (Exp-5 retina) | Held-out acc |
|---|---:|
| all DN (1,314) | 20.6 % |
| nested training-selected DN | 19.7 % |
| anatomically selected `bio_dn_top10 / 20 / 50` | 19.8 % / 17.2 % / **22.0 %** |
| P20 (frozen Experiment-4/6 informative set) | 18.9 % |
| DNa02 (L+R) | 21.7 % |
| DNg100 (L+R) | 18.6 % |
| DNg13 (L+R) | 19.2 % |
| DNp01 (L+R) | 19.2 % |

Every DN readout is at chance. The best-case anatomically selected subset
(`bio_dn_top50`, chosen favourably by test accuracy) reached 22.0 % with
permutation p = 0.108 (null mean 20.1 %, null 95th 22.5 %) and a bootstrap 95 %
interval of 20.2 %–23.9 % that includes chance. Training-based selection (nested,
19.7 %) did no better, so this is not a selection-method problem.

## Did connectivity-selected visual-projection cells beat the broad population? No.

| Visual-projection readout | Held-out acc |
|---|---:|
| broad visual-projection (9,201) | **28.9 %** |
| `bio_vp_top50 / 100 / 250` | 22.0 % / 25.6 % / 26.3 % |
| `t5_reader_vp_top50 / 100` (LPLC2/LPLC1/VS) | 25.6 % / 25.9 % |

Hypothesis (a) — that T5-reading lobula-plate cells (LPLC2/LPLC1/VS, which the
connectome audit correctly identified as the strongest direct T5 targets) retain
more than the broad population — is **not supported**: they retain slightly *less*
(≈26 % vs 28.9 %). The visual-projection stage has already lost the signal
regardless of how the subset is chosen.

## The decisive control: same neurons, two sensory paths

The 20 P20 descending neurons and the anatomically selected DN subsets were run
through the *engineered* Experiment-3 pathway on the identical fold plan:

| Readout | Biological retina | Engineered LC10a/LPLC2 injection |
|---|---:|---:|
| all DN | 20.6 % | **63.6 %** |
| nested selected DN | 19.7 % | **86.3 %** |
| P20 (frozen E4/E6) | 18.9 % | **87.5 %** |
| `bio_dn_top20` | 17.2 % | 38.3 % |

The readout neurons and classifier are identical; only the sensory injection
differs. The engineered pathway injects strong drive directly into 460 LC10a/LPLC2
projection cells, one synapse upstream of the informative DNs. The biological
pathway must instead traverse R1–R6 → lamina → medulla → T4/T5 → visual-projection,
and the screen-specific signal is diluted to nothing by the DN stage. **The
bottleneck is the biological sensory-transfer cascade, not the descending neurons
and not the readout.**

## Why the anatomy predicts this

The label-free audit quantifies the dilution. Every DN's visual-influence score —
the fraction of its recurrent input that traces to the mapped R1–R6 receptors — is
tiny: max 0.0126 (DNp11 R), median 0.0012, and no DN receives any direct
photoreceptor synapse (all are ≥ 3 hops downstream). The audit is nonetheless
anatomically valid: its top-ranked visual DNs are the known looming/escape cluster
(DNp11, DNp04, DNp03, DNp01, DNp05), and the strongest direct T5 targets are exactly
LPLC2 / LPLC1 / VS. But even these best-connected DNs draw ~1 % of their input from
vision, and that is not enough for screen-specific firing once the signal has passed
through the spiking optic lobe.

## Named neurons and P20

- **DNg13**: anatomical visual rank 486/1314, 4 hops from R1–R6, 15 visual-projection
  presynaptic partners, 0 direct T5 inputs; held-out accuracy 19.2 % (chance). DNg13
  is **not** a biological Pokémon-vision neuron; its role in the P20 A-channel does
  not rest on a strong dedicated visual pathway.
- **DNa02** (steering readout): visual rank 284–430/1314; 21.7 % (chance).
- **DNg100** (locomotion readout): visual rank 623–631/1314, the weakest of the named
  cells, consistent with a command neuron rather than a sensory readout; 18.6 %.
- **DNp01** (giant-fibre looming detector): the most visually connected named cell
  (rank 10 R, 56 L) and biologically a genuine looming detector, yet 19.2 % on
  Pokémon screens — its short visual path carries looming/motion salience, not the
  luminance-family identity these stimuli differ in.
- **P20** median visual rank 242/1314 (spread: DNp04 #5 … DNp13 #582). The
  statistically informative set is a *mix* of moderately and weakly visual DNs, not
  the anatomically most-visual ones — further evidence that the engineered A-channel
  signal is aggregate drive, not a dedicated biological visual pathway.

## Controls

Spatial-shuffle (fixed pixel permutation preserving each frame's luminance histogram)
is classified by R1–R6 with **100 % retention to a single class** (e.g.
shuffled-bedroom → bedroom 640/640, shuffled-menu → bedroom 640/640,
shuffled-title → dialogue 640/640). So even the strong R1–R6/lamina accuracy is
driven by luminance statistics, not spatial structure — the same caution
Experiment 6 raised. At the DN stage, no-vision, uniform-gray, and all shuffled
controls scatter like the (already broken) classifier, consistent with DN output
being at chance regardless of input.

## Pathway comparison

| Pathway | early / encoder | T5 | visual-projection | all DN | selected DN |
|---|---:|---:|---:|---:|---:|
| Experiment 3 engineered (Exp-6 held-out) | 96.9 % (projection) | — | — | 63.6 % | 86.3 % (nested) / 87.5 % (P20) |
| Experiment 5 retina (5-screen) | 100 % (R1–R6) | 76.7 % | 26.7 % | 20.0 % | 16.7 % |
| Experiment 15 biological (Exp-6 held-out) | 97.0 % (R1–R6) | 80.9 % | 28.9 % | 20.6 % | 22.0 % (best anatomical) |

Experiment 15 confirms Experiment 5 on a harder, held-out-instance protocol and
adds the matched engineered comparison and the connectivity audit. The metrics are
comparable where labelled; the engineered "early" figure is projection-population
accuracy, not a retina stage.

## Answers to the required questions

1. Experiment-5 R1–R6 mapping reproduced: **yes** (3,335 receptors; R1–R6 97.0 %,
   lamina 96.4 % on held-out instances).
2. Strongest information loss: **T5 → visual-projection → DN** (81 % → 29 % → chance).
3. T5 > T4 again: **yes** (80.9 % vs 50.5 %).
4. Most informative T5 targets: LPLC2/LPLC1/VS by connectivity, but none beats the
   broad visual-projection population for screen classification.
5. Visual-projection cells retaining information: none beyond the broad population;
   connectivity subsets were slightly worse (≈26 % vs 28.9 %).
6. Central-brain intermediate populations preserving it: none identified; the
   collapse is already complete at the visual-projection stage.
7. DNs with strongest anatomical visual paths: the looming/escape cluster
   (DNp11, DNp04, DNp03, DNp01); still at chance on these stimuli.
8. P20 cells emerging independently as strong biological candidates: **no**
   (median rank 242/1314; a mix, not a dedicated pathway).
9. DNg13: rank 486/1314, weak visual connectivity, 19.2 % (chance) — not a
   Pokémon-vision neuron.
10. DNa02 21.7 %, DNg100 18.6 % — both at chance; DNg100 is anatomically the least
    visual of the readouts.
11. Biologically connected DN subsets outperform Experiment 5? **No** (22.0 % vs
    20.0 %, not significant).
12. Generalise to held-out instances? **No** (this is the held-out result).
13. Generalise to held-out seeds? **No** (seeds are held out; still chance).
14. Spatial-shuffle retains the effect: the early-stage effect is **luminance-based**
    (survives shuffle); the DN stage has no effect to retain.
15. Distance to engineered performance: engineered 63.6 %/87.5 %, biological
    20.6 %/22.0 % — a ~65-point gap into DNs.
16. Justify a future closed-loop biological-sensory experiment? **No** — the DN
    signal does not exist through the biological retina.

## Limitations

Inferred connectome geometry on an arbitrary display, not measured fly vision.
FlyBrain's spiking lamina may not preserve the graded biological signals that carry
identity. Pokémon frames are not natural fly stimuli, and the early-stage accuracy
is largely luminance-histogram driven (shuffle controls). The visual-influence
metric is a structural diffusion, not a functional transfer measure; a functional
subset could in principle exist that anatomy alone does not rank. These narrow
capture families are not broad Pokémon-state coverage.

## Conclusion

Through the real MaleCNS retina, Pokémon screen identity is present and strong up to
T5, then collapses across the T5 → visual-projection → descending-neuron transfer;
no connectivity-selected biological subset recovers it. The engineered Experiment-3
pathway works only because it injects directly into projection neurons one synapse
above the informative DNs, bypassing the lossy cascade. The bottleneck is
biological sensory transfer, and this is where any future work must focus. No
gameplay control, reward, reinforcement learning, action decoder, or Pokémon RAM
input was used; Experiments 1–14 were not modified.

## Reproduction

```sh
redfly-benchmark/.venv/bin/python run_pathway_audit.py
OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python run_biological_pathway_experiment.py
OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python analyze_biological_pathway_experiment.py --permutations 1000
```
