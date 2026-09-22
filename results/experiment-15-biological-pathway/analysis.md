# Experiment 15 — Biological Visual Pathway Reconstruction

**Verdict: FAIL (for five-way Pokémon screen-family identity).** Under the current
Experiment-5 R1–R6 retinal encoder and FlyBrain dynamics, screen-family identity
remains strong through T5 (80.9 % held-out) but collapses across the
visual-projection and descending-neuron stages. No connectivity-selected biological
DN or visual-projection subset recovers it. The properly nested anatomical-DN
endpoint is **22.0 %** (chance 20 %, balanced 20.5 %, permutation p = 0.096,
bootstrap 95 % CI 19.8 %–24.2 %), versus **87.5 %** for the same P20 readout neurons
driven by the engineered Experiment-3 pathway. No closed-loop test was run.

This is a clean, narrowly scoped negative result. It does **not** show that "vision
dies" biologically — only that this specific luminance-distinguished, five-way
scene-identity signal is not preserved into DNs. Whether the pathway preserves
motion / optic-flow / looming features (which T4/T5 and their targets are built to
encode) is a different question that Experiment 15 did not test and that motivates
Experiment 16.

## Correction note (post-original-commit)

The original Experiment-15 analysis (commit `3b8bc1c`) chose the "primary" anatomical
DN subset by `max(..., key=accuracy)` over the held-out results of `bio_dn_top10/20/50`
— i.e. it selected the population size using outer-test accuracy, which is not a valid
preregistered primary. This has been corrected: the primary endpoint now selects
K ∈ {10, 20, 50} by **inner-fold cross-validation only** (`nested_fixed_subsets` in
`flymon/generalization.py`), freezing K per outer fold before touching that fold's
held-out data, and the 1000-permutation null repeats the full nested selection under
permuted labels. The fixed top10/20/50 results remain below as secondary descriptive
numbers. **The numerical conclusion is unchanged** (nested primary 22.0 %, p = 0.096
vs the earlier test-selected 22.0 %, p = 0.108; both fail). Per-fold K choices
(top10×4, top20×3, top50×25) and inner-CV scores are persisted in
`classification.json`. No raw neural data was altered.

## What was done

The Experiment-6 frozen dataset (5 screen families, 32 instances, 20 CNS seeds
201–220) was driven through the **unchanged Experiment-5 biological retina**
(inferred 2-D R1–R6 geometry → eye-drive), and 200-step spike counts were recorded
at every stage. Descending-neuron and visual-projection subsets were selected **from
connectivity only** (`run_pathway_audit.py`, a label-free restart-diffusion score on
the released weight matrix from the 3,335 mapped R1–R6 receptors), so no subset can
leak test labels. All accuracies are unseen-instance + unseen-seed nested
cross-validation using the Experiment-6 framework. The engineered pathway is the
Experiment-6 recorded DN counts on the identical fold plan.

## Stage-by-stage held-out screen accuracy (biological retina)

| Stage | Held-out acc | Between/within margin | Interpretation |
|---|---:|---:|---|
| R1–R6 | **97.0 %** | large | image enters intact |
| lamina (L1/L2/L3/L5) | **96.4 %** | large | preserved |
| T4 | 50.5 % | — | partial loss (motion channel) |
| T5 | **80.9 %** | 1.36 | still strongly informative |
| visual-projection (9,201) | 28.9 % | 0.02 | near-total collapse of scene identity |
| all DN (1,314) | 20.6 % | −0.35 | chance; centroids inseparable |

The collapse of **screen-family identity** is sharp and localised: T5 retains 80.9 %
with a positive class margin (1.36), the broad visual-projection population falls to a
margin of 0.02, and by the descending neurons the between-class distance is smaller
than the within-class distance (margin −0.35). This reproduces Experiment 5's stage-D
diagnosis on held-out image instances. Note this metric is scene identity only —
T4/T5 and their targets may be deliberately transforming scene identity into motion
features, which this classifier would not reward.

## Did any biological subset rescue the DN scene-identity stage? No.

| Biological DN readout (Exp-5 retina) | Held-out acc |
|---|---:|
| all DN (1,314) | 20.6 % |
| nested training-selected DN (ANOVA ranking) | 19.7 % |
| **primary: nested anatomical K∈{10,20,50}** | **22.0 %** (p = 0.096) |
| secondary fixed `bio_dn_top10 / 20 / 50` | 19.8 % / 17.2 % / 22.0 % |
| P20 (frozen Experiment-4/6 informative set) | 18.9 % |
| DNa02 (L+R) | 21.7 % |
| DNg100 (L+R) | 18.6 % |
| DNg13 (L+R) | 19.2 % |
| DNp01 (L+R) | 19.2 % |

Every DN readout is at chance for scene identity. The primary nested anatomical
endpoint reached 22.0 % (balanced 20.5 %) with permutation p = 0.096 (null mean
20.1 %, null 95th 22.5 %) and a bootstrap 95 % interval of 19.8 %–24.2 % that includes
chance. Training-based ANOVA selection (nested, 19.7 %) did no better, so this is not
a selection-method problem.

## Did connectivity-selected visual-projection cells beat the broad population? No.

| Visual-projection readout | Held-out acc |
|---|---:|
| broad visual-projection (9,201) | **28.9 %** |
| `bio_vp_top50 / 100 / 250` | 22.0 % / 25.6 % / 26.3 % |
| `t5_reader_vp_top50 / 100` (LPLC2/LPLC1/VS) | 25.6 % / 25.9 % |

Hypothesis (a) — that T5-reading lobula-plate cells (LPLC2/LPLC1/VS, which the
connectome audit correctly identified as the strongest direct T5 targets) retain
more scene identity than the broad population — is **not supported**: they retain
slightly *less* (≈26 % vs 28.9 %). The visual-projection stage has already discarded
scene identity regardless of subset. (This says nothing about whether those cells
carry *motion* information — see Experiment 16.)

## The decisive control: same neurons, two sensory paths

The 20 P20 descending neurons and the anatomically selected DN subsets were run
through the *engineered* Experiment-3 pathway on the identical fold plan:

| Readout | Biological retina | Engineered LC10a/LPLC2 injection |
|---|---:|---:|
| all DN | 20.6 % | **63.6 %** |
| nested selected DN | 19.7 % | **86.3 %** |
| P20 (frozen E4/E6) | 18.9 % | **87.5 %** |
| `bio_dn_top20` | 17.2 % | 38.3 % |

The readout neurons and classifier are identical; only the sensory injection differs.
The engineered pathway injects strong drive directly into 460 LC10a/LPLC2 projection
cells, one synapse upstream of the informative DNs. The biological pathway must
traverse R1–R6 → lamina → medulla → T4/T5 → visual-projection, and the
scene-identity signal is not preserved to the DN stage. The scene-identity bottleneck
is the biological sensory-transfer cascade, not the descending neurons or the readout.

## Anatomical ranking (label-free structural heuristic)

The label-free audit ranks neurons by a **normalized structural restart-diffusion
visual-influence score** — `r = (1−α)s + α·(Â r)`, α = 0.85, where `Â` is `|W|` with
each postsynaptic input row normalised to sum to 1, `s` restarts unit mass on the
mapped R1–R6 receptors, and synaptic sign is ignored during ranking. **This is a
structural ranking heuristic, not a physiological measurement of input current, and
"score ≈ 0.01" should not be read as "1 % of a DN's input is visual."** With that
caveat, the score is interpretable: its top-ranked DNs are the known looming/escape
cluster (DNp11, DNp04, DNp03, DNp01, DNp05), and the strongest direct T5 targets are
LPLC2 / LPLC1 / VS — both matching fly biology. The scores are small and tightly
spread across DNs (max 0.0126, median 0.0012, all ≥ 3 directed hops from R1–R6), so
anatomy alone does not single out a scene-identity-preserving DN subset.

## Named neurons and P20

- **DNg13**: structural visual-influence rank 486/1314, 4 hops from R1–R6, 15
  visual-projection presynaptic partners, 0 direct T5 inputs. It **did not preserve
  five-way Pokémon screen-family identity above chance** under the current Exp-5
  encoder and FlyBrain dynamics (19.2 %). This is a scene-identity result only; DNg13
  is biologically a steering/visual-motor cell, and whether it carries optic-flow /
  directional information here is untested (Experiment 16).
- **DNa02** (steering readout): visual rank 284–430/1314; 21.7 % on scene identity
  (chance) — again, direction/flow untested here.
- **DNg100** (locomotion readout): visual rank 623–631/1314, the weakest of the named
  cells, consistent with a command neuron rather than a sensory readout; 18.6 %.
- **DNp01** (giant-fibre looming detector): the most visually connected named cell
  (rank 10 R, 56 L) and biologically a genuine looming detector; 19.2 % on Pokémon
  screens — its short visual path is expected to carry looming/motion salience, which
  these luminance-family screens do not probe.
- **P20** median visual rank 242/1314 (spread: DNp04 #5 … DNp13 #582). The
  statistically informative set is a *mix*, not the anatomically most-visual DNs —
  consistent with the engineered A-channel signal being aggregate drive, not a
  dedicated scene-identity pathway.

## Controls

Spatial-shuffle (fixed pixel permutation preserving each frame's luminance histogram)
is classified by R1–R6 with **100 % retention to a single class** (e.g.
shuffled-bedroom → bedroom 640/640). So even the strong R1–R6/lamina accuracy is
driven by luminance statistics, not spatial structure — the same caution Experiment 6
raised. At the DN stage, no-vision, uniform-gray, and all shuffled controls scatter
like the (already scene-identity-free) DN output.

## Pathway comparison

| Pathway | early / encoder | T5 | visual-projection | all DN | selected DN |
|---|---:|---:|---:|---:|---:|
| Experiment 3 engineered (Exp-6 held-out) | 96.9 % (projection) | — | — | 63.6 % | 86.3 % (nested) / 87.5 % (P20) |
| Experiment 5 retina (5-screen) | 100 % (R1–R6) | 76.7 % | 26.7 % | 20.0 % | 16.7 % |
| Experiment 15 biological (Exp-6 held-out) | 97.0 % (R1–R6) | 80.9 % | 28.9 % | 20.6 % | 22.0 % (nested anatomical) |

## Answers to the required questions

1. Experiment-5 R1–R6 mapping reproduced: **yes** (3,335 receptors; R1–R6 97.0 %,
   lamina 96.4 % on held-out instances).
2. Strongest scene-identity loss: **T5 → visual-projection → DN** (81 % → 29 % → chance).
3. T5 > T4 again: **yes** (80.9 % vs 50.5 %).
4. Most connectivity-supported T5 targets: LPLC2/LPLC1/VS; none beats the broad
   visual-projection population for scene classification (motion untested).
5. Visual-projection cells retaining scene identity: none beyond the broad
   population; connectivity subsets were slightly worse (≈26 % vs 28.9 %).
6. Central-brain intermediate populations preserving scene identity: none identified.
7. DNs with strongest structural visual paths: the looming/escape cluster
   (DNp11, DNp04, DNp03, DNp01); still at chance on scene identity.
8. P20 cells emerging independently as strong biological scene-identity candidates:
   **no** (median rank 242/1314; a mix).
9. DNg13: rank 486/1314, weak structural visual connectivity, 19.2 % — did not
   preserve scene-family identity here; steering/optic-flow role untested.
10. DNa02 21.7 %, DNg100 18.6 % on scene identity — both at chance.
11. Biologically connected DN subsets outperform Experiment 5 on scene identity?
    **No** (22.0 % vs 20.0 %, not significant).
12–13. Generalise to held-out instances/seeds? **No** (this is the held-out result).
14. Spatial-shuffle: the early-stage effect is **luminance-based** (survives shuffle);
    the DN stage has no scene-identity effect to retain.
15. Distance to engineered performance: engineered 63.6 %/87.5 %, biological
    20.6 %/22.0 % — a ~65-point scene-identity gap into DNs.
16. Justify a future closed-loop biological-sensory experiment? **Not yet** — no
    scene-identity DN signal exists through the biological retina, and the motion /
    optic-flow question must be answered first (Experiment 16).

## Limitations

Inferred connectome geometry on an arbitrary display, not measured fly vision.
FlyBrain's spiking lamina may not preserve the graded biological signals that carry
identity. Pokémon frames are not natural fly stimuli, and the early-stage accuracy is
largely luminance-histogram driven (shuffle controls). The visual-influence metric is
a structural diffusion heuristic, not a functional transfer measure. Crucially, the
scene-identity endpoint may be the *wrong question* for this circuit: T4/T5 and their
targets are motion detectors, and a motion/looming/optic-flow signal could survive to
DNs even though scene identity does not — this is exactly what Experiment 16 tests.

## Conclusion

Through the real MaleCNS retina, Pokémon **scene-family identity** is present and
strong up to T5, then is not preserved across the T5 → visual-projection →
descending-neuron transfer under the current encoder and dynamics; no
connectivity-selected biological subset recovers it. The engineered Experiment-3
pathway works only because it injects directly into projection neurons one synapse
above the informative DNs. This does not establish that biologically appropriate
motion information fails to transfer — only that arbitrary scene identity does not.
No gameplay control, reward, reinforcement learning, action decoder, or Pokémon RAM
input was used; Experiments 1–14 were not modified.

## Reproduction

```sh
redfly-benchmark/.venv/bin/python run_pathway_audit.py
OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python run_biological_pathway_experiment.py
OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python analyze_biological_pathway_experiment.py --permutations 1000
```
