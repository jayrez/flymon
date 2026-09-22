# Experiment 15 — Biological Visual Pathway Reconstruction (preregistration)

Frozen before any Experiment-15 classification result is observed. Only the
label-free connectivity audit (`run_pathway_audit.py`, no stimulus / no neural
activity / no class label) has been inspected at the time of writing. Experiments
1–14 are not modified.

## Question

Experiment 5 established that Pokémon screen identity enters the biological
MaleCNS retina intact (R1–R6 100 %, lamina 100 %) but collapses downstream
(T4 ≈ 43 %, T5 ≈ 77 %, visual-projection ≈ 27 %, all-DN 20 % = chance).
Experiment 3 showed that an *engineered* LC10a/LPLC2 projection preserves strong
DN information. **Does a biologically connected, connectivity-selected pathway
from the Experiment-5 R1–R6 representation preserve stimulus-specific
information into specific visual-projection and descending neurons better than
the broad Experiment-5 downstream populations?**

## Hypothesis

A subset of biologically connected optic-lobe → visual-projection →
descending-neuron pathways preserves screen-family information even though the
broad DN population is at chance. Concretely: (a) visual-projection cells that
directly read T5 (lobula-plate columnar cells) retain more information than the
9,201-cell visual-projection superclass, and (b) descending neurons with the
strongest anatomical visual input retain more information than the 1,314-cell DN
population and than Experiment 5's broad DN readout.

## Candidate pathway selection procedure (label-free, already executed)

`flymon/pathway.py` computes, from the released signed MaleCNS weight matrix
`W[post, pre]` only:

1. **Visual-influence score** — restart diffusion `r = (1−α)s + α·(Â r)`, α=0.85,
   where `Â` is the input-normalised magnitude graph and `s` restarts on the 3,335
   Experiment-5 mapped R1–R6 receptors. `r[i]` = anatomical fraction of neuron
   *i*'s recurrent input tracing to the photoreceptors.
2. **Directed hop distance** from the mapped R1–R6 set and from T5.
3. **Direct presynaptic contributor counts** (visual, T5, visual-projection).

Selection rules, fixed now:

* `bio_dn_topK` = the K descending neurons with highest visual-influence score,
  K ∈ {10, 20, 50}; the classifier's inner cross-validation picks K.
* `bio_vp_topK` = the K visual-projection cells with highest visual-influence
  score, K ∈ {50, 100, 250}.
* `t5_reader_vp` = visual-projection cells ranked by direct T5 presynaptic inputs.

Because these selections use anatomy only, they never see test labels or test
neural activity, so the frozen subset introduces no selection leakage.

Audit outcome already recorded in `connectivity-audit.json`: the top visual DNs
are the looming/escape cluster (DNp11, DNp04, DNp03, DNp01, DNp05…); the T5-reading
visual-projection cells are LPLC2 / LPLC1 / VS; every DN's visual-influence is
small in absolute terms (max 0.0126, median 0.0012); DNg13 ranks 486/1314.

## Stimuli

The frozen Experiment-6 visual-instance dataset (`datasets/experiment-06/`): five
screen families (bedroom, dialogue, menu, title, intro), 32 primary instances
(menu-06 excluded on inspection, as in Experiment 6), 10 real RGBA framebuffers
each. Chance = 20 %; largest-class baseline 25 %.

## Neural path and encoder

The unchanged Experiment-5 biological retina: inferred 2-D R1–R6 geometry
(`results/experiment-05-retinotopic/retina-mapping.json`), linear-light Rec.709
luminance, bilinear sampling at inferred (u,v), and the Flymon temporal transform
`clip(0.45·luminance + 1.6·|Δ|, 0, 1)` into `eye_drive`. FlyBrain MaleCNS v1.0,
166,700 neurons, 25,582,938 connections, CUDA. 10 frames × 20 steps = 200 steps
per trial. No engineered LC10a/LPLC2 injection is used on the biological path.

## Train/test split

Instance- and seed-disjoint nested cross-validation from `flymon/generalization.py`
(the Experiment-6 framework, unchanged). Seeds 201–220 (20 CNS seeds), four seed
blocks. Every outer fold holds out whole image instances at all seeds and whole
seed blocks at all instances; inner folds re-rank features only on inner-training
data. Leakage audits (`audit_plan`) assert zero image/instance/seed overlap.

## Recorded populations

Per trial, 200-step summed spike counts for: R1–R6, lamina (L1/L2/L3/L5), T4, T5,
the visual-projection superclass, and all 1,314 DNs. Frozen anatomically selected
subsets (`bio_dn_topK`, `bio_vp_topK`, `t5_reader_vp`) index into these.

## Controls

* **no-vision** (`eye_drive=None`) at every seed.
* **uniform gray** full-field frames.
* **spatial-shuffle**: one fixed pixel permutation (RNG 606) applied to the first
  accepted instance of each class, preserving each frame's luminance histogram.

Spatial-shuffle preserving accuracy is treated as evidence that the effect rides
on luminance statistics, not spatial structure (per Experiment 6).

## Metrics

Per stage and per subset: held-out 5-way nearest-centroid accuracy and balanced
accuracy (instance+seed disjoint); between/within centroid distance and
generalization margin; per-class recall; DN population-size selection stability.
Engineered-pathway comparison reuses Experiment-6 recorded DN counts on the
identical fold plan. 1000-permutation instance-level null for the primary endpoint.

## Primary endpoint

Held-out (unseen instance + unseen seed) 5-way class accuracy of the
**anatomically selected biological DN subset** (`bio_dn_topK`, K by inner CV),
driven through the Experiment-5 biological retina, evaluated with the frozen
label-free selection.

## PASS / WEAK / FAIL (fixed before results)

* **PASS** — the biological anatomically-selected DN subset reaches ≥ 45 %
  held-out accuracy, permutation p < 0.05, exceeds the biological all-DN readout
  by ≥ 10 points, exceeds the Experiment-5 broad-DN result, and the spatial-shuffle
  control does not account for the whole effect.
* **WEAK** — a real signal survives into some biologically connected downstream
  population (T5-reading visual-projection subset or a DN subset above chance,
  permutation p < 0.05) but generalization is marginal, effect sizes are small,
  spatial-shuffle undermines interpretation, or the signal does not reach DNs
  robustly.
* **FAIL** — the reconstructed biological pathway does not preserve useful DN
  signal beyond Experiment-5 baselines, or above-chance results disappear under
  held-out evaluation or controls.

Thresholds are not changed after observing results.

## Optional closed-loop gate

No gameplay is run unless (1) the DN-level biological signal passes held-out
validation, (2) it is anatomically defensible, and (3) it improves on Experiment 5.
If those conditions are not met, the experiment stops after offline analysis — a
valid Experiment-15 result. Motor decoding (LEFT/RIGHT/UP/A) is not retuned.

## Reproduction

```sh
redfly-benchmark/.venv/bin/python run_pathway_audit.py
OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python run_biological_pathway_experiment.py
OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python analyze_biological_pathway_experiment.py
```
