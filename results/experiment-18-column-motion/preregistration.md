# Experiment 18 — Column-Resolved Motion Pathway (preregistration)

Frozen before any held-out evaluation. Only the **anatomical** geometry audit (which
uses released connectome metadata and no stimulus, no response, no label) and
calibration-set model behaviour may be inspected while freezing this document.
**No gameplay, closed loop, reward, RL, behaviour cloning, controller change, DOWN,
Pokémon RAM / coordinates / map-room IDs / dialogue or battle flags / screen-class
labels / desired actions.** Experiments 1–17 are not modified; the Experiment-17
adapter keeps its reference-equivalence guarantee and any Experiment-18 model is opt-in.

## Question

**Does MaleCNS contain the spatially organised synaptic geometry needed for T4/T5
direction selectivity, and can a small biologically constrained, column-resolved
visual model recover that selectivity on held-out stimuli?**

Experiment 17 concluded FAIL/Category E after applying temporal interventions at the
**whole-cell-type** level (`delay_steps={"Mi9":1,...}`). That is spatially blind. This
experiment tests the missing possibility: the required geometry may already be in the
connectome, column by column.

Experiment 17's stronger claim ("linear summation can never compute motion") is
**not** treated as settled and is itself under test here.

## Part A — flyvis positive control (bounded)

One bounded attempt to obtain a *pretrained* flyvis network in an isolated
environment, outside the repository. Outcome categories F1 (works natively and on
Flymon stimuli), F2 (native only), F3 (not operational after bounded setup → retain
the Hassenstein-Reichardt control). flyvis availability is **not** the primary
endpoint and its failure is not Experiment 18's verdict.

## Part B — geometry audit (anatomy only)

Optic-column coordinates come from released `assignedOlHex1/2` annotations, converted
to the same axial cartesian frame `flymon/retina.py` already uses
(`x = h1 − 0.5·h2`, `y = √3/2·h2`). Coordinates are **directly annotated only**; no
coordinate is inferred from any response.

T4/T5 cells themselves carry no `assignedOlHex` annotation, so their own column is not
required: the measured quantity is the **translation-invariant offset vector between
the weighted centroids of two input arms**, which does not depend on where the
postsynaptic cell is placed.

Arm membership is fixed from literature + connectome sign, never from results:

| pathway | fast arm | slow / sign-inverting arm |
|---|---|---|
| T4 (ON) | Mi1 | Mi9, Mi4 |
| T5 (OFF) | Tm1, Tm2 | Tm9 |

Tm3, CT1, LPLC1/2 carry no hex annotation and are excluded from geometry (documented,
not silently dropped). Inclusion rule: a T4/T5 neuron is eligible if it has **≥3
column-resolved fast-arm partners and ≥3 column-resolved slow-arm partners**.

Per eligible neuron: weighted centroids of each arm, offset vector
`v = centroid(fast) − centroid(slow)`, its magnitude and angle. Per subtype: circular
mean angle, resultant concentration `R`, n eligible, resolved-weight fraction,
between-subtype angular separation, and left/right homolog consistency.

**Predicted preferred direction is derived from anatomy**: for a delay-and-compare
detector the preferred motion runs from the slow input toward the fast input, i.e.
along `v`. Eye-space `v` maps to screen direction using the existing retina mapping
(left eye `du=+dx`, right eye `du=−dx` because it is mirrored into the display; both
`dv=−dy`). This prediction is fixed before any response is measured.

### Geometry gate

* **PASS** — ≥60 % of T4/T5 cells eligible, subtype concentration `R ≥ 0.3`, and a/b
  and c/d forming approximately antiparallel pairs on approximately orthogonal axes.
* **PARTIAL** — spatial asymmetry present but subtype orientation weak or inconsistent.
* **FAIL** — no reproducible relevant spatial structure, or too little resolvable metadata.

Dynamics will not be engineered to compensate for a geometry failure.

## Part C — column-resolved model hierarchy

Opt-in, offline, graded; stock `flybrain.FlyBrain` is untouched. One model time step =
one rendered stimulus frame. Column luminance comes from the unchanged Experiment-5
retina sampling, averaged per optic column (neural superposition).

Cell-class signal chain, globally specified (no per-neuron or per-subtype fitting):

```
column luminance -> contrast (subtract temporal mean) -> polarity (ON:+ / OFF:-)
  -> low-pass with cell-class tau  -> [M2+: half-wave rectify] -> weighted sum by
     actual MaleCNS synaptic weights over column-resolved partners -> [M3: exponent]
```

| model | content |
|---|---|
| **M0** | anatomy only, instantaneous, linear (no temporal filter) |
| **M1** | M0 + cell-class temporal filters (fast/slow), still linear |
| **M2** | M1 + half-wave rectification of medulla outputs |
| **M3** | M2 + output exponent (squaring) |
| **M_HR** | explicit multiplicative correlator — a **separate control architecture**, never the default "repaired MaleCNS mechanism" |

Calibration grid (small, transparent): `tau_fast ∈ {1, 2}` frames,
`tau_slow ∈ {3, 5, 8}` frames, `output exponent ∈ {1, 2}` (exponent only for M3).
No optimiser, no random search, no per-subtype parameters.

## Stimuli

Primary controlled motion reuses `flymon/ethology.py`. Because Experiment-16's bars
sweep ~16 px/frame (≈9 optic columns per frame) while the measured arm offset is
~0.5 columns, speed is an explicit generalisation axis rather than a fixed constant.
Parametric bars use a fixed 60 frames with variable speed; luminance is controlled and
bar area is constant per frame.

**Calibration** (model selection only): bar width 18 px, contrast 1.0, canonical start,
speeds **{0.5, 2} px/frame**, all four directions, ON and OFF.

**Held-out** (evaluated once): speeds **{0.25, 1, 4} px/frame**, bar widths **{12, 26}**,
contrasts **{0.25, 0.5, 0.75}**, shifted start positions, and new grating phases —
all disjoint from calibration values.

The **exact unchanged Experiment-16 catalogue** is additionally reported as a separate
canonical-stimulus result, so no easier stimulus can be substituted silently.

Controls: `gray`, `dark`, `bright`, `frozen_first`, `frozen_mid`, `shuffle`, `reverse`.
Secondary: `flow_left/right/expand/contract`, `loom/recede`.

## Seeds

The column-resolved front end is **deterministic**, so seed-based inference is not
fabricated. The statistical unit is the **individual T4/T5 neuron** (each has its own
measured receptive-field geometry). Seeds 1301–1320 (calibration) and 1321–1340
(held-out) are reserved for stochastic components only (geometry-shuffle ablation
RNG); both ranges were verified unused by Experiments 1–17.

## Metrics

For a preferred/null pair, with responses reported in absolute units alongside:

```
DSI = (R_pref − R_null) / (R_pref + R_null + epsilon),  epsilon = 1e-9
```

Both the **raw preferred − null effect** and the DSI are reported, so a small-count
ratio cannot inflate the conclusion (the Experiment-16/17 failure mode).

Preferred direction is taken from **anatomy** (Part B), fixed before held-out
evaluation. Anatomy-predicted vs response-measured preferred direction agreement is
itself reported.

## Statistics

Bootstrap over eligible neurons (10 000 resamples) for confidence intervals; sign
consistency across neurons; subtype-level consistency. Geometry-shuffle permutation
(presynaptic column assignments shuffled within cell type) provides the null for
"geometry is causal". Time windows from one trial are never treated as independent.
Subtype-specific claims are Holm-corrected across the 8 subtypes.

## Model selection

On **calibration data only**, prefer the **simplest** model meeting a minimum
meaningful effect: mean absolute subtype DSI ≥ 0.05 **and** consistent subtype sign
**and** dynamic-vs-frozen separation. Simplicity order M0 < M1 < M2 < M3. If every
model is at noise on calibration, that is reported as such — "best of a null set" will
not be oversold. M_HR is never selected as the biological result.

## Required controls (on the frozen selected model)

preferred vs null · dynamic vs frozen · ordered vs shuffled · forward vs reversed ·
ON/OFF specificity (T4 should favour ON motion, T5 OFF motion — tested, not forced) ·
**geometry ablation** (shuffle presynaptic column assignments within cell type) ·
**temporal ablation** (collapse fast/slow to one tau) · **nonlinearity ablation**
(fall back to the preceding model).

## PASS / PARTIAL / FAIL

* **PASS** — reproducible subtype-specific geometry; frozen model gives substantial
  held-out preferred-vs-null effects with DSI clearly above E16/E17 noise; direction
  preference generalises to unseen speeds/contrasts/widths/positions; reversal flips
  the signal; frozen and shuffled controls collapse it; geometry and temporal
  ablations suppress it; T4 favours ON and T5 favours OFF; subtype directionality is
  coherent rather than one lucky class.
* **PARTIAL** — geometry clearly present but only some subtypes recovered, or weak
  generalisation, or incomplete ON/OFF specificity, or strong dependence on an added
  nonlinearity.
* **FAIL** — geometry absent/incoherent, no simple dynamics give held-out selectivity,
  only the explicit correlator works, results vanish on generalisation, or subtype
  preference is inconsistent.

A single subtype at p < 0.05 does not earn PASS.

## Mechanistic categories

A geometry absent · B geometry present, linear temporal sufficient · C geometry
present, rectification/nonlinearity required · D only explicit correlator succeeds ·
E pretrained flyvis succeeds where MaleCNS reconstruction fails · F inconclusive /
metadata limitation.

## Prohibited supervision

No per-neuron or per-subtype direction labels may enter model construction. Subtype
preferred direction must emerge from connectivity, fixed cell-class dynamics, spatial
offset and sign structure. Direction labels are used for **evaluation only**.

## Stop condition

Stop after T4/T5. VP → DN transfer is Experiment 19 and is not executed here.
