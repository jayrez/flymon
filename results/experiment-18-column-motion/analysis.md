# Experiment 18 — Column-Resolved Motion Pathway

**Verdict: FAIL. Mechanistic category: D** (geometry present, but only an explicit
correlator-type computation succeeds). **Geometry gate: PASS.**

Two findings, and they point in opposite directions:

1. **MaleCNS does contain the spatial geometry for direction selectivity.** Recovered
   from connectivity alone, T4/T5 fast-vs-slow input offsets form the canonical
   four-direction layout: antiparallel pairs (179.3°, 178.2°) on orthogonal axes
   (95.7°, 98.4°), across ~1,700 cells per subtype at 99 %+ eligibility, with T5
   orientation very tight (R ≈ 0.88).
2. **No biologically simple model in the preregistered family reads it out.** The
   selected model reaches mean |DSI| 0.0446 but is *incoherent with its own anatomical
   prediction* (two subtypes prefer the opposite direction). The explicit multiplicative
   correlator — a control architecture, not a biological claim — reaches mean |DSI|
   **0.354** and agrees with the anatomical prediction in **85 %** of responsive
   neurons, for **all eight subtypes**.

This substantially revises Experiment 17. E17 concluded Category E ("no simple repair")
after applying delays at the **whole-cell-type** level, which is spatially blind. The
geometry was there all along; E17's intervention could not have used it. But E17's
deeper point survives in sharper form: the missing ingredient is a *correlation-like
nonlinearity*, not a parameter setting.

## Part A — flyvis positive control: F3 (not operational)

flyvis 1.2.0 installs and imports in an isolated directory (torch 2.14.0+cpu), but
ships **no console script**, so the documented `flyvis download-pretrained` does not
exist here; its only downloaders fetch the Sintel *training* dataset, and
`Ensemble("flow/0000")` fails on a results directory nothing populates. Direction
selectivity in flyvis is a property of *trained* networks, so an untrained network was
deliberately **not** substituted. Category **E** (flyvis succeeds where MaleCNS fails)
could not be evaluated. Details in `flyvis-notes.md`. The Experiment-17 HR control
(|DSI| = 1.0 on the same retina samples) and the in-experiment `M_HR` architecture
serve as positive controls instead. Nothing from flyvis is committed.

## Part B — geometry (see `geometry.md`)

Gate **PASS**: 99.6 % mean eligibility, minimum concentration R = 0.530, antiparallel
pairs, orthogonal axes, balanced left/right. Caveat: only ~42 % of each T4/T5 cell's
synaptic weight is column-resolvable, because Tm3 and CT1 carry no hex annotation.

## Part C — model hierarchy (calibration, calibration stimuli only)

| model | mean \|DSI\| | min sign consistency | dynamic − frozen |
|---|---:|---:|---:|
| M0 (instantaneous linear) | **0.0000** | 0.07 | +0.017 |
| M1 (+ cell-class temporal filters) | 0.0214 – 0.0441 | 0.23 – 0.30 | +0.015 |
| M2 (+ input rectification) | 0.0182 – 0.0353 | 0.11 – 0.14 | +0.014 |
| M3 (+ output squaring) | 0.0238 – 0.0413 | 0.27 – 0.30 | +0.002 |
| **M_HR** (explicit correlator, *control*) | **0.3879** | 0.46 | 0.000 |

**M0 gives exactly zero**, which is the expected analytic result: a purely
instantaneous linear sum over a static spatial arrangement cannot be direction
selective, however well organised that arrangement is. No biological model met the
preregistered selection threshold (mean |DSI| ≥ 0.05 **and** min sign ≥ 0.6 **and**
dynamic > frozen), so `selected-config.json` records
`met_preregistered_threshold: false` and the selection is explicitly a best-of-a-null-set:
**M1, tau_fast 2, tau_slow 8**. All models share a final output rectification (a firing
rate cannot be negative), so M1 is a linear-nonlinear model and is described as such.

## Held-out results (evaluated once, frozen model)

Statistical unit is the individual neuron; bootstrap over neurons, Holm-corrected
across the eight subtypes. "signResp" = fraction of *responsive* neurons whose
preferred-minus-null response matches the **anatomically predicted** direction.

### Selected biological model (M1, tau 2/8)

| subtype | DSI | 95 % CI | effect | resp. frac | signResp | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|
| T4a | **−0.0318** | [−0.039, −0.025] | −0.00050 | 0.63 | **0.36** | 1.6e−299 |
| T4b | +0.0629 | [+0.056, +0.070] | +0.00193 | 0.64 | 0.76 | 7.1e−66 |
| T4c | +0.0357 | [+0.029, +0.042] | +0.00089 | 0.62 | 0.67 | 5.9e−28 |
| T4d | +0.0124 | [+0.005, +0.020] | +0.00124 | 0.62 | 0.58 | 1.4e−06 |
| T5a | **−0.0752** | [−0.083, −0.068] | −0.00176 | 0.63 | **0.19** | 1.6e−299 |
| T5b | +0.0868 | [+0.080, +0.094] | +0.00334 | 0.65 | 0.85 | 1.4e−117 |
| T5c | +0.0470 | [+0.040, +0.054] | +0.00092 | 0.62 | 0.65 | 4.7e−23 |
| T5d | −0.0049 | [−0.014, +0.004] | +0.00113 | 0.65 | 0.50 | 0.88 (n.s.) |

Five subtypes agree with anatomy, **two (T4a, T5a) significantly prefer the opposite
direction**, and one (T5d) is null. That is not a working detector: the tiny p-values
reflect ~1,700 neurons, not a meaningful effect, which is exactly why the raw effects
(≤ 0.003) and the sign coherence are reported alongside.

### Explicit correlator control (M_HR — *not* the biological result)

| subtype | DSI | 95 % CI | signResp | p (Holm) |
|---|---:|---:|---:|---:|
| T4a | +0.3376 | [+0.312, +0.362] | 0.81 | 2.9e−88 |
| T4b | +0.3311 | [+0.305, +0.357] | 0.82 | 1.0e−91 |
| T4c | +0.3212 | [+0.297, +0.346] | 0.81 | 1.3e−87 |
| T4d | +0.3244 | [+0.298, +0.349] | 0.81 | 5.6e−89 |
| T5a | +0.2932 | [+0.267, +0.320] | 0.81 | 4.7e−82 |
| T5b | +0.4560 | [+0.432, +0.479] | 0.94 | 1.2e−169 |
| T5c | +0.4135 | [+0.390, +0.438] | 0.91 | 9.3e−161 |
| T5d | +0.3531 | [+0.329, +0.377] | 0.88 | 4.2e−134 |

**All eight subtypes positive, 81–94 % anatomical agreement.** Running the same
MaleCNS geometry through a multiplicative correlator recovers coherent four-direction
tuning; running it through the linear/rectifying family does not.

## Generalisation and controls

Held-out mean |DSI| by axis: speed 0.0342, width 0.0429, contrast 0.0362,
position 0.0399 — stable across all four unseen axes, and uniformly small.

| control | mean rate | vs dynamic (0.01556) |
|---|---:|---|
| frozen | **0.00000** | complete collapse |
| gray | **0.00000** | complete collapse |
| shuffle | 0.00632 | −59 % |
| reverse | 0.01372 | ≈ dynamic (still motion, opposite direction) |

Frozen and uniform-gray give *exactly* zero, and temporal shuffling removes 59 % — the
model is genuinely motion-driven. It simply is not *direction*-selective.

| ablation | mean \|DSI\| | change |
|---|---:|---|
| selected model | 0.0446 | — |
| geometry shuffle | 0.0168 | **−62 %** |
| temporal flat (tau_slow = tau_fast) | 0.0216 | **−52 %** |
| explicit correlator | 0.3537 | +693 % |

Both ablations substantially suppress what little directionality exists, so the column
geometry and the fast/slow temporal split are both causally involved — the linear
readout just cannot exploit them.

**ON/OFF specificity: absent.** T4 favours OFF by 0.00005 and T5 favours ON by 0.0001 —
negligible and in the wrong direction. The model's polarity is applied per presynaptic
cell class, which evidently does not produce pathway-level ON/OFF preference.

## Required answers

1. **Does MaleCNS contain spatial T4/T5 receptive-field geometry consistent with
   direction computation?** **YES** — clearly, and in the canonical four-direction
   arrangement.
2. **Was the Experiment-17 failure caused mainly by overly coarse dynamics?**
   **Partly.** E17's whole-cell-type delays were spatially blind and could not have
   used this geometry, so its Category-E framing was too strong. But correcting the
   spatial resolution alone does not fix it.
3. **Are temporal filters alone sufficient?** **NO** (M1 incoherent; M0 exactly zero).
4. **Is rectification / a simple nonlinearity required?** **Not sufficient either** —
   M2 and M3 are no better than M1 and no more coherent.
5. **Does an explicit correlator outperform the reconstructed pathway?** **YES,
   decisively** — 0.354 vs 0.045 mean |DSI|, and 85 % vs 57 % anatomical agreement.
6. **Does pretrained flyvis succeed where MaleCNS fails?** **Unknown** — F3, could not
   be evaluated.
7. **Is biologically meaningful T4/T5 direction selectivity now validated?** **NO.**
8. **Should Experiment 19 proceed to VP → DN transfer?** **NO.**

## Interpretation

The connectome carries the *spatial* half of a motion detector in full, subtype-resolved
detail. What no model in the preregistered biological family supplies is the
*multiplicative* half. A delay-and-compare detector needs a correlation between a fast
input and a spatially offset delayed input; a weighted sum followed by a static output
nonlinearity — which is what M1–M3 are, and what FlyBrain's LIF neuron is — produces a
direction-*biased* response, not a direction-*selective* one. That is precisely what we
observe: a small, stimulus-driven, geometry-dependent, but sign-incoherent effect.

The honest formulation is therefore: **MaleCNS contains subtype-specific spatial input
geometry capable of supporting direction-selective T4/T5 responses, and that geometry
is demonstrably sufficient when read out by a correlator — but not when read out by the
linear/rectifying models tested here.** This is an opt-in biologically constrained
reconstruction, not a claim that FlyBrain computes motion.

## Limitations

Only ~42 % of T4/T5 synaptic weight is column-resolvable (Tm3 and CT1 lack hex
annotations), so the measured receptive fields are partial. The temporal grid was
deliberately small and the best M1 sat at its `tau_slow = 8` edge, so a longer-delay
regime was not explored — noted, but not pursued post hoc. The model is a graded
front end evaluated by time-averaged response; a direction-selective *transient* could
exist without a direction-selective mean. flyvis category E remains untested.
Stimulus speeds are matched to a ~0.5-column offset only approximately.

## Recommendation

Experiment 19 should **not** proceed to VP → DN transfer. The two decisive follow-ups
are (a) obtain or train a flyvis ensemble and settle category E, and (b) test whether a
genuinely multiplicative/divisive dendritic interaction — the one ingredient the
correlator has and the LIF lacks — restores coherent T4/T5 tuning on this same
connectome geometry. The geometry itself no longer needs to be questioned.

## Reproduction

```sh
python run_column_motion_experiment.py geometry
python run_column_motion_experiment.py calibrate
python run_column_motion_experiment.py heldout
python analyze_column_motion_experiment.py
```
