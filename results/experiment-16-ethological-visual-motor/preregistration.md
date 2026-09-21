# Experiment 16 — Ethological Visual-Motor Transfer (preregistration)

Frozen before held-out evaluation. Only calibration seeds (1101–1120) may be
inspected before freezing; held-out seeds (1121–1140) are untouched until the
analysis is run. This is offline neural validation — **no gameplay, no closed loop,
no controller change, no reward/RL/behaviour-cloning, no game RAM / coordinates /
room-id / action labels.** Experiments 1–15 are not modified. `sources.md` fixes the
biological expectations and stimulus–population pairings.

## Question

Experiment 15 showed that five-way Pokémon *screen-family identity* is not preserved
past the optic lobe into descending neurons. It did **not** test whether the pathway
preserves the features the optic lobe is actually built for. Experiment 16 asks:
**does the current R1–R6 → optic-lobe → visual-projection → descending-neuron
simulation preserve biologically appropriate motion / optic-flow / looming signals?**

## Hypothesis

T4/T5 and downstream visual-projection neurons encode ON/OFF motion, direction,
optic flow and looming rather than arbitrary scene identity, so a biologically
appropriate visual-motor signal may survive T5 → VP → DN even though scene identity
does not.

## Stimuli (deterministic, luminance-controlled; `flymon/ethology.py`)

144×160 RGBA, 10 frames, frozen Experiment-5 retina encoder, gain frozen at 1×.
Static: dark, gray, bright, vertical edge, horizontal edge, dark disc. ON motion
(bright bar on dark bg) left/right/up/down. OFF motion (dark bar on bright bg)
left/right/up/down. Looming: expanding dark / bright disc. Receding: contracting
dark / bright disc (= time-reversed loom, identical luminance histogram). Optic
flow: drifting grating left / right / expand / contract. Temporal controls for
off_left, off_right, on_left, loom_dark, flow_expand: frozen-first, frozen-mid,
shuffle (fixed permutation), reverse. Within each polarity the moving-bar area is
constant per frame; every temporal control reuses its source's frames, so mean
luminance and histogram are matched to the source and (for direction pairs) to each
other. Mean luminance per stimulus is recorded in `stimulus-manifest.json`.

## Timing

10 frames × 20 brain steps = 200 steps; adaptation 0 steps (reset then stimulate),
matching Experiments 5 and 15. Frozen before results.

## Seeds

Calibration 1101–1120 (fix per-subtype preferred direction and any thresholds).
Held-out 1121–1140 (all primary statistics). The independent unit is the CNS seed /
complete stimulus trial — never an individual brain step. Comparisons that share a
seed across two conditions use matched-seed (paired) tests.

## Recorded populations (subtype- and side-resolved)

R1–R6, lamina, T4a/b/c/d, T5a/b/c/d (per side), LPLC2 (per side), LPLC1, LC4, VS,
DNp01/DNp03/DNp04/DNp11 (per side), DNg13/DNa02 (per side), DNg100, MDN. Population
membership is fixed by MaleCNS cell type/side (`population-manifest.json`) — never by
held-out effect size.

## Metrics (formulas frozen now)

Per (stimulus, seed): mean per-neuron 200-step spike count `R` for each population.
- **Motion/feature contrast** for conditions A,B sharing seeds: per seed
  `d_s = R_A(s) − R_B(s)`; report mean, median, sign-consistency (fraction of seeds
  with `d_s` matching the mean sign), 95 % bootstrap CI (10 000 resamples over seeds),
  and a **matched-seed sign-flip permutation p** (10 000 flips).
- **Direction-selectivity index** for a subtype with calibration-fixed preferred /
  null directions: `DSI = (R_pref − R_null) / (R_pref + R_null + 1e−6)`, evaluated on
  held-out seeds, tested as a matched-seed contrast (pref vs null).

## Gate A (first gate)

Using held-out seeds, does the current pathway produce reproducible biological motion
structure — ON/OFF separation (T4 for ON, T5 for OFF motion vs matched frozen) and/or
direction-selective T4/T5 subtype responses? **Gate A passes** if at least one of
(i) T4 ON-motion-vs-frozen or T5 OFF-motion-vs-frozen separation, or (ii) T4/T5
subtype direction selectivity is significant (matched-seed permutation p < 0.05) and
sign-consistent on held-out seeds, with a non-negligible effect. Not every subtype
must behave perfectly.

**If Gate A fails, the main cascade analysis STOPS.** The conclusion is then that the
current retina/optic-lobe dynamics fail before the VP transfer stage (Category A). No
DN-level biological feature claim is made, and no gain/dynamics tuning is done in this
experiment (that would be Experiment 17).

## Primary endpoint (one, fixed now)

**Held-out T5 OFF-motion direction selectivity, then its preservation downstream.**
- Stage 1 (Gate A core): the mean held-out matched-seed `(R_pref − R_null)` for the
  T5 horizontal subtypes (T5a, T5b, both sides; preferred direction fixed per subtype
  on calibration seeds) is significantly > 0 (permutation p < 0.05, sign-consistent).
- Stage 2: a preregistered T5-reading VP population (LPLC2 + VS, connectivity-defined
  from Experiment-15 audit + literature) preserves a significant preferred-vs-null
  separation.
- Stage 3: a preregistered descending population preserves the same sign-consistent
  feature (looming cluster DNp01/03/04/11 for loom vs recede; DNg13/DNa02 lateral
  asymmetry for horizontal flow).
Preferred directions and population memberships are fixed before held-out evaluation;
we do not pick the subtype/direction that "worked best" afterwards.

## Secondary / exploratory endpoints

T4 ON-motion direction selectivity; looming transfer through LPLC2 (loom vs recede,
loom vs static, ordered vs shuffle); optic-flow transfer to DNg13/DNa02; dynamic-vs-
frozen and ordered-vs-shuffle effects; side-specific responses; DNp01 loom response.
All marked exploratory; a DN response with silent upstream motion detectors is treated
as a possible luminance-change confound, not a validated motion computation.

## flyvis functional benchmark (optional)

If `flyvis` imports and a pretrained model loads without destabilising the
environment, run the same stimuli through it as an independent positive control that
the stimuli *can* evoke T4/T5 directional structure. Compare response relationships /
direction selectivity / ON-OFF structure only, not neuron-by-neuron identity. If
installation is impractical, document and skip.

## Natural Pokémon follow-up (secondary, gated)

Only if a synthetic feature is validated to the DN level: re-run a natural sequence
(normal / frozen / shuffle / reverse) and ask whether the validated feature appears.
Not used to redefine the synthetic feature. No closed loop.

## Verdict categories

- **A** ON/OFF or direction structure not reproducibly present at T4/T5 (failure
  upstream / optic-lobe dynamics).
- **B** T4/T5 show motion structure but the matching VP populations do not preserve it.
- **C** feature survives to VP but not to DNs.
- **D** feature survives T4/T5 → VP → identified DNs on held-out seeds.
- **E** D plus reproducible transfer during natural Pokémon motion under proper controls.

Thresholds are not changed after observing held-out results.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python run_ethological_experiment.py
OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python analyze_ethological_experiment.py
```
