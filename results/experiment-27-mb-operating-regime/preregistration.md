# Experiment 27 — Mushroom-body operating regime (preregistration)

Written after the stock reproduction and before any intervention was run. No learning,
conditioning, Pokémon objective or E25 code is involved.

## Base

* Branch `experiment-27-mb-operating-regime`, from `origin/experiment-26-plasticity-validation`
  `336f06b`. It contains the E26 audit `af29393` and the validation commit `336f06b`.
* None of PRs #9, #10, #11 is merged. `origin/main` = `b86ddb3`.

## Stock reproduction (E26 seeds 2701–2706; done before this file)

Stock FlyBrain through the unchanged E26 harness reproduces E26 exactly:

* KCg-d visual totals are bit-identical for every stimulus;
* zeroing all 418 plastic edges gives −0.34 % (identical);
* KC, MBON01/11, PAM01, PPL101, APL and DPM duty is 0.971–0.973 (identical).

**Noted difference:** controller-DN duty differs by 1 spike in 36,000 cell-steps from E26's
separate diagnostic run. The step wrapper is verified bit-identical to `FlyBrain.step` over
600 steps, so this is run-to-run GPU floating-point variation.

**Assay correction before any intervention:** the DAN-headroom baseline window now starts
after a 200-step warm-up. The earlier 50-step window included the post-reset transient
(≈ 17 steps to saturation) and produced a spurious headroom in stock.

## Refractory audit (FlyBrain 0.1.0)

* **Parameter:** `FlyBrain(refractory=seconds)`, converted to
  `refractory_steps = round(refractory/dt)`. It is global (all neurons) and 0 in stock.
* **Mechanism:** `v[(steps − last_spike) ≤ refractory_steps] = 0` is applied before the
  threshold check, so k steps caps the duty at 1/(k+1) (k = 1 → 25 Hz).
* **Reset:** `reset()` sets `last_spike = −10⁶`, so nothing leaks across resets.
* **Wrapper:** the E26/E27 step wrapper replicates this exactly.
* **Consequence for R1:** raw duty below 80 % is trivially met by any k ≥ 1, so **R1 uses duty
  normalised to the refractory ceiling**.

## Frozen

The following are unchanged:

* the connectome matrix;
* the 418-edge plastic set (hash `65dbc6264c04a2fa`);
* KCg-d, MBON01/11, PAM01, PPL101;
* the T4 visual injection (hash `87829806…`);
* the E26 visual stimuli.

**Stimulus label correction:** the E26 label `pokemon_house1f` is renamed
`pokemon_bedroom_after_walk`, because the frame stayed on map 38.

The E26 measurement protocols are reused:

* visual trials of 60 decisions;
* DAN stimulation in a 10-step window over the amplitude grid 0.1–0.8;
* maximum effect: all KCg-d driven at 0.3 V for 300 steps; baseline vs zeroed vs doubled
  (diagnostic).

## Seeds

* Calibration (selection): 2801–2806.
* Confirmation (once, after freezing): 2811–2820.

Both blocks were verified unused.

## Intervention ladder (strict order, stop at the first passing family)

**A. Refractory** — `refractory_ms` ∈ {0 (stock), 20, 40, 60, 80, 100}.

**B. APL output gain** — only if A fails.

* Scales the output of the existing APL synapses by g ∈ {1.5, 2, 3, 5}. The APL topology is
  unchanged.
* Runs at the base refractory: the smallest A value that passes R1 and R6, else 0.

**C. Shared KC input gain** — only if A and B fail.

* One gain g ∈ {0.75, 0.5, 0.35, 0.25} on all input current to all 4,064 KCs (one shared
  parameter).
* Same base refractory.
* If used, it is reported as an engineered operating-point correction.

**Non-selectable mechanistic diagnostic:** FlyBrain's documented `sensory_input=False`. Its
docstring says the stock olfactory receptor neurons "excite each other into a runaway loop
and sit near maximum rate at rest". It is measured to explain the saturation, but it is **not
eligible for selection**, because it is outside the preregistered ladder.

## Criteria (calibration seeds; stock is always reported alongside)

Target populations T = {KCg-d, MBON01, MBON11, PAM01, PPL101}.

| id | criterion |
|---|---|
| R1 | every T population has baseline duty / refractory ceiling < 0.80 (600 steps from reset) |
| R2 | median T-population rate > 1 Hz and no T population silent |
| R3 | ≥ 1 visual condition changes KCg-d total spikes by ≥ 5 % relative, versus both `no_visual` and `gray`, same sign in ≥ 5/6 seeds; d_z and fraction of cells modulated reported |
| R4 | at the E26 bridge amplitude rule re-applied at this operating point (smallest amplitude with ≥ 50 % of DANs spiking once in the window, both channels), stimulated-window duty − steady-state baseline duty ≥ **10 percentage points**, in ≥ 5/6 seeds, for **both** PAM01 and PPL101 |
| R5 | zeroing all 418 edges changes MBON01 + MBON11 spikes by ≤ −10 %, in ≥ 5/6 seeds (the unchanged E26 gate) |
| R6 | finite voltages; whole-brain fraction of neurons at ≥ 90 % of the ceiling < 1 %; fraction silent < 50 % |

## Ranking and stop rules

* **Ranking** (lexicographic): pass R1 and R2 → R3 → R4 → R5 → least invasive (family order
  A < B < C, then smallest deviation from stock).
* **Stop at the first family containing a point that passes R1–R6.** Freeze its least
  invasive passing point.
* **If no point passes:** freeze nothing. The best point is reported descriptively and the
  confirmation set is run on it, labelled non-passing.
* **Thresholds are fixed:** R5 is never lowered.

## Verdicts

| verdict | condition |
|---|---|
| **RESPONSIVE REGIME FOUND** | R1–R6 pass on calibration and hold on confirmation |
| **PARTIAL** | saturation is relieved and visual / DAN responsiveness improves, but R5 fails |
| **NO RESPONSIVE REGIME** | otherwise |

**Scope:** no conditioning is run in E27, in any case.
