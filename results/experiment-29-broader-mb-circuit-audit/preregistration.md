# Experiment 29 — Broader mushroom-body circuit audit (preregistration)

This file was frozen before any E29 dynamic (simulation) result existed. Its hash is recorded in
`preregistration-freeze.json`.

- **Base:** `origin/main` 50e3685 (the E28 merge).
- **Branch:** `experiment-29-broader-mb-circuit-audit`.
- **Scope:** an audit plus candidate-circuit selection. There is no learning, conditioning, learning-rate
  calibration, weight mutation, Pokémon behaviour optimisation or E25 work. No emulator RAM enters the
  neural model.
- **Not reopened:** the E28 parameter search. No new refractory × APL grid is run.

## Question

Is the original 418-edge KCg-d → MBON01/MBON11 circuit too narrow? Specifically, does frozen Pokémon
visual information reach KC classes or MB compartments that were excluded from the first audit, and does
any broader connectome-derived KC → MBON ← DAN circuit show both visual coding and meaningful existing MBON
leverage?

## Visual-probe regime (frozen)

- Refractory 60 ms (3 steps), APL gain ×8, KC gain 1, `sensory_input = True`.
- **E29 uses 60 ms + APL×8 solely to expose visually responsive mushroom-body populations. It is not
  claimed to be a viable learning operating point.**
- This regime is not optimised.
- The second regime appears only where named below: 80 ms + APL×3, the E28 leverage-friendly point. No
  third regime is used, and there is no interpolation.

## Seeds (`seed-freeze.json`)

| Block | Seeds |
|---|---|
| Visual screen | 3001–3006 |
| Dynamic leverage | 3011–3016 |
| Confirmation and sensitivity | 3021–3030 |

The blocks are disjoint from each other and from all E26–E28 seeds. None of these values appears as a seed
anywhere in the repository.

## Static audit (`audit_broader_mb_circuit.py`; connectome only)

- **KC population:** every FlyBrain cell whose `cell_type` matches `^KC`, with all types included. The
  per-type counts must sum to the total. Body IDs are persisted and the population is hashed.
- **Visual-projection set:** every neuron with superclass `visual_projection`. This is the same rule as the
  E15 pathway audit. There is no depth filter; hops from the frozen T4 set are recorded descriptively only.
  The set is hashed and is not changed after the KC visual results.
- **Static visual input per KC class:**
  - direct VP edges and distinct VP presynaptic cells;
  - VP |W| relative to total input |W| (FlyBrain normalised W), with raw synapse counts reported
    alongside;
  - per-cell distribution.
  - Category: *none* (no direct VP edge), *substantial* (VP fraction ≥ 1 %) or *sparse* (anything else).
  - This measures anatomy only, not responsiveness.
- **KC → MBON matrix:** every KC type × every MBON type with at least one edge in W, recording edges,
  participating cells, signed / absolute / median / maximum weight, per-MBON-cell contribution, hemisphere
  pattern and raw synapses.
- **Static leverage:** candidate |W| from the KC class divided by total |W| into the MBON. It is reported
  pooled over the MBON type and per cell (median / maximum / minimum). It is a screening metric only.
- **DAN audit:** DAN types are `^PAM\d`, `^PPL1\d` and `^PPL2\d`. For each KC type × MBON pair the audit
  records DAN → MBON, DAN → KC-class, KC-class → DAN and MBON → DAN raw synapses, together with population
  size, laterality and transmitter.
  - **Topology-derived match:** at least 50 raw synapses onto the MBON type **and** at least 50 onto the KC
    class. The primary DAN is the match with the most DAN → MBON synapses.
  - Metadata names (hemibrain / flyWire) are reported separately and never used for matching.

## KC visual-response screen (60/8; seeds 3001–3006)

- **Pathway:** the frozen E27/E28 T4 injection, 60 decisions × 10 steps per trial.
- **Stimuli:** no_visual, gray, vertical, horizontal, drifting_vertical, pokemon_bedroom,
  pokemon_bedroom_after_walk (the corrected E27 label) and pokemon_walk.
- **Recording:** every KC cell.

**Visual gate (frozen; never adapted).** A KC class is visually responsive if at least one non-gray
stimulus meets all of the following:

- class size ≥ 10 cells;
- |relative Δ vs no-visual| ≥ 5 %;
- the same direction vs no-visual in ≥ 5/6 seeds;
- |relative Δ vs gray| ≥ 5 %;
- the same sign against both references.

Class Δ is based on total class spikes per trial. Subthreshold results are persisted.

Descriptive per-class metrics:

- baseline and gray rates, best stimulus, mean paired Δ, dz, fraction active;
- fraction of cells modulated (E27 definition: per-cell sign consistent in ≥ 5/6 seeds and
  |mean Δ| ≥ 0.2 · max(baseline, 1));
- median and upper-quartile per-cell modulation, and the top cells;
- for passing classes only:
  - a bootstrap CI over seeds;
  - per-seed Δ;
  - class-size sensitivity (random 50 % cell subsamples);
  - the share of |Δ| carried by the top 10 % of cells (broad vs subset).

These metrics are never used for selection.

## Candidate ranking

Written to `candidate-ranking-preregistration.md` and hash-frozen after the static audit and the visual gate,
before any leverage simulation.

- **Candidate unit:** one KC type → one MBON type, with its topology-matched primary DAN.
- **Eligibility:** the KC class passes the visual gate, at least one KC → MBON edge exists, and a matched DAN
  exists.
- **Ranking order:**
  1. passes the visual gate;
  2. matched DAN exists;
  3. pooled static MBON input fraction (descending);
  4. KC → MBON |W| sum;
  5. participating KC cells;
  6. then name (deterministic).
- Items 6–8 of the suggested ranking (minimal complexity, KC classes and MBON classes) are identical for all
  single-pair units.
- **References:** the E26 rows KCg-d → MBON01 (PAM01) and KCg-d → MBON11 (PPL101) are always run as labelled
  reference controls, whether or not they are in the top 10.
- **Cap:** at most 10 dynamically tested candidates, taken as the top 10 of the frozen ranking.

## Dynamic leverage (seeds 3011–3016)

- **Protocol:** as in E26–E28, all cells of the candidate KC class are driven at 0.3 V for 300 steps.
- **Conditions:**
  - *baseline:* existing edges untouched;
  - *zeroed:* every existing edge of that KC type → MBON type set to zero through the non-mutating
    plastic-delta mechanism (δ = −w0; `_W` is never modified).
- **Measure:** spikes of the whole target MBON type.
- **Regimes:** the primary regime is 60/8 (the regime of the visual screen). 80/3 is also measured for every
  tested candidate, so that the verdict class can be determined; it is the second of the two E28-defined
  regimes.
- **Leverage pass:** relative MBON change ≤ −10 % and lower in ≥ 5/6 seeds. MBON types are never pooled.

## Viability and confirmation

- **Viable:** visual gate pass, leverage pass, and a matched DAN.
  - *Coexisting:* leverage passes at 60/8, the same regime as the visual gate.
  - *Split:* leverage passes only at 80/3.
- **Confirmation set:** at most 3 candidates. Coexisting-viable candidates come first, in frozen-ranking
  order, then split-viable ones in ranking order.
- **Confirmation seeds:** 3021–3030, run once, with no retuning.
- **Consistency thresholds at confirmation:** scaled to ≥ 9/10 seeds (⌈10 · 5/6⌉) for both the visual gate
  and leverage.
- **Confirmation regimes:** a coexisting candidate is confirmed at 60/8 (visual and leverage). For a split
  candidate, visual is confirmed at 60/8 and leverage at 80/3.

## Two-regime sensitivity (confirmed candidates; seeds 3021–3030)

- Each confirmed candidate is measured at exactly 60/8 and 80/3 for:
  - the KC-class visual gate;
  - target-MBON leverage;
  - matched-DAN baseline and dynamic range (E28 fixed-amplitude assay: 200-step warm-up, 100-step baseline,
    10-step stimulation at amplitude 0.3). The DAN is usable if Δ ≥ 10 pp and positive in ≥ 9/10 seeds.
- If no candidate confirms, the best viable candidate by ranking (or else the top-ranked tested candidate)
  is run descriptively and labelled as such.

## Verdict classes

- **STRONG CANDIDATE FOUND:** the visual gate, ≥ 10 % dynamic leverage and a matched DAN all hold, the
  candidate confirms on held-out seeds, **and** visual coding and leverage coexist in 60/8 or in 80/3.
- **CANDIDATE BUT OPERATING-REGIME CONFLICT REMAINS:** a coherent KC → MBON → DAN candidate has visual coding
  and has leverage, but not in the same predefined regime.
- **NO BROADER CIRCUIT:** no candidate satisfies visual + leverage + DAN.

## E30 success requirements (documented now; binding on E30)

- **A — rule engagement:** KC eligibility exists; the correct DAN channel activates; the intended edges
  change; no-DAN, DAN-only and wrong-compartment-DAN controls do not produce an equivalent change.
- **B — magnitude:** enough edge contribution changes to produce a clear downstream effect.
- **C — persistence:** after a neural reset with the plastic state preserved, replaying the CS leaves the
  target MBON response changed relative to pre-conditioning and control.
- **D — specificity:** paired KC + matched DAN beats unpaired, paired + wrong DAN, and shuffled-timing or
  shuffled-eligibility controls.
- **E — replication:** the result holds on fresh seeds.
- **F — interpretability:** the effect is attributable to the frozen candidate circuit, with no hidden
  controller optimisation.

After E30 the biological track continues only if it is roughly ≥ 80 % of the way to live biological learning
(criteria 1–10 in the E29 task statement). Otherwise the recommendation is **PIVOT TO E25 LIVE CONTROLLER**.
