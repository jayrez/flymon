# Experiment 29 — Broader mushroom-body circuit audit

**Verdict: NO BROADER CIRCUIT** (on held-out confirmation).

**Recommendation: DO NOT PROCEED — PIVOT EARLY** (to the E25 live controller).

Summary:

- The broader audit found KC → MBON ← DAN circuits with far more MBON leverage than the original
  418-edge circuit.
  - The best units are α′β′ KC → MBON13/16/17. Zeroing their existing edges lowers MBON firing by 30–34 %
    in 10/10 fresh seeds.
  - Their matched PPL1 DANs have usable fixed-amplitude range.
- The visual component did **not** replicate.
  - On the 6 screen seeds, four KC classes passed the frozen visual gate at 60 ms + APL×8: KCapbp-ap1,
    KCapbp-ap2, KCapbp-m and KCab-p.
  - On the 10 held-out confirmation seeds, **no KC class passed**, at 60/8 or at 80/3.
  - The screen "responses" of the α′β′ classes, which have zero or negligible direct visual-projection
    input, behave like population-wide trial-state fluctuations. They do not look like stimulus coding.
- The limiting problem has therefore moved from *leverage* to the *reliability of Pokémon-driven KC
  activity*. That is item 1 of the 80 % continuation criterion.

No learning, conditioning, learning-rate calibration or weight mutation was run. `_W` was verified unchanged
in every simulation, and no emulator RAM entered the model.

## 1. Repository and freezes

| Item | Value |
|---|---|
| Base | `origin/main` 50e3685665457ff434ff9d9bae9ac1fdd8a385f8 (E28 merge; contains E26 336f06b / E27 b36e30d / E28 a576fc6) |
| Branch | `experiment-29-broader-mb-circuit-audit` |
| `preregistration.md` | hash-frozen in `preregistration-freeze.json` before any simulation |
| Seeds | `seed-freeze.json`: 3001–3006 visual, 3011–3016 leverage, 3021–3030 confirmation; disjoint and unused before |
| Candidate ranking | `candidate-ranking-preregistration.md`, hash-frozen in `candidate-ranking-freeze.json` before any leverage simulation |

The confirmation stage also ran a **descriptive, non-preregistered** diagnostic: target MBON and DAN activity
under Pokémon drive alone, with the candidate edges intact vs zeroed. It is labelled as such in the JSON and
was used for no decision.

## 2. KC population audit (`kc-population-audit.json`)

**4,064 KCs in 12 types.** The type counts sum to the total; body IDs are persisted; population sha256 is in
the JSON.

| Type | Cells | L / R |
|---|---|---|
| KCab | 1,681 | 821 / 860 |
| KCg-m | 1,346 | 691 / 655 |
| KCapbp-ap2 | 291 | 133 / 158 |
| KCg-d | 206 | 99 / 107 |
| KCapbp-m | 205 | 110 / 95 |
| KCapbp-ap1 | 197 | 96 / 101 |
| KCab-p | 129 | 64 / 65 |
| KCg-s1 | 2 | — |
| KCg-s2 | 2 | — |
| KCa′b′-ap1 | 2 | — |
| KC | 2 | — |
| KCg | 1 | — |

All KCs have a consensus transmitter of ACh (prediction only). Incoming and outgoing edge counts, |W| and raw
synapse totals are in the JSON.

## 3. Visual-projection set (`visual-projection-set.json`)

- **Rule:** superclass `visual_projection`, the same rule as the E15 pathway audit. There is no depth filter.
- **Size:** 9,201 cells, 323 types, 4,589 L / 4,612 R.
- **Hashes:** body-ID sha256 3f31cf6de31b51a3…; file sha256 1f5f5f6d… frozen in `preregistration-freeze.json`.
- **Distance from the frozen T4 set** (6,845 cells): 2,293 VPNs are 1 hop, 6,869 are 2 hops, 38 are 3 hops,
  and 1 is unreached.
- The set was not modified after any result.

## 4. Static visual input (connectivity only)

| KC type | Direct VP edges | VP cells | VP share of input \|W\| | Category | Top VP types |
|---|---|---|---|---|---|
| KCg-d | 1,088 | 200 | 7.13 % | substantial | aMe12, MTe30, LTe25 |
| KCab-p | 347 | 33 | 2.48 % | substantial | aMe26, MTe37, LTe72 |
| KCg-s1 (2 cells) | 115 | 87 | 19.7 % | substantial | — |
| KCg (1 cell) | 16 | 16 | 6.8 % | substantial | — |
| KCg-m | 41 | 29 | 0.06 % | sparse | aMe12, MTe22, LTe16 |
| KCapbp-ap1 | 1 | 1 | 0.005 % | sparse | LTe51 |
| KCab | 1 | 1 | 0.001 % | sparse | LTe51 |
| KCg-s2 | 7 | 7 | 0.60 % | sparse | — |
| KCapbp-ap2, KCapbp-m, KCa′b′-ap1, KC | 0 | 0 | 0 | none | — |

Among classes with at least 10 cells, only KCg-d and KCab-p receive substantial direct visual-projection
input. This reproduces the E26 audit.

## 5. KC visual-response screen (60 ms + APL×8, seeds 3001–3006; `kc-visual-response.json`)

| KC type | Cells | VP input | Best stimulus | Δ no-visual | Δ gray | Consistency | Cells modulated | Gate |
|---|---|---|---|---|---|---|---|---|
| KCapbp-ap2 | 291 | 0 % | pokemon_bedroom | +37.8 % | +32.5 % | 5/6 | 0.59 | **pass** |
| KCapbp-ap1 | 197 | 0.005 % | pokemon_bedroom | +35.5 % | +24.0 % | 6/6 | 0.69 | **pass** |
| KCapbp-m | 205 | 0 % | pokemon_bedroom | +30.1 % | +26.5 % | 5/6 | 0.46 | **pass** |
| KCab-p | 129 | 2.48 % | pokemon_walk | −6.2 % | −6.6 % | 5/6 | 0.02 | **pass** |
| KCg-d | 206 | 7.13 % | pokemon_walk | +15.5 % | +10.6 % | 4/6 | 0.14 | fail (consistency) |
| KCg-m | 1,346 | 0.06 % | bedroom_after_walk | −6.7 % | −5.9 % | 4/6 | 0.00 | fail |
| KCab | 1,681 | 0.001 % | pokemon_walk | −4.6 % | −4.9 % | 4/6 | 0.00 | fail |
| KCa′b′-ap1 | 2 | 0 % | pokemon_bedroom | +33.8 % | +32.0 % | 6/6 | 0.50 | fail (size < 10) |
| KCg-s1 | 2 | 19.7 % | bedroom_after_walk | −14.8 % | −9.4 % | 4/6 | 0.50 | fail |
| KCg-s2 | 2 | 0.6 % | pokemon_bedroom | +6.0 % | +8.2 % | 4/6 | 0.00 | fail |
| KCg | 1 | 6.8 % | vertical | −17.9 % | −15.6 % | 3/6 | 0.00 | fail |
| KC | 2 | 0 % | vertical | +2.4 % | +3.0 % | 4/6 | 0.00 | fail |

Reliability metrics for the passing classes (descriptive):

- **Bootstrap 95 % CI** of the relative Δ:
  - KCapbp-ap2 +15 % to +63 %;
  - KCapbp-ap1 +14 % to +65 %;
  - KCapbp-m +7 % to +55 %;
  - KCab-p −12 % to +2 %.
- **Half-subsample gate pass rate:** 1.0 for the α′β′ classes, 0.94 for KCab-p.
- **Pattern:** "broad" (top 10 % of cells carry < 50 % of |Δ|).

## 6. KC → MBON connectivity and static leverage (`kc-mbon-connectivity.json`, `static-leverage-screen.json`)

The full 12 KC types × 36 MBON types matrix is persisted: 258 pairs with edges; the matrix hash is in the
JSON. Strongest static targets (candidate |W| as a share of MBON input |W|, pooled over the MBON type):

| KC class | Strongest MBON targets (static input share) |
|---|---|
| KCapbp-ap2 | MBON17 42.7 %, MBON16 41.3 %, MBON13 41.0 %, MBON03 39.1 % |
| KCapbp-m | MBON13 29.9 %, MBON17 29.2 %, MBON15-like 28.3 % |
| KCapbp-ap1 | MBON16 28.2 %, MBON28 27.6 %, MBON31 22.3 % |
| KCab-p | MBON19 24.1 %, MBON06 10.3 %, MBON23 9.8 % |
| KCg-d (not visually passing here) | MBON27 41.4 %, MBON05 10.6 %, MBON33 8.7 %, MBON20 8.5 %, MBON09 8.4 %, MBON32 7.7 %, **MBON01 7.6 %**, **MBON11 6.9 %** |

The original E26 edges rank 7th and 8th even among KCg-d's own targets, and 130th and 131st overall. Of all
MBONs, the E26 audit chose two where KCg-d supplies only ~7 % of the input.

## 7. DAN compartment audit (`dan-compartment-audit.json`)

**Topology match:** at least 50 raw synapses DAN → MBON **and** at least 50 DAN → KC-class. Metadata names are
reported separately; in MaleCNS they are just the type codes, so the metadata adds no independent compartment
label.

| Unit | Matched DAN (n, L/R, ground-truth NT) | DAN→MBON | DAN→KC | KC→DAN | MBON→DAN |
|---|---|---|---|---|---|
| KCapbp-ap2 → MBON17 | PPL104 (2, 1/1, dopamine) | 118 | 983 | 1,220 | 68 |
| KCapbp-ap2 → MBON16 | PPL104 | 302 | 983 | 1,220 | 160 |
| KCapbp-ap2 → MBON13 | PPL105 (2, 1/1, dopamine) | 657 | 1,252 | 1,984 | 17 |
| KCg-d → MBON01 (ref) | PAM01 (44, 23/21, dopamine) | 1,597 | 3,254 | 3,538 | 128 |
| KCg-d → MBON11 (ref) | PPL101 (2, 1/1, dopamine) | 2,311 | 634 | 1,216 | 205 |
| KCg-d → MBON27 | PAM01 | 305 | 3,254 | 3,538 | 4 |
| KCab-p → MBON19 | PPL105 | 98 | 918 | 804 | 14 |

This is a literature interpretation only; it was not used for matching. In hemibrain nomenclature these are
presumably α′3 (PPL104; MBON16/17) and α′2 (PPL105; MBON13) compartments of the α′β′ lobe.

## 8. Frozen candidate ranking and dynamic leverage (seeds 3011–3016; `dynamic-leverage-results.json`)

There were 77 eligible units. The top 10 were tested; the E26 references were run as labelled controls.

| Rank | KC → MBON | DAN | Edges | Static input | Δ at 60/8 | Consistency | Δ at 80/3 | Consistency |
|---|---|---|---|---|---|---|---|---|
| 1 | KCapbp-ap2 → MBON17 | PPL104 | 285 | 42.7 % | −28.8 % | 6/6 | −31.4 % | 6/6 |
| 2 | KCapbp-ap2 → MBON16 | PPL104 | 286 | 41.3 % | −31.7 % | 6/6 | −37.6 % | 6/6 |
| 3 | KCapbp-ap2 → MBON13 | PPL105 | 290 | 41.0 % | −30.5 % | 6/6 | −29.5 % | 6/6 |
| 4 | KCapbp-ap2 → MBON03 | PAM06 | 577 | 39.1 % | −26.7 % | 6/6 | −21.7 % | 6/6 |
| 5 | KCapbp-m → MBON13 | PPL105 | 205 | 29.9 % | −40.8 % | 6/6 | −32.8 % | 6/6 |
| 6 | KCapbp-m → MBON17 | PPL104 | 201 | 29.2 % | −36.2 % | 6/6 | −33.1 % | 6/6 |
| 7 | KCapbp-m → MBON15-like | PPL103 | 380 | 28.3 % | −33.1 % | 6/6 | −28.7 % | 6/6 |
| 8 | KCapbp-ap1 → MBON16 | PPL104 | 181 | 28.2 % | −40.3 % | 6/6 | −39.5 % | 6/6 |
| 9 | KCapbp-ap2 → MBON15-like | PPL103 | 482 | 28.0 % | −23.0 % | 6/6 | −24.1 % | 6/6 |
| 10 | KCapbp-ap1 → MBON28 | PPL104 | 182 | 27.6 % | −56.1 % | 6/6 | −56.6 % | 6/6 |
| ref | KCg-d → MBON01 | PAM01 | 212 | 7.6 % | −2.5 % | 2/6 | −16.6 % | 6/6 |
| ref | KCg-d → MBON11 | PPL101 | 206 | 6.9 % | −1.5 % | 3/6 | −15.0 % | 6/6 |

All 10 candidates pass leverage in the same regime as their visual screen (60/8). At the screen stage all 10
were therefore **"coexisting-viable"** (`candidate-circuits.json`). Dynamic leverage tracks static input
share, but it is not proportional to it.

## 9. Confirmation (seeds 3021–3030, run once; `confirmation-results.json`)

Confirmation set (frozen order): KCapbp-ap2 → MBON17, → MBON16, → MBON13.

| Unit | Visual gate at 60/8 | Leverage at 60/8 | Matched DAN at 60/8 | Confirmed |
|---|---|---|---|---|
| KCapbp-ap2 → MBON17 | **fail** | −30.3 % (10/10) | PPL104 +15.2 pp (10/10) | **no** |
| KCapbp-ap2 → MBON16 | **fail** | −33.5 % (10/10) | PPL104 +15.2 pp | **no** |
| KCapbp-ap2 → MBON13 | **fail** | −32.2 % (10/10) | PPL105 +10.7 pp | **no** |

Visual detail for KCapbp-ap2 at 60/8:

- **Best stimulus:** pokemon_bedroom, +11.8 % vs no-visual but consistent in only 7/10 seeds.
- **Against gray:** +4.0 % (6/10).
- **Cells modulated:** 0 %.
- **Other classes:** no KC class of any type passed the gate on the confirmation seeds.
- **Per-seed Δ vs no-visual:** ranges from −1,003 to +2,233 spikes. On the screen seeds it ranged from −125 to
  +2,150.
- **Interpretation:** the whole α′β′ population moves together from trial to trial, which is a trial-state
  effect. On 6 seeds it happened to line up.

## 10. Two-regime sensitivity (`operating-regime-sensitivity.json`; exactly 60/8 and 80/3)

| Candidate | Regime | Visual gate | MBON leverage | DAN usable | KC rate | MBON rate |
|---|---|---|---|---|---|---|
| KCapbp-ap2 → MBON17 | 60/8 | fail (+11.8 %, 7/10) | −30.3 % pass | PPL104 +15.2 pp, yes | 0.83 Hz | 7.0 Hz |
| KCapbp-ap2 → MBON17 | 80/3 | fail (+2.6 %) | −30.4 % pass | PPL104 +12.8 pp, yes | 2.48 Hz | 7.0 Hz |
| KCapbp-ap2 → MBON16 | 60/8 | fail | −33.5 % pass | PPL104 +15.2 pp, yes | 0.83 Hz | 7.2 Hz |
| KCapbp-ap2 → MBON16 | 80/3 | fail | −37.6 % pass | PPL104 +12.8 pp, yes | 2.48 Hz | 7.0 Hz |
| KCapbp-ap2 → MBON13 | 60/8 | fail | −32.2 % pass | PPL105 +10.7 pp, yes | 0.83 Hz | 7.2 Hz |
| KCapbp-ap2 → MBON13 | 80/3 | fail | −30.5 % pass | PPL105 +11.6 pp, yes | 2.48 Hz | 7.0 Hz |

Leverage and DAN range are regime-robust. Visual coding is absent in both regimes.

**Descriptive only** (Pokémon drive, no artificial KC injection, 10 seeds). At 60/8 the target MBONs change by
only +2 % to +6 % with the visual stimulus. Zeroing the candidate edges under that visual drive still lowers
them by 15–29 % (7–9/10 seeds). So these edges carry MBON output during ordinary activity, but the MBON output
carries almost no stimulus information.

## 11. Original 418-edge comparison

- **Static share:** the E26 edges supply ~7 % of MBON01/11 input.
- **Leverage:** they reach ≥ 10 % only at 80/3 (−16.6 % / −15.0 %). At 60/8 they reach −2.5 % / −1.5 %.
- **Best new units:** 4–6× the static share, and 25–56 % dynamic leverage in *both* regimes.

The "too narrow" hypothesis is therefore **correct for leverage**. It does not rescue visual coding: the
classes with real visual anatomy (KCg-d, KCab-p) did not pass the gate on the confirmation seeds, and the
classes with leverage have no visual anatomy.

## 12. Required answers

1. **Base commit:** 50e3685665457ff434ff9d9bae9ac1fdd8a385f8.
2. **How many KC types exist?** 12.
3. **How many KC cells exist?** 4,064.
4. **Which KC types receive direct visual-projection input?**
   - Substantial: KCg-d (7.1 %), KCab-p (2.5 %), and the tiny KCg-s1 (2 cells) and KCg (1 cell).
   - Sparse: KCg-m (0.06 %), KCapbp-ap1, KCab and KCg-s2.
   - None: KCapbp-ap2, KCapbp-m, KCa′b′-ap1 and KC.
5. **Which KC types respond to Pokémon/T4 stimulation?**
   - On the screen seeds: α′β′ KCs (+30–38 % to the bedroom frame), KCab-p (−6 %) and KCg-d (+15 %, but
     inconsistent).
   - On held-out seeds, none reliably.
6. **Which pass the preregistered visual gate?**
   - Screen: KCapbp-ap1, KCapbp-ap2, KCapbp-m and KCab-p.
   - Confirmation: **none**.
7. **Fraction of cells modulated** (screen): ap1 0.69, ap2 0.59, m 0.46, KCab-p 0.02. At confirmation,
   KCapbp-ap2 had 0.00.
8. **Which MBON types receive the strongest input from those KC classes?**
   - ap2: MBON17, MBON16, MBON13 and MBON03.
   - m: MBON13, MBON17 and MBON15-like.
   - ap1: MBON16 and MBON28.
   - KCab-p: MBON19.
9. **Static MBON input share per candidate class:** 39–43 % (ap2), 28–30 % (m), 28 % (ap1), 24 % (KCab-p →
   MBON19). See §6.
10. **How does the original circuit rank statically?** KCg-d → MBON01 is 7.6 % and → MBON11 is 6.9 %: 7th and
    8th among KCg-d targets, and 130th / 131st of all pairs. It was not eligible here because KCg-d failed the
    gate.
11. **Which candidates were frozen for dynamic testing?** The ten units listed in §8.
12. **Which produce ≥ 10 % MBON effect dynamically?** All 10 at 60/8 and at 80/3 (−23 % to −57 %).
13. **Which have plausible matched DAN populations?** All 10 (PPL104, PPL105, PPL103 and PAM06, by topology).
14. **Which full KC → MBON → DAN circuits are viable?** All 10 at screen stage. None survives confirmation,
    because the visual gate fails.
15. **Which candidate ranks first, and why?** KCapbp-ap2 → MBON17 (PPL104). It is the largest static input
    share (42.7 %) among visually passing classes with a matched DAN.
16. **Does it confirm on fresh seeds?** No. Leverage and DAN confirm (10/10); the visual gate does not.
17. **Does its visual response survive at 80/3?** No (+2.6 %).
18. **Does its leverage survive at 60/8?** Yes (−30.3 %, 10/10).
19. **Does any candidate have visual coding and leverage in the same regime?** Only on the 6 screen seeds
    (60/8). Not on held-out seeds.
20. **Is E30 justified?** No, not as a Pokémon-driven learning experiment. A reliable Pokémon-driven CS does
    not exist in any tested regime.
21. **What exact circuit should E30 freeze?** None is justified. If E30 is run anyway as a pure
    plasticity-mechanics test with artificial KC drive, the circuit is KCapbp-ap2 → MBON17 (285 existing
    edges; edge sha256 in `dynamic-leverage-results.json`) gated by PPL104. That test cannot satisfy
    continuation criterion 1.
22. **How likely is E30 to meet the 80 % continuation threshold?** Low, roughly 10–15 %.
    - Criteria 2 and 3 (leverage, DAN) are met.
    - Criteria 4–8 are plausible, given the E26 machinery.
    - Criterion 1 (reliable Pokémon-driven KC activity) failed on held-out seeds for every class.
    - Criterion 9 (no unresolved circuit or regime problem) fails.
    - Fixing the visual front end would need a new rescue intervention of the kind the decision rule
      forbids.

## 13. E30 recommendation

**DO NOT PROCEED — PIVOT EARLY.**

- **Current progress** toward the 80 % continuation threshold: about 35 %. Leverage and DAN range are now
  solved structurally; plasticity machinery exists from E26.
- **Biggest blocker:** Pokémon frames do not reliably drive any KC population on held-out seeds, in either
  E28-defined regime.
- **Why not continue:** E30 as specified would validate plasticity on a circuit whose conditioned stimulus is
  not a Pokémon visual event.

After E30, the biological track continues only if it is roughly ≥ 80 % of the way to live biological learning
(Pokémon frames reliably drive KCs; KC leverage on a coherent MBON; usable matched DAN; DAN-gated,
edge-specific plasticity; measurable MBON change; KC + DAN timing specificity; persistence across neural
resets; held-out replication; no unresolved saturation / leverage / circuit problem). Otherwise:
**PIVOT TO E25 LIVE CONTROLLER.** E29 already shows criterion 1 unmet, so pivoting now saves an experiment.

E25 is to be described as an externally evolved readout of frozen fly visual-pathway activity, not
within-lifetime biological learning.
