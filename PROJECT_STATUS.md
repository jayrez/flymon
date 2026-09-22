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
| 15 | Can a connectivity-selected biological pathway carry five-way screen-family identity into DNs? | **FAIL** (for scene identity); collapse across T5 → visual-projection → DN |
| 16 | Does the pathway preserve biological motion / optic-flow / looming features? | **FAIL** (Category A); T4/T5 do not reproduce direction selectivity — failure is optic-lobe dynamics, upstream of Exp-15's transfer |
| 17 | Why does retinal motion input not become T4/T5 direction selectivity, and can dynamics repair it? | **FAIL** (Category E); stimuli and gain exonerated, signal flow restored, but the LIF architecture cannot compute motion |
| 18 | Does MaleCNS contain the spatial geometry for T4/T5 direction selectivity, and can a column-resolved model recover it? | **FAIL** (Category D) but **geometry PASS**; the four-direction geometry is in the connectome, yet only an explicit correlator reads it out |
| 19 | What minimal mechanism converts that geometry into direction selectivity — is a correlator really required? | **FAIL** (Category F); extended timing and hidden-transient explanations rejected; only antisymmetric opponency on sign-preserving drives works (0.356, 8/8) |
| 20 | Can a minimal sign-preserving conductance / two-compartment dendrite reproduce that opponent computation? | **FAIL** (Category E); small bias only (0.025 vs oracle 0.356) and it is insensitive to arm order, sign and geometry |

## Latest result (Experiment 15)

Driving the Experiment-6 dataset through the biological Experiment-5 retina
reproduces the degradation pattern on held-out image instances: R1–R6 97.0 %,
lamina 96.4 %, T4 50.5 %, **T5 80.9 %**, visual-projection 28.9 %, all-DN 20.6 %
(chance 20 %). A label-free connectome audit ranks DNs by a structural
restart-diffusion visual-influence score; its top cells are the known looming/escape
cluster (DNp01/03/04/11) and the strongest T5 targets are LPLC2/LPLC1/VS — but no
connectivity-selected DN or visual-projection subset exceeds chance on **five-way
screen-family identity** (properly nested anatomical endpoint 22.0 %, p = 0.096). The
decisive control: the *same* 20 P20 neurons score **87.5 %** via the engineered
LC10a/LPLC2 injection but **18.9 %** via the biological retina. Scene-family identity
is thus not preserved through the biological optic-lobe cascade. DNg13 ranks
486/1314 structurally and did not preserve screen-family identity here, but its
steering / optic-flow role is untested. Experiment 15 tested only scene identity;
whether motion / optic-flow / looming features survive is the Experiment-16 question.
No closed-loop test was run.

## Latest result (Experiment 16)

Experiment 16 answered the Experiment-15 open question with ethological synthetic
motion / looming / optic-flow stimuli through the same frozen retina (held-out CNS
seeds 1121–1140, matched-seed permutation tests). The **retina (R1–R6) robustly
encodes motion** (motion-vs-frozen change response +38 spikes for optic flow, +13 for
OFF motion, p = 0.0001, sign-consistency 1.00), but **T4/T5 do not reproduce
direction selectivity** (max |DSI| = 0.012; no subtype significant) and their
change response is negligible (≤ 6 % of baseline). The preregistered primary endpoint
(T5 OFF-motion direction selectivity) was null (0.005 spike, p = 0.80). **Gate A
failed → Category A:** the current optic-lobe LIF dynamics do not compute motion, so
the VP and DN transfer stages were not tested. This localises the Experiment-15
collapse to the optic-lobe dynamics themselves (upstream of the T5 → VP → DN transfer)
rather than to biologically appropriate feature compression. Recommended next step is
**Experiment 17: dynamics / gain repair** (with a flyvis functional positive control),
not closed-loop control. flyvis was not installed and was deliberately not forced.

## Latest result (Experiment 17)

Experiment 17 asked *why* Experiment 16 failed and whether a biologically constrained
dynamics change could repair it, using the unchanged Experiment-16 stimuli, fresh
calibration seeds 1201–1220 and held-out seeds 1221–1240.

Three findings, in order of increasing importance:

1. **The stimuli are fine.** A Hassenstein-Reichardt correlator on the *same* retina
   samples gives **|DSI| = 1.0** with a clean sign flip, exactly 0 for gray and frozen
   input, 76 % suppression under temporal shuffling, and sign reversal under time
   reversal. (flyvis 1.2.0 installed in an isolated directory but its pretrained
   weights were unobtainable, so it could not serve as the positive control.)
2. **Gain is irrelevant.** Across 0.5×–8× retinal gain, R1–R6 firing rises 5×
   (26 → 131) while T4/T5 firing does not move (2.3 / 2.2) and max |DSI| stays at
   noise with no trend. **Gain hypothesis rejected.**
3. **Signal flow was genuinely broken, and was genuinely repaired — without
   restoring the computation.** Every R1–R6 → lamina weight in MaleCNS is inhibitory
   (9,636 edges, negative fraction 1.000), so in a spiking model with no maintained
   LMC depolarisation the lamina is pinned at its floor. Adding a tonic term raised
   L1 12.6×, L2 41×, Tm2 11× and T5a 4.7×, and made the cascade stimulus-locked — yet
   the held-out primary endpoint stayed null: T5 OFF preferred−null **−0.0014 spike,
   DSI −0.00011, p = 0.950**, CI [−0.041, +0.040].

**Verdict FAIL, Category E (no simple repair).** The excitatory and inhibitory arms
show a −1 step lag for moving *and* frozen stimuli alike, i.e. no motion-dependent
temporal structure. The underlying reason is architectural: a motion detector computes
a *correlation*, while FlyBrain integrates a linear weighted sum of spikes and
thresholds it, so delayed inhibition plus undelayed excitation yields a linear
asymmetry, never a correlator. This explains Experiments 15 and 16 together — the
optic lobe transmits luminance change but never computes motion, so nothing
motion-specific can reach visual-projection or descending neurons. Experiment 18
should **not** proceed to VP → DN transfer; it needs an architecturally different
optic-lobe front end (graded photoreceptor/LMC transfer plus a nonlinear T4/T5
interaction, or a flyvis-derived front end validated against the HR control).

## Latest result (Experiment 18)

Experiment 18 went upstream of Experiment 17's whole-cell-type interventions and asked
whether the **optic-column geometry** needed for direction selectivity is present at
all, using only released `assignedOlHex` annotations (no stimulus, no labels).

**Geometry gate: PASS — and this revises Experiment 17.** Measuring, per T4/T5 neuron,
the offset between the weighted centroids of its fast arm (Mi1 for T4; Tm1/Tm2 for T5)
and its slow/sign-inverting arm (Mi9/Mi4; Tm9) recovers the canonical four-direction
layout from connectivity alone: antiparallel pairs (T4a↔T4b 179.3°, T5a↔T5b 178.2°) on
orthogonal axes (95.7°, 98.4°), ~1,700 cells per subtype at 99 %+ eligibility, with T5
orientation very tight (R ≈ 0.88). Experiment 17 applied delays at the whole-cell-type
level, which is spatially blind, so it could never have used this geometry.

**But no biologically simple model reads it out.** Across a preregistered hierarchy —
M0 instantaneous linear (mean |DSI| exactly **0.0000**, the expected analytic result),
M1 + cell-class temporal filters, M2 + rectification, M3 + output squaring — none met
the calibration threshold. The selected M1 reaches held-out mean |DSI| 0.045 but is
*incoherent with its own anatomical prediction*: two subtypes (T4a, T5a) significantly
prefer the **opposite** direction. By contrast an explicit multiplicative correlator
(`M_HR`, a control architecture, not a biological claim) run on the **same** geometry
reaches **0.354** and agrees with anatomy in **81–94 %** of responsive neurons for
**all eight** subtypes. Geometry-shuffle (−62 %) and temporal-flat (−52 %) ablations
confirm both ingredients are causally involved; frozen and gray controls give exactly
zero and temporal shuffle removes 59 %, so the model is genuinely motion-driven —
just not direction-selective. ON/OFF specificity is absent.

**Verdict FAIL, category D** (geometry present, only an explicit correlator succeeds).
The connectome supplies the *spatial* half of a motion detector in full; what the
linear/rectifying family — and FlyBrain's LIF neuron — lacks is the *multiplicative*
half. flyvis remained unobtainable (F3: no console script, no checkpoint download
path), so category E is still untested. Experiment 19 should **not** proceed to
VP → DN transfer; it should settle flyvis and test a genuinely multiplicative/divisive
dendritic interaction on this same geometry.

## Latest result (Experiment 19)

Experiment 19 asked what minimal mechanism turns Experiment-18's T4/T5 geometry into
direction selectivity, and deliberately tested the two least interesting explanations
first.

**Both were rejected.** Extending `tau_slow` from E18's edge value 8 out to 24 lifted
mean |DSI| only 0.052 → 0.093 with coherence stuck at 2/8, so E18's failure was **not**
a temporal-range artifact (H0). Three predetermined temporal metrics — whole-sequence
mean, a stimulus-defined motion window, and an anatomy-local event window around each
neuron's own predicted receptive-field crossing — all agree and are all negative, so no
transient was hidden by mean-rate averaging (H1).

**The decisive result is a 2 × 2 decomposition** (held-out mean DSI, coherent subtypes):

| | product only | antisymmetric opponent |
|---|---:|---:|
| **signed drives** | +0.056 [4/8] | **+0.356 [8/8]** |
| **magnitude drives** | −0.104 [1/8] | +0.036 [4/8] |

Both ingredients are necessary and neither suffices: plain multiplication does **not**
work, and opponency on sign-stripped inputs does not either. Only their conjunction is
coherent (all eight subtypes positive, mean sign consistency 0.84, min subtype +0.317,
all Holm p < 1e−70). Coincidence, divisive/shunting (|DSI| ≈ 0.01–0.06) and a generic
dendritic subunit all failed.

The selected biological model inverted all four **T4** subtypes (−0.22 to −0.29), and
that is mechanistically informative: T4's slow arm is inhibitory (Mi9, Mi4) while T5's
is excitatory (Tm9), so a rule built on rectified magnitudes discards exactly the sign
asymmetry the computation consumes. Ablations confirm the substrate is sound —
geometry shuffle p = 0.005 (~42 SD from a tight null), temporal-flat −66 %, interaction
ablation −82 % — and **ON/OFF specificity was restored** (all four T4 prefer ON, all
four T5 prefer OFF) with no subtype-specific coefficients. Resolved-weight
stratification shows the ~42 % coverage limit is not the cause.

**Verdict FAIL, category F.** The supportable claim is that a minimal interaction of
the *antisymmetric opponent class computed on sign-preserving drives* is sufficient to
read out the MaleCNS geometry — not that biology evaluates the textbook HR equation
(indeed the minimal 1-frame opponent product generalised better than E18's HR
formulation). **Experiment 20 should not proceed to VP→DN**; the working mechanism is
a control architecture, not a validated biological readout. Next steps: resolve flyvis
(still F3, settling category E) and implement a genuine conductance/two-compartment
dendritic model rather than a divisive approximation.

## Latest result (Experiment 20)

Experiment 20 asked whether plausible membrane dynamics can implement Experiment-19's
signed-opponent computation **without evaluating the opponent equation**. Synaptic sign
was mapped to conductance channels physically: weight sign selects excitatory vs
inhibitory, reversal potential supplies the polarity, and conductances stay
non-negative.

**Verdict FAIL, category E.** Held-out mean DSI: C0 single-compartment **+0.0067**,
C1 two-compartment **+0.0057**, C2 two-compartment + output threshold **+0.0246** —
against the E19 oracle's **+0.3562** (8/8 coherent). The selected model reaches 6.9 % of
the oracle with only 5/8 subtypes positive and two (T4a, T5a) significantly inverted.

The important part is mechanistic, not the missed threshold. **Swapping which arm feeds
which compartment changed the result by 2.0 %**; a delay-and-compare mechanism is
*defined* by that ordering, so near-invariance means the model is not computing anything
spatially ordered. Consistently, **geometry shuffle leaves 77 % of the effect intact**
(null +0.0144 vs observed +0.0187 over 200 permutations), destroying synaptic sign costs
only 22 %, and neutralising inhibitory reversal only 6.5 % — yet sign and inhibition were
exactly what E19 identified as essential. Coupling removal (−90 %) and temporal
flattening (−62 %) do matter, but coupling dependence alone is not evidence of an
opponent computation: a coupled-RC-plus-threshold cascade yields a small directional bias
for any smooth drifting stimulus.

One clear positive: **ON/OFF specificity is reproduced strongly** — all four T4 prefer ON
(~40–60×) and all four T5 prefer OFF (~13×) with no per-subtype coefficients, because the
conductance mapping preserves the T4 fast-excitatory/slow-inhibitory vs T5 both-excitatory
asymmetry. Pathway polarity is not what is missing; directional ordering is.

**Experiment 21 should not proceed to VP→DN.** The E19 oracle remains a computational
constraint whose biological implementation is unresolved. The proposed next step is a
*structurally asymmetric* dendrite — on-path/location-dependent inhibition, non-reciprocal
coupling, nonlinear subunits, or active conductances — with the arm-swap ablation adopted
as an early rejection criterion.

## Reproducing and navigating

- Experiment code is at the repository root (`run_*_experiment.py` / `analyze_*`),
  with trial data and reports beside each `results/experiment-NN-*/analysis.md`.
- Experiment 15: `run_pathway_audit.py` (label-free connectivity),
  `run_biological_pathway_experiment.py` (biological retina sweep),
  `analyze_biological_pathway_experiment.py` (held-out analysis);
  connectivity tracing lives in `flymon/pathway.py`.
- Experiment 20: `run_conductance_dendrite_experiment.py` (calibrate/heldout),
  `analyze_conductance_dendrite_experiment.py`; membrane model in
  `flymon/conductance_dendrite.py` (reuses E18 geometry and E19 stimuli/metrics).
- Experiment 19: `run_motion_nonlinearity_experiment.py` (calibrate/heldout),
  `analyze_motion_nonlinearity_experiment.py`; mechanism family in
  `flymon/motion_nonlinearity.py` (reuses the E18 geometry unchanged).
- Experiment 18: `run_column_motion_experiment.py` (geometry/calibrate/heldout),
  `analyze_column_motion_experiment.py`; column geometry in `flymon/optic_columns.py`
  and the opt-in graded model in `flymon/column_motion.py`.
- Experiment 17: `run_optic_dynamics_experiment.py` (audit/gain/calibrate/traces/heldout),
  `run_positive_control.py`, `analyze_optic_dynamics_experiment.py`; opt-in dynamics
  adapter in `flymon/optic_dynamics.py` (reference path bit-identical to stock FlyBrain).
- Tests: `test_generalization.py`, `test_interface.py`, `test_progression.py`,
  `test_pathway.py`, `test_ethology.py`, `test_optic_dynamics.py`,
  `test_column_motion.py`, `test_motion_nonlinearity.py`,
  `test_conductance_dendrite.py` (CPU only;
  `FLYMON_GPU_TESTS=1` adds the GPU reference-equivalence check).
- Set `POKEMON_ROM` to a locally owned ROM for capture steps; keep it outside the
  repository.
