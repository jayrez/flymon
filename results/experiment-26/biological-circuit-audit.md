# Experiment 26 — Biological circuit audit (Phases 1–2)

Every population below was resolved independently from Flymon's local MaleCNS v1.0 sources.
No neuron ID is copied from any other project (DOOMFLY included).

**Sources and SHA-256** (also in `circuit-audit.json`):

| file | SHA-256 |
|---|---|
| `brain.npz` | `cc9bd1ec…` |
| `weights.npz` | `c29919aa…` |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | `2177e246…` |
| `body-neurotransmitters-male-cns-v1.0.feather` | `95c92892…` |

Raw synapse counts come from `connectome-weights-male-cns-v1.0-minconf-0.5.feather`.

* **Script:** `audit_learning_circuit.py` (deterministic).
* **Records:**
  * `population-records.json.gz` — per neuron: FlyBrain index, bodyId, type, side,
    annotation type / flyWire / hemibrain type, class, superclass, consensus / predicted /
    ground-truth transmitter and confidence;
  * `circuit-audit.json` — connectivity;
  * `proposed-plastic-edges.json`.

## 1. Populations found

**Selectors.** Populations are selected by `cell_type` regex on the FlyBrain index. They are
cross-checked against the annotation `class`, and the transmitter comes from
`body-neurotransmitters` (`consensus_nt`, `ground_truth`).

| population | selector | n | types | class | consensus NT | ground-truth NT |
|---|---|---|---|---|---|---|
| Kenyon cells | `^KC` | 4,064 | 12 | `Kenyon_Cell` (all) | ACh 4,064 | none annotated |
| MBONs | `^MBON` | 97 | 36 | `MBON` (all) | ACh 50, Glu 26, GABA 21 | ACh 32, Glu 16, GABA 17, none 32 |
| PAM DANs | `^PAM\d` | 316 | 15 | `DAN` (all) | dopamine 316 | dopamine 316 |
| PPL1 DANs | `^PPL1\d` | 16 | 8 | `DAN` (all) | dopamine 16 | dopamine 16 |
| PPL2 DANs | `^PPL2\d` | 8 | 4 | `DAN` | dopamine 6, unclear 2 | dopamine 6 |
| DPM | `^DPM$` | 2 | 1 | none | dopamine (predicted) | **none** |
| APL | `^APL$` | 2 | 1 | none | GABA | GABA |
| octopaminergic | `^OA-` | 43 | 18 | mostly none | octopamine 33, unclear 6, ACh 4 | octopamine 33 |

Across the whole model, 392 neurons have consensus dopamine: PAM 316, PPL 22, CB.FB 18,
PPM 12, and others.

**Ambiguities recorded:**

* **DPM.** MaleCNS predicts dopamine, with no ground truth. The literature describes DPM as
  serotonergic / GABAergic (amnesiac neuropeptide). DPM is **excluded**.
* **PPM1204.** The type label is a compound (`PPM1204,PS139`) and 4 PPM neurons are predicted
  glutamate. PPM is not used.
* **KC transmitter.** KC ACh is prediction-only (no ground truth).
* **FlyBrain sign rule.** The rule treats dopamine and octopamine as **excitatory fast
  synapses** (sign +1). The model has no neuromodulatory dynamics.

## 2. Mushroom-body connectivity (raw synapse counts)

| block | edges | synapses |
|---|---|---|
| KC → MBON | 61,210 | 463,640 |
| KC → KC | 642,933 | 1,153,845 |
| PAM → KC | 110,450 | 173,789 |
| PPL1 → KC | 17,149 | 49,256 |
| PAM → MBON | 2,693 | 28,546 |
| PPL1 → MBON | 430 | 11,070 |
| KC → PAM / PPL1 | 124,837 / 20,405 | 204,282 / 77,087 |
| APL → KC | 4,633 | 196,200 |
| MBON → PAM / PPL1 (feedback) | 2,119 / 419 | 6,585 / 4,435 |

The DAN → MBON matrix recovers the known compartment pairing without labels. The strongest
pairs:

| DAN → MBON | synapses |
|---|---|
| PAM06 → MBON03 | 3,296 |
| PAM10 → MBON06 | 3,274 |
| **PPL101 → MBON11** | **2,311** |
| PAM08 → MBON05 | 2,197 |
| PAM11 → MBON07 | 1,918 |
| PAM12 → MBON09 | 1,804 |
| **PAM01 → MBON01** | **1,597** |
| PAM04 → MBON02 | 1,168 |

**KC input composition** (synapses onto all KCs):

| source | synapses |
|---|---|
| KC | 1.15 M |
| central-brain intrinsic (mostly olfactory projection neurons) | 451 k |
| APL | 196 k |
| PAM | 174 k |
| PPL1 | 49 k |
| DPM | 33 k |
| **visual_projection** | **11 k** |

**Visual input is concentrated in two KC subtypes.** Direct visual-projection input is
7.4 % of KCg-d input (206 cells) and 2.5 % of KCab-p input. It is ≈ 0 for KCab, KCg-m and
the α′β′ KCs.

The top visual projection types into KCg-d (FlyBrain W) are aMe12, MTe30, LTe25, MTe32 and
aMe20. These are medulla / lobula tangential projection types.

## 3. Proposed minimal plastic circuit

**Sensory layer: KCg-d** (206 neurons, 107 R / 99 L, ACh). This is the only substantial KC
population with direct visual input, so it is the only KC population a Pokémon screen can
plausibly drive.

**Two γ-lobe compartments.** Each has a dopaminergic gate and an output neuron, identified
without labels by DAN → MBON and DAN → KC synapses:

| channel | gating DAN (n) | DAN → KCg-d synapses | DAN → MBON synapses | output MBON (n) | MBON NT (ground truth) | KCg-d → MBON synapses |
|---|---|---|---|---|---|---|
| aversive-candidate (γ1pedc) | **PPL101** (2; L, R; dopamine GT, confidence 0.84) | 634 | 2,311 → MBON11 | **MBON11** (2; L, R) | GABA | 3,273 |
| appetitive-candidate (γ5β′2a) | **PAM01** (44; 23 L / 21 R; dopamine GT, confidence ≥ 0.85) | 3,254 | 1,597 → MBON01 | **MBON01** (2; L, R) | glutamate | 3,720 |

**Biological rationale** (literature, to be cited in the report):

* PPL1-γ1pedc is the canonical aversive-reinforcement DAN; its MBON is MBON-γ1pedc>α/β
  (MBON11 in this naming).
* PAM-γ5 is associated with appetitive / relief reinforcement; its MBON is MBON-γ5β′2a
  (MBON01).
* Dopamine paired with KC activity is reported to **depress** co-active KC → MBON synapses in
  the DAN's own compartment.

The direction of plasticity is **not hard-coded**. The Phase 5 rule will be preregistered with
a signed compartment coefficient, with depression as the literature-supported default.

### Modifiable edge set

* **Edges:** exactly the existing FlyBrain edges KCg-d → MBON11 (**206**) and KCg-d → MBON01
  (**212**), **418 edges** in total.
* **Everything else is frozen:** the other 25,582,520 connections.
* **Deterministic SHA-256** of the sorted (pre bodyId, post bodyId) list:
  **`65dbc6264c04a2fa…`**. The full list with baseline weights is in
  `proposed-plastic-edges.json`.
* **Baseline FlyBrain weights:** 4.1e−5 to 2.1e−3. They are input-normalised; each MBON's
  total |input| is ~1.

**Gating populations for the reinforcement bridge:** PPL101 (aversive channel) and PAM01
(appetitive channel).

## 4. Reachability in the simulated network (FlyBrain W, hop counts)

| path | minimum hops |
|---|---|
| T4 → KCg-d-feeding visual projection neurons | 2 |
| T4 → KCg-d | 2 |
| KCg-d → PPL101 | 1 |
| KCg-d → PAM01 | 1 |
| MBON11 → controller DNs (DNa02, DNg100, MDN, DNp01) | 2 |
| MBON01 → controller DNs | 2 |

Only MBON26, 27, 30, 31, 32 and 35 contact controller DNs directly.

## 5. Risks and missing / ambiguous information

1. **Visual drive of KCg-d is unverified.** Pokémon input enters FlyBrain only through the
   frozen T4 injection. E24 showed that T4 injection barely changes DN output. A 2-hop path
   does not establish that game frames produce KCg-d spikes. **This must be measured first in
   Phase 7 setup.** If KCg-d activity does not differ between two controlled visual states,
   conditioning cannot be tested honestly, and E26 cannot exceed PARTIAL.
2. **Effect size.** The plastic weights are tiny fractions of MBON input. Even their full
   removal may change MBON firing only slightly. Phase 6 must quantify the maximum achievable
   MBON effect at the weight bounds.
3. **Dopamine semantics.** In FlyBrain, DAN spikes act as fast excitatory synapses onto KCs
   and MBONs. Neuromodulation will be an added mechanism that reads DAN spikes. Direct
   excitation is a confound that group B measures.
4. **Reinforcement events.** No existing save state contains HP, EXP or battle events. A
   controlled battle state is needed, or a different reliably detectable event. Candidate
   RAM fields (party HP, EXP, battle outcome) are **not yet verified** and will be verified
   in Phase 3.
5. **Unused inputs.** KC transmitter and KC ↔ KC connectivity are taken as given. APL feedback
   inhibition and MBON → DAN feedback are left in the frozen network, not the plastic set.
6. **Scope of claims.** Hemisphere-specific compartments are pooled across L and R in this
   audit. Every claim is about the modelled network under an engineered interface, not about
   fly experience.

## 6. Decision

The required populations can be identified confidently:

* KCs, MBONs, PAM and PPL1 DANs each carry a class label, a consensus transmitter and, for
  DANs and most MBONs, ground truth;
* compartment pairing is recovered from synapse counts.

**Proceed to Phase 3 (reinforcement bridge) and Phase 4–6 (eligibility, plasticity,
synthetic validation)** on the 418-edge set above.

One precondition must be met before any Pokémon conditioning claim: KCg-d must respond
differentially to controlled game frames through the frozen visual pathway (risk 1).
