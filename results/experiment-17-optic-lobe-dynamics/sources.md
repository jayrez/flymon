# Experiment 17 — biological references for optic-lobe dynamics

References fixing the mechanistic expectations used to design the Experiment-17
repair candidates. Tags: **[well-established]**, **[connectomically-supported]**,
**[exploratory]**. Live URL retrieval was not used; citations are given by
author/venue for verification.

## Photoreceptor → lamina is sign-inverting

- Hardie, *Nature* 1989; Hardie & Raghu, *Nature* 2001: Drosophila photoreceptors
  (R1–R6) are **graded, non-spiking** neurons that depolarise to light and release
  **histamine**, which opens chloride channels and **hyperpolarises** the large
  monopolar cells L1/L2/L3. Light therefore *inhibits* LMCs. **[well-established]**
- Laughlin & Hardie, *J. Comp. Physiol.* 1978; Järvilehto & Zettler: LMCs respond to
  light with **hyperpolarisation** and to light decrements with depolarisation; they
  sit at a **maintained (tonic) depolarised resting potential** that the histaminergic
  input modulates bidirectionally. **[well-established]**
- MaleCNS confirmation (this experiment's `connectivity-audit.json`): **100 % of
  R1–R6 → L1/L2/L3/L5 weights are negative** (9,636 edges, mean −0.085, sum −819).
  **[connectomically-supported]**

**Consequence for a spiking model.** If R1–R6 spiking only ever *subtracts* current
from lamina neurons that have no maintained depolarisation, the lamina is pinned at
its floor and carries no graded signal. Restoring a tonic depolarising term for LMCs
is the minimal biologically faithful correction, and is the basis of candidate C.

## T4/T5 input circuitry

- Takemura et al., *Nature* 2013 / *eLife* 2017; Shinomiya et al., *eLife* 2019:
  **T4** (ON pathway) is driven by **Mi1** and **Tm3** (excitatory) with **Mi9**,
  **Mi4** and **CT1** providing sign-inverted / inhibitory input; **T5** (OFF pathway)
  is driven by **Tm1, Tm2, Tm4, Tm9** with **CT1** inhibition. **[well-established]**
- MaleCNS confirmation: T4a top inputs Mi1 (+471), Tm3 (+204), Mi9 (−151), CT1 (−93);
  T5a top inputs Tm2 (+260), Tm9 (+242), Tm1 (+190), CT1 (−178), Tm4 (+147).
  **[connectomically-supported]**

## Direction selectivity requires a temporal asymmetry

- Hassenstein & Reichardt 1956 (delay-and-correlate); Barlow & Levick 1965
  (delayed inhibition / null-direction suppression). Direction selectivity arises from
  **comparing a fast input from one column with a temporally delayed input from a
  neighbouring column**. **[well-established]**
- Behnia et al., *Nature* 2014; Arenz et al., *Curr. Biol.* 2017; Gruntman et al.,
  *Nat. Neurosci.* 2018: the T4/T5 arms differ in **temporal filtering** — Mi9/Mi4/CT1
  are **slow / sustained** relative to fast Mi1/Tm3, and this lead–lag is what makes
  T4/T5 directionally tuned. **[well-established]**
- Maisak et al., *Nature* 2013: T4/T5 subtypes a/b/c/d are tuned to the four cardinal
  directions. **[well-established]**

**Consequence for a spiking model.** A leaky-integrate-and-fire network in which every
neuron shares one membrane time constant and every synapse has zero transmission
delay has **no mechanism to produce direction selectivity**, regardless of gain.
Candidates D–G add population-specific slow time constants and/or explicit
presynaptic delays on the connectomically identified inhibitory arm (Mi9, Mi4, CT1).

## flyvis (functional positive control)

- Lappalainen et al., *Nature* 2024, "Connectome-constrained networks predict neural
  activity across the fly visual system"; package `flyvis` (turagalab). A
  connectome-constrained but **task-optimised** model whose T4/T5 units show
  direction selectivity. Used only to test whether the Experiment-16 stimuli are
  capable of evoking directional structure in a model built to have it. Learned
  weights are **not** transplanted into MaleCNS; only response *relationships*
  (preferred vs null, ON vs OFF, dynamic vs frozen) are compared. **[well-established]**
