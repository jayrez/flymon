# Experiment 18 — T4/T5 optic-column geometry (anatomy only)

**Geometry gate: PASS.** MaleCNS contains subtype-specific, spatially offset T4/T5
input geometry of exactly the form a delay-and-compare motion detector needs. Nothing
in this document uses a stimulus, a response, or a class label.

## Method

Optic-column coordinates come from released `assignedOlHex1/2` annotations, converted
to the axial cartesian frame `flymon/retina.py` already uses
(`x = h1 − 0.5·h2`, `y = √3/2·h2`). Only directly annotated coordinates are used.

T4/T5 cells themselves carry **no** hex annotation in v1.0 (0 % coverage), and neither
do Tm3 or CT1. That does not block the measurement: the quantity of interest is the
offset vector between the weighted centroids of two *input arms*, which is translation
invariant and so independent of where the postsynaptic cell sits.

Arms are fixed from literature + connectome sign, never from results:

| pathway | fast arm | slow / sign-inverting arm |
|---|---|---|
| T4 (ON) | Mi1 (99.4 % column coverage) | Mi9 (99.2 %), Mi4 (99.2 %) |
| T5 (OFF) | Tm1 (99.4 %), Tm2 (99.5 %) | Tm9 (98.4 %) |

Eligibility: ≥3 column-resolved partners in each arm.

## Result

| subtype | eligible | resolved weight | mean angle | concentration R | mean \|offset\| | L / R cells |
|---|---:|---:|---:|---:|---:|---|
| T4a | 1680/1684 (99.8 %) | 0.420 | 147.2° | 0.587 | 0.391 | 831 / 849 |
| T4b | 1687/1690 (99.8 %) | 0.401 | 326.5° | 0.552 | 0.379 | 841 / 846 |
| T4c | 1773/1778 (99.7 %) | 0.428 | 51.5° | 0.547 | 0.379 | 890 / 883 |
| T4d | 1705/1710 (99.7 %) | 0.418 | 252.4° | 0.530 | 0.408 | 845 / 860 |
| T5a | 1648/1664 (99.0 %) | 0.412 | 144.9° | **0.884** | 0.669 | 811 / 837 |
| T5b | 1707/1715 (99.5 %) | 0.414 | 323.1° | **0.893** | 0.640 | 855 / 852 |
| T5c | 1712/1720 (99.5 %) | 0.445 | 46.5° | **0.881** | 0.606 | 854 / 858 |
| T5d | 1617/1620 (99.8 %) | 0.444 | 251.0° | **0.874** | 0.636 | 809 / 808 |

**Antiparallel pairs:** T4a↔T4b 179.3°, T5a↔T5b 178.2°, T4c↔T4d 159.0°, T5c↔T5d 155.5°.
**Orthogonal axes:** T4 a/b vs c/d 95.7°, T5 a/b vs c/d 98.4°.

This is the canonical four-direction layout — two roughly orthogonal axes, each an
opposed pair — recovered from connectivity alone, with ~1,700 cells per subtype and
balanced left/right representation. T5 orientation is markedly tighter (R ≈ 0.88) than
T4 (R ≈ 0.55).

## Limitation

Only **~42 % of each T4/T5 cell's total synaptic weight** is column-resolvable, because
Tm3 and CT1 (major partners) carry no hex annotation in v1.0. The geometry reported
here is therefore the geometry of the annotated Mi/Tm arms, not of the complete
dendritic input field. The unresolved fraction could sharpen or blur the true offset.

## Note on eye mirroring

The Experiment-5 display maps the left eye directly and mirrors the right eye into the
screen. A single eye-space offset therefore predicts **opposite screen directions** for
left- and right-eye homologs of the same subtype — which is correct for a real fly
whose eyes face opposite ways. Predicted direction is computed per neuron using that
neuron's own side; the subtype-level "predicted direction" vote is descriptive only.
