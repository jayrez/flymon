# Experiment 2 design, recorded before implementation

## Upstream implementation review

- [`flybrain.eyes.FeatureDetectors`](https://github.com/alextitonis/fly.ai/blob/main/flybrain/eyes.py) selects visual-projection cells by `brain.cells(type, side)` and returns `(indices, voltage)` pairs for `FlyBrain.step(inject=...)`. Its `opp=(dx,size)` route drives LC10a steadily (`chase`) and LPLC2 on angular growth (`loom`). LC4 needs an explicit threat score; LPLC1 needs projectile objects. Those are semantic game inputs, not available from our five static screenshots without unsupported labels.
- `FlyBrain.step()` accepts `inject` separately from `eye_drive`; Experiment 2 uses `eye_drive=None` in every condition. The no-vision control also uses an empty `inject`. `reset(seed)` and returned spike indices support the exact Experiment 1 counting procedure.
- `flybrain.reservoir.Trace` provides decaying features and `run` wraps steps. Exact per-neuron firing counts are the controlled endpoint, so Experiment 1's direct spike-index aggregation is reused. No readout is trained.

## Frame adapter and assumptions

The upstream helper accepts object geometry rather than pixels. A deterministic adapter will convert each `(144,160,4)` framebuffer to Rec.709 grayscale, label 8-connected pixels darker than 0.5, and select the largest component with at least 16 pixels. Its horizontal centre becomes `dx = centre_x - 79.5` and its `size` is `sqrt(pixel_area)`. This is a coarse visual-object proxy, not recognition of a Pokémon character or opponent. The midpoint threshold and minimum area are fixed across all screens and will not be tuned to their identities. A pre-run inspection found that the selected regions are often large backgrounds or text boxes, so upstream channel amplitudes may saturate and discard size information; this is an explicit limitation, not a reason to change the fixed mapping after seeing the result.

The same static image is presented for 200 neural steps. Before step 1, a zero-size object at the selected location primes the upstream growth state without stimulating the brain. Its appearance then generates one LPLC2 onset; unchanged geometry drives LC10a afterward. This appearance transient is a **looming-like proxy**, not measured motion toward the fly. LC4 and LPLC1 remain unused because no threat or projectile evidence is present. The baseline has no selected object and no external injection.

## Biological target mapping from MaleCNS metadata

| Upstream channel | Type | Left | Right | Total | Used |
| --- | --- | ---: | ---: | ---: | --- |
| loom | LPLC2 | 94 | 91 | 185 | onset |
| chase | LC10a | 135 | 140 | 275 | steady |
| threat | LC4 | 71 | 55 | 126 | no semantic threat input |
| shot | LPLC1 | 68 | 66 | 134 | no projectile input |

The intended selected target population is 460 unique LPLC2/LC10a neurons; each frame will stimulate only the side assigned by its component centre. Exact per-screen injected sides, amplitudes, and counts will be recorded. These classes have MaleCNS superclass `visual_projection` and are selected by upstream metadata, not body IDs.

## Controlled comparison

The same five screens are reconstructed by Experiment 1's `capture_conditions()`, including `states/bedroom.state`. Seeds `101..112`, 200 steps, CUDA model parameters, `sensory_input=True`, DN selection, baseline structure, spike aggregation, and distance metrics remain fixed. Experiment 1 files will be copied to a versioned directory while their original paths remain available. Debug images will show the component mask, chosen object bounding box, and LPLC2/LC10a projection amounts.
