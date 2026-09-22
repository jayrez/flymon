# Experiment 16 — biological references and expected responses

These references define the *a priori* expected responses used to preregister
Experiment 16's stimulus–population pairings. They are drawn from the established
Drosophila optic-lobe / descending-neuron literature and the MaleCNS connectome.
Network access for live URL retrieval was not used in this run; citations are given
by author/venue so they can be verified. Each expected response is tagged
**[well-established]**, **[connectomically-suggested]**, or **[exploratory]**.
The implementation is not forced to match these expectations — they only fix the
pairings before held-out evaluation.

## T4 / T5 — directional motion, ON vs OFF
- Maisak et al., *Nature* 2013 ("A directional tuning map of Drosophila
  elementary motion detectors"): **T4 responds to moving ON (bright) edges, T5 to
  moving OFF (dark) edges**; each has four subtypes (a/b/c/d) tuned to the four
  cardinal directions (front-to-back, back-to-front, upward, downward). **[well-established]**
- Fisher et al., *Neuron* 2015; Shinomiya et al. 2019 (T4/T5 circuitry): subtype
  a/b = horizontal, c/d = vertical motion; layer identity sets preferred direction.
  Expected here: **T4 subtypes separate ON motion from static; T5 subtypes separate
  OFF motion from static; opposing directions differ within a subtype.** **[well-established]**
- Caveat: the exact a↔direction assignment is display/eye-pose dependent in this
  simulation, so preferred direction per subtype is fixed empirically on
  calibration seeds (1101–1120), not asserted from the layer letter. **[exploratory]**

## LPLC2 / LPLC1 — looming and object motion
- Klapoetke et al., *Nature* 2017 ("Ultra-selective looming detection from radial
  motion opponency"): **LPLC2 is a looming detector**, driven by outward (expanding)
  radial motion, pooling T4/T5 across directions; a key input to the giant fibre.
  Expected: **LPLC2 responds more to looming (expanding dark disc) than to receding
  or static discs.** **[well-established]**
- Klapoetke/Card lab; Städele et al.: **LPLC1** also loom/object-motion related but
  with different opponency. Expected: object/loom sensitivity. **[connectomically-suggested]**
- MaleCNS connectome (this project's Experiment-15 audit): LPLC2/LPLC1 are the
  strongest direct T5 targets among visual-projection cells. **[connectomically-suggested]**

## VS cells — vertical-system optic flow
- Krapp & Hengstenberg, *Nature* 1996; Boergens et al.; Kurtz: **VS cells are
  lobula-plate tangential cells encoding vertical/rotational wide-field optic flow.**
  Expected: **VS responds to global flow, direction-dependent, more to dynamic flow
  than frozen.** **[well-established]**

## Descending neurons
- DNp01 (giant fibre): von Reyn et al., *Nat. Neurosci.* 2014/2017 — **looming-driven
  escape/takeoff command; receives LPLC2/LC4.** Expected: loom > recede/static. **[well-established]**
- DNp02/DNp04/DNp06/DNp11: Card lab, Namiki et al. 2018 — visual-projection-driven
  escape/takeoff cluster; several loom/motion sensitive. Expected: loom/motion
  modulation. **[connectomically-suggested]**
- DNp03: Namiki et al. 2018 — visually driven descending neuron. **[connectomically-suggested]**
- DNa02: Rayshubskiy et al. 2020; Namiki 2018 — **steering command with left/right
  asymmetry tied to turning/optic flow.** Expected: lateral asymmetry to horizontal
  flow. **[well-established]** for steering role; optic-flow drive here **[exploratory]**.
- DNg13: MaleCNS type; steering/visual-motor associated. Expected: possible
  horizontal-flow asymmetry. **[exploratory]**
- DNg100: locomotor command reference; Bidaye/Cande. Not expected to encode scene
  identity; global-motion modulation secondary. **[exploratory]**
- MDN (moonwalker DN): Bidaye et al., *Science* 2014 / 2020 — **backward-walking
  command;** possible modulation by receding/backward-like flow. **[exploratory]**

## flyvis (functional benchmark)
- Lappalainen et al., *Nature* 2024 ("Connectome-constrained networks predict neural
  activity across the fly visual system"), code `flyvis` (turagalab): a deep-mechanistic
  model of the fly visual system with T4/T5 direction selectivity. Used, if the
  environment permits, as an independent functional positive control for whether the
  synthetic stimuli evoke expected T4/T5-like directional structure — not spliced into
  MaleCNS. **[well-established]** model; cross-model neuron identity is not assumed.
