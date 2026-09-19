# Experiment 6 capture dataset

Each instance directory contains ten unmodified 160×144 PyBoy RGBA framebuffer
PNGs and metadata describing deterministic capture actions, frame offsets,
pixel hashes, PNG hashes and (where used) the savestate hash. The ROM stays external
and is referenced only via `POKEMON_ROM`; no RAM values are used.

The authoritative primary inclusion list is
`results/experiment-06-generalization/dataset-manifest.json`. It accepts 32
instances: bedroom 5, dialogue 8, menu 5, title 8, intro 6. Capture IDs have gaps
because exact duplicate framebuffers were rejected. A repeated frame within a
single sequence is permitted; a frame shared across distinct instances is not.

`menu-06` is retained as an excluded capture: visual inspection found the menu
had closed. It is neither a primary example nor a control. `ood-options-01` is a
separate options-screen probe and never enters primary training or inference.

The bedroom class follows the canonical `states/bedroom.state` filename; frames
show the room reached by that supplied state and scripted movement. Dialogue is
limited to Oak's opening text, menu to the in-room pause layout, intro to opening
animation poses, and title to naturally changing Pokémon sprites. This is a
test of these narrow visual families, not arbitrary semantic game states.

Images are immutable: the loader checks both file and pixel hashes, and refuses
cross-instance overlap. Use `capture_visual_instance.py --recipe recipe.json` to
capture a new ID rather than overwrite an existing one. Example recipe:

```json
{
  "class_label": "bedroom",
  "instance_id": "bedroom-new",
  "savestate": "states/bedroom.state",
  "actions": [{"wait": 1}, {"button": "left", "held": 8, "release": 40}]
}
```

Do not add new instances to the frozen Experiment 6 manifest. The published
protocol and manifest hashes bind the actual experiment to the reviewed dataset.
