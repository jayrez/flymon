"""Experiment 17 Gate 0: stimulus-adequacy positive control.

flyvis installs cleanly in an isolated environment but its *pretrained* ensembles
are not obtainable here, and an untrained connectome-initialised network would not
demonstrate direction selectivity, so it cannot serve as a trained positive control
(status recorded in flyvis-benchmark.json).

Instead this runs a textbook Hassenstein-Reichardt (HR) elementary motion detector
directly on the SAME photoreceptor signals the FlyBrain eye-drive is built from
(identical Experiment-5 retina sampling, identical Experiment-16 stimuli). If a
minimal two-input correlator recovers direction from those exact samples, then the
stimuli and the retinal sampling carry direction information, and any failure to
produce T4/T5 direction selectivity lies in the network dynamics, not the input.

HR unit for a horizontally adjacent receptor pair (A left of B):
    HR(t) = A(t-1)*B(t) - B(t-1)*A(t)
summed over pairs and time. Rightward motion reaches A before B and gives a positive
sum; leftward motion gives a negative sum. No learning, no labels, no game state.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flymon import ethology
from flymon.retina import bilinear_sample, linear_luminance, load_mapping

ROOT = Path(__file__).resolve().parent
EXP5 = ROOT / "results" / "experiment-05-retinotopic"
RESULTS = ROOT / "results" / "experiment-17-optic-lobe-dynamics"


def optic_columns(records):
    """Pool receptors per inferred optic column (neural superposition: the R1-R6 of a
    column converge on one cartridge). Returns column->receptor indices, and mean u."""
    cols = {}
    for i, r in enumerate(records):
        cols.setdefault((r.eye_side, int(r.h1), int(r.h2)), []).append(i)
    keys = sorted(cols)
    members = [np.asarray(cols[k], int) for k in keys]
    u = np.array([np.mean([records[i].u for i in m]) for m in members])
    return keys, members, u


def horizontal_pairs(records):
    """Adjacent optic-column pairs along a hex row, oriented left->right on screen.

    The right eye is mirrored into the display, so pair orientation is taken from
    the screen coordinate u, not from the hex index."""
    keys, members, u = optic_columns(records)
    index = {k: n for n, k in enumerate(keys)}
    pairs = []
    for n, (side, h1, h2) in enumerate(keys):
        m = index.get((side, h1 + 1, h2))
        if m is None:
            continue
        a, b = (n, m) if u[n] <= u[m] else (m, n)   # a is left on screen
        pairs.append((a, b))
    return np.asarray(pairs, int), members


def column_samples(samples, members):
    """(frames, receptors) -> (frames, columns) by averaging each column's receptors."""
    return np.stack([samples[:, m].mean(axis=1) for m in members], axis=1)


def hr_response(samples, pairs, members):
    """samples: (frames, receptors) luminance. Returns summed HR correlator output."""
    cs = column_samples(samples, members)
    a = cs[:, pairs[:, 0]]
    b = cs[:, pairs[:, 1]]
    # contrast around the temporal mean so a uniform field contributes nothing
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    return float(np.sum(a[:-1] * b[1:] - b[:-1] * a[1:]))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    records, _ = load_mapping(EXP5 / "retina-mapping.json")
    uv = np.asarray([(r.u, r.v) for r in records], np.float32)
    pairs, members = horizontal_pairs(records)
    cat = ethology.catalogue()

    def samples_for(name):
        return np.stack([bilinear_sample(linear_luminance(f), uv) for f in cat[name]])

    out = {}
    for name in ("off_left", "off_right", "on_left", "on_right", "off_up", "off_down",
                 "flow_left", "flow_right", "gray",
                 "off_left__frozen_first", "off_left__shuffle", "off_left__reverse"):
        out[name] = hr_response(samples_for(name), pairs, members)

    def dsi(a, b):
        return (out[a] - out[b]) / (abs(out[a]) + abs(out[b]) + 1e-9)

    summary = dict(
        n_column_pairs=int(len(pairs)), n_columns=int(len(members)),
        hr_response=out,
        off_horizontal_dsi=dsi("off_left", "off_right"),
        on_horizontal_dsi=dsi("on_left", "on_right"),
        flow_horizontal_dsi=dsi("flow_left", "flow_right"),
        sign_flips_with_direction=dict(
            off=bool(np.sign(out["off_left"]) != np.sign(out["off_right"])),
            on=bool(np.sign(out["on_left"]) != np.sign(out["on_right"])),
            flow=bool(np.sign(out["flow_left"]) != np.sign(out["flow_right"]))),
        vertical_motion_horizontal_hr=dict(off_up=out["off_up"], off_down=out["off_down"]),
        controls=dict(gray=out["gray"], frozen=out["off_left__frozen_first"],
                      shuffle=out["off_left__shuffle"], reverse=out["off_left__reverse"]),
        frozen_suppression=1.0 - abs(out["off_left__frozen_first"]) / (abs(out["off_left"]) + 1e-9),
        shuffle_suppression=1.0 - abs(out["off_left__shuffle"]) / (abs(out["off_left"]) + 1e-9),
        reverse_sign_flip=bool(np.sign(out["off_left__reverse"]) != np.sign(out["off_left"])),
        method="Hassenstein-Reichardt correlator on Experiment-5 retina samples of the "
               "unchanged Experiment-16 stimuli; no learning, no labels",
    )
    (RESULTS / "positive-control.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"optic columns: {len(members)}  horizontal column pairs: {len(pairs)}")
    for k in ("off_left", "off_right", "on_left", "on_right", "flow_left", "flow_right",
              "gray", "off_left__frozen_first", "off_left__shuffle", "off_left__reverse"):
        print(f"  HR {k:<26} {out[k]:+12.2f}")
    print(f"\nOFF horizontal DSI  {summary['off_horizontal_dsi']:+.4f}  "
          f"(sign flips: {summary['sign_flips_with_direction']['off']})")
    print(f"ON  horizontal DSI  {summary['on_horizontal_dsi']:+.4f}  "
          f"(sign flips: {summary['sign_flips_with_direction']['on']})")
    print(f"flow horizontal DSI {summary['flow_horizontal_dsi']:+.4f}")
    print(f"frozen suppression  {summary['frozen_suppression']:.3f}  "
          f"shuffle suppression {summary['shuffle_suppression']:.3f}  "
          f"reverse flips sign: {summary['reverse_sign_flip']}")


if __name__ == "__main__":
    main()
