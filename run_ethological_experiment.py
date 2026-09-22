"""Experiment 16 Stage 0: drive ethological motion/looming/flow stimuli through the
frozen Experiment-5 biological retina and record subtype-resolved spike counts.

Gain and dynamics are the frozen Experiment-5 / FlyBrain defaults (no gain sweep).
No game state, RAM, reward, or action label is used.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from flymon import ethology
from flymon.dataset import digest
from flymon.retina import bilinear_sample, full_eye_drive, linear_luminance, load_mapping

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "redfly-benchmark" / "data"
EXP5 = ROOT / "results" / "experiment-05-retinotopic"
RESULTS = ROOT / "results" / "experiment-16-ethological-visual-motor"
CAL_SEEDS = tuple(range(1101, 1121))
HELDOUT_SEEDS = tuple(range(1121, 1141))
SEEDS = CAL_SEEDS + HELDOUT_SEEDS
STEPS_PER_FRAME = 20
ADAPTATION_STEPS = 0  # matches Experiment 5/15 (reset then stimulate)


def build_groups(brain):
    """Disjoint recorded populations, subtype- and side-resolved where relevant."""
    ct = brain.cell_type.astype(str)
    side = brain.side.astype(str)
    groups = {}
    groups["R1_6"] = np.flatnonzero(ct == "R1-6")
    groups["lamina"] = np.flatnonzero(np.isin(ct, ["L1", "L2", "L3", "L5"]))

    def by_side(mask, base):
        for s in ("L", "R"):
            idx = np.flatnonzero(mask & (side == s))
            if len(idx):
                groups[f"{base}_{s}"] = idx

    for sub in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"):
        by_side(ct == sub, sub)
    by_side(ct == "LPLC2", "LPLC2")
    groups["LPLC1"] = np.flatnonzero(ct == "LPLC1")
    groups["LC4"] = np.flatnonzero(ct == "LC4")
    groups["VS"] = np.flatnonzero(np.char.startswith(ct, "VS"))
    for dn in ("DNp01", "DNp03", "DNp04", "DNp11", "DNg13", "DNa02"):
        by_side(ct == dn, dn)
    groups["DNg100"] = np.flatnonzero(ct == "DNg100")
    groups["MDN"] = np.flatnonzero(ct == "MDN")
    return groups


def group_id_array(n, groups):
    names = list(groups)
    gid = np.full(n, -1, np.int32)
    for k, name in enumerate(names):
        if np.any(gid[groups[name]] != -1):
            raise RuntimeError(f"group overlap at {name}")
        gid[groups[name]] = k
    return names, gid


def encode(frames_rgba, receptors, visual_count):
    uv = np.asarray([(r.u, r.v) for r in receptors], np.float32)
    previous = np.full(len(receptors), 0.9, np.float32)
    drives = []
    for frame in frames_rgba:
        sample = bilinear_sample(linear_luminance(frame), uv)
        drives.append(full_eye_drive(visual_count, receptors, sample, previous, temporal=True))
        previous = sample
    return drives


def run_trial(brain, seed, drives, gid, G, frame_count):
    brain.reset(seed=seed)
    for _ in range(ADAPTATION_STEPS):
        brain.step(eye_drive=None)
    totals = np.zeros(G, np.int64)
    for fi in range(frame_count):
        drive = drives[fi]
        for _ in range(STEPS_PER_FRAME):
            fired = brain.step(eye_drive=drive)
            g = gid[fired]
            g = g[g >= 0]
            totals += np.bincount(g, minlength=G).astype(np.int64)
    return totals


def main():
    from flybrain import FlyBrain
    RESULTS.mkdir(parents=True, exist_ok=True)
    receptors, retina_diag = load_mapping(EXP5 / "retina-mapping.json")
    brain = FlyBrain(data=DATA, device="cuda", batch=1)
    if brain.n != 166700 or brain._W.nnz != 25582938:
        raise RuntimeError("unexpected MaleCNS graph")
    visual_count = len(brain.visual)
    meta = np.load(DATA / "brain.npz")
    ids = meta["ids"]

    groups = build_groups(brain)
    names, gid = group_id_array(brain.n, groups)
    G = len(names)
    group_sizes = {n: int(len(groups[n])) for n in names}

    stimuli = ethology.catalogue()
    stim_names = list(stimuli)
    frame_count = ethology.FRAMES
    print(f"{len(stim_names)} stimuli x {len(SEEDS)} seeds; {G} recorded populations", flush=True)

    drives_by_stim = {name: encode(seq, receptors, visual_count) for name, seq in stimuli.items()}
    responses = np.zeros((len(stim_names), len(SEEDS), G), np.int64)

    started = time.monotonic()
    for si, name in enumerate(stim_names):
        drives = drives_by_stim[name]
        for sj, seed in enumerate(SEEDS):
            responses[si, sj] = run_trial(brain, seed, drives, gid, G, frame_count)
        if (si + 1) % 6 == 0 or si == len(stim_names) - 1:
            print(f"  [{si + 1:>2}/{len(stim_names)}] {name:<20} elapsed {time.monotonic()-started:6.1f}s", flush=True)

    np.savez_compressed(RESULTS / "responses.npz",
                        stim_names=np.asarray(stim_names), seeds=np.asarray(SEEDS),
                        cal_seeds=np.asarray(CAL_SEEDS), heldout_seeds=np.asarray(HELDOUT_SEEDS),
                        group_names=np.asarray(names),
                        group_sizes=np.asarray([group_sizes[n] for n in names]),
                        responses=responses)

    stim_manifest = {name: dict(mean_luminance=ethology.mean_luminance(seq),
                                frames=int(len(seq)),
                                sequence_sha256=digest(seq.tobytes()))
                     for name, seq in stimuli.items()}
    (RESULTS / "stimulus-manifest.json").write_text(json.dumps(dict(
        frames=frame_count, steps_per_frame=STEPS_PER_FRAME, adaptation_steps=ADAPTATION_STEPS,
        shuffle_order=ethology.shuffle_order().tolist(), stimuli=stim_manifest), indent=2) + "\n")

    pop_manifest = {n: dict(size=group_sizes[n],
                            flywire_ids=[int(x) for x in ids[groups[n]]],
                            cell_types=sorted(set(brain.cell_type[groups[n]].astype(str).tolist())),
                            sides=sorted(set(brain.side[groups[n]].astype(str).tolist())))
                    for n in names}
    (RESULTS / "population-manifest.json").write_text(json.dumps(pop_manifest, indent=2) + "\n")

    (RESULTS / "run-summary.json").write_text(json.dumps(dict(
        model="flybrain 0.1.0 MaleCNS v1.0", device="cuda", eye_gain=float(brain.eye_gain),
        gain_note="frozen Experiment-5 eye-drive / FlyBrain defaults; no gain sweep",
        steps=frame_count * STEPS_PER_FRAME, calibration_seeds=list(CAL_SEEDS),
        heldout_seeds=list(HELDOUT_SEEDS), n_stimuli=len(stim_names), n_populations=G,
        retina_mapping_diagnostics=retina_diag, wall_seconds=time.monotonic() - started), indent=2) + "\n")
    print(f"saved responses.npz and manifests ({time.monotonic()-started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
