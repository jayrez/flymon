"""Experiment 15 Stage 4: drive the Experiment-6 dataset through the Experiment-5
biological retina and record per-stage spike counts.

The engineered LC10a/LPLC2 injection is NOT used here; the image enters only
through the inferred R1-R6 eye-drive. Spike counts are stored for R1-R6, lamina,
T4, T5, the visual-projection superclass and all descending neurons, so the
offline analysis can measure held-out screen information at each biological stage
and for the anatomically selected subsets from run_pathway_audit.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from flymon.dataset import CLASSES, load_dataset, digest
from flymon.retina import bilinear_sample, full_eye_drive, linear_luminance, load_mapping

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "redfly-benchmark" / "data"
EXP5 = ROOT / "results" / "experiment-05-retinotopic"
EXP6 = ROOT / "results" / "experiment-06-generalization"
RESULTS = ROOT / "results" / "experiment-15-biological-pathway"
SEEDS = tuple(range(201, 221))
FRAME_COUNT = 10
STEPS_PER_FRAME = 20
SHUFFLE_RNG = 606

STAGES = ("r1_6", "lamina", "t4", "t5", "visual_projection", "descending")


def stage_indices(brain):
    ct = brain.cell_type.astype(str)
    sc = brain.superclass.astype(str)
    return {
        "r1_6": np.flatnonzero(ct == "R1-6"),
        "lamina": np.flatnonzero(np.isin(ct, ["L1", "L2", "L3", "L5"])),
        "t4": np.flatnonzero(np.char.startswith(ct, "T4")),
        "t5": np.flatnonzero(np.char.startswith(ct, "T5")),
        "visual_projection": np.flatnonzero(sc == "visual_projection"),
        "descending": np.flatnonzero(sc == "descending_neuron"),
    }


def encode_bio(frames, records, visual_count):
    """Ten temporal eye-drive vectors, exactly the Experiment-5 biological encoder."""
    uv = np.asarray([(r.u, r.v) for r in records], np.float32)
    previous = np.full(len(records), 0.9, np.float32)
    drives = []
    for frame in frames:
        sample = bilinear_sample(linear_luminance(frame), uv)
        drives.append(full_eye_drive(visual_count, records, sample, previous, temporal=True))
        previous = sample
    return drives


def run_trial(brain, seed, drives, slots, sizes):
    brain.reset(seed=seed)
    totals = {k: np.zeros(sizes[k], np.int32) for k in slots}
    step_s = 0.0
    for fi in range(FRAME_COUNT):
        drive = None if drives is None else drives[fi]
        for _ in range(STEPS_PER_FRAME):
            t = time.perf_counter()
            fired = brain.step(eye_drive=drive)
            step_s += time.perf_counter() - t
            for name, slot in slots.items():
                active = slot[fired]
                active = active[active >= 0]
                totals[name] += np.bincount(active, minlength=sizes[name]).astype(np.int32)
    return totals, step_s


def build_conditions(records, sequences, primary_ids, visual_count, encode):
    conditions = {}
    by_id = {r["instance_id"]: s for r, s in zip(records, sequences)}
    for iid in primary_ids:
        conditions[iid] = encode(by_id[iid])
    conditions["baseline_none"] = None
    gray = [np.concatenate([np.full((144, 160, 3), 128, np.uint8),
                            np.full((144, 160, 1), 255, np.uint8)], axis=2) for _ in range(FRAME_COUNT)]
    conditions["uniform_gray"] = encode(gray)
    permutation = np.random.default_rng(SHUFFLE_RNG).permutation(144 * 160)
    for label in CLASSES:
        iid = next(i for i in primary_ids
                   if next(r for r in records if r["instance_id"] == i)["class_label"] == label)
        seq = by_id[iid]
        shuffled = seq.reshape(FRAME_COUNT, -1, 4)[:, permutation].reshape(seq.shape)
        conditions[f"shuffled-{iid}"] = encode(shuffled)
    return conditions, permutation


def main():
    from flybrain import FlyBrain
    RESULTS.mkdir(parents=True, exist_ok=True)
    records, sequences = load_dataset()
    manifest = json.loads((EXP6 / "dataset-manifest.json").read_text())
    primary_ids = list(manifest["primary_ids"])
    labels = {r["instance_id"]: r["class_label"] for r in records}

    receptors, retina_diag = load_mapping(EXP5 / "retina-mapping.json")
    brain = FlyBrain(data=DATA, device="cuda", batch=1)
    if brain.n != 166700 or brain._W.nnz != 25582938:
        raise RuntimeError("unexpected MaleCNS graph")
    visual_count = len(brain.visual)

    slots_idx = stage_indices(brain)
    sizes = {k: len(v) for k, v in slots_idx.items()}
    slots = {}
    for name, idx in slots_idx.items():
        slot = np.full(brain.n, -1, np.int32)
        slot[idx] = np.arange(len(idx))
        slots[name] = slot

    encode = lambda frames: encode_bio(frames, receptors, visual_count)
    conditions, permutation = build_conditions(records, sequences, primary_ids, visual_count, encode)
    cond_names = list(conditions)
    print(f"{len(cond_names)} conditions x {len(SEEDS)} seeds; stage sizes {sizes}", flush=True)

    stage_arrays = {k: np.zeros((len(cond_names), len(SEEDS), sizes[k]), np.int32) for k in STAGES}
    brain.reset(seed=0)
    for _ in range(20):
        brain.step(eye_drive=None)

    started = time.monotonic()
    total_step_s = 0.0
    for ci, name in enumerate(cond_names):
        drives = conditions[name]
        for si, seed in enumerate(SEEDS):
            totals, step_s = run_trial(brain, seed, drives, slots, sizes)
            total_step_s += step_s
            for k in STAGES:
                stage_arrays[k][ci, si] = totals[k]
        print(f"  [{ci + 1:>2}/{len(cond_names)}] {name:<22} elapsed {time.monotonic() - started:6.1f}s", flush=True)

    meta = np.load(DATA / "brain.npz")
    ids = meta["ids"]
    np.savez_compressed(
        RESULTS / "stage-vectors.npz",
        conditions=np.asarray(cond_names),
        seeds=np.asarray(SEEDS),
        labels=np.asarray([labels.get(n, "control") for n in cond_names]),
        primary_ids=np.asarray(primary_ids),
        **{f"stage_{k}": v for k, v in stage_arrays.items()},
        **{f"idx_{k}": slots_idx[k] for k in STAGES},
        dn_ids=ids[slots_idx["descending"]],
        dn_types=brain.cell_type[slots_idx["descending"]],
        dn_side=brain.side[slots_idx["descending"]],
        vp_ids=ids[slots_idx["visual_projection"]],
        vp_types=brain.cell_type[slots_idx["visual_projection"]],
        t5_ids=ids[slots_idx["t5"]],
    )
    summary = dict(
        model="flybrain 0.1.0 MaleCNS v1.0", device="cuda", steps=FRAME_COUNT * STEPS_PER_FRAME,
        seeds=list(SEEDS), conditions=cond_names, primary_ids=primary_ids, stage_sizes=sizes,
        encoder="Experiment-5 biological R1-R6 eye-drive, temporal transform",
        retina_mapping_diagnostics=retina_diag,
        shuffle_pixel_permutation_sha256=digest(permutation.tobytes()),
        mean_step_ms=1000 * total_step_s / (len(cond_names) * len(SEEDS) * FRAME_COUNT * STEPS_PER_FRAME),
        wall_seconds=time.monotonic() - started,
    )
    (RESULTS / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved stage-vectors.npz and run-summary.json ({summary['wall_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
