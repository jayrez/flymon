"""Experiment 15 analysis: held-out screen information at each biological stage,
and for anatomically selected (label-free) DN / visual-projection subsets, versus
the Experiment-5 broad populations and the Experiment-6 engineered pathway.

No control, held-out image, or held-out seed ever enters feature ranking. The
anatomically selected subsets come from run_pathway_audit.py (connectivity only).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from flymon.dataset import CLASSES, digest, write_json
from flymon.generalization import (PreparedFold, split_plan, prepare_plan, nested, unselected,
                                   record, bootstrap, permutation_labels, instance_blocks,
                                   nested_fixed_subsets)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "experiment-15-biological-pathway"
CAPTURES = ROOT / "captures" / "experiment-15"
EXP6 = ROOT / "results" / "experiment-06-generalization"
STAGES = ("r1_6", "lamina", "t4", "t5", "visual_projection", "descending")
NCLASS = len(CLASSES)


def bar_svg(path, labels, values, title, ref=None):
    from xml.sax.saxutils import escape
    width = 760; height = 70 + len(labels) * 30; scale = max(max(values), 1e-9)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>', f'<text x="15" y="26" font-size="15">{escape(title)}</text>']
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 48 + i * 30
        parts += [f'<text x="205" y="{y+15}" text-anchor="end" font-size="12">{escape(str(label))}</text>',
                  f'<rect x="215" y="{y}" width="{470*value/scale:.1f}" height="20" fill="#275e85"/>',
                  f'<text x="695" y="{y+15}" font-size="12">{value:.3f}</text>']
    if ref is not None:
        x = 215 + 470 * ref / scale
        parts.append(f'<line x1="{x:.1f}" y1="40" x2="{x:.1f}" y2="{height-15}" stroke="#c0392b" stroke-dasharray="4"/>')
        parts.append(f'<text x="{x:.1f}" y="38" font-size="10" fill="#c0392b" text-anchor="middle">chance</text>')
    path.write_text('\n'.join(parts + ['</svg>']))


def frozen_eval(x, y, plan, frozen_idx, prepared=None):
    """Leak-free nearest-centroid on a fixed feature subset: training centroids only."""
    frozen_idx = np.asarray(frozen_idx, int)
    p = np.full(x.shape[:2], -1, int)
    if prepared is None:
        outers = [PreparedFold.make(x, f) for f in plan]
    else:  # prepare_plan yields (outer, inners) tuples
        outers = [pf[0] if isinstance(pf, tuple) else pf for pf in prepared]
    for f, outer in zip(plan, outers):
        means, _ = outer.fit(y, NCLASS)
        fp = np.argmin(cdist(outer.test[:, frozen_idx], means[:, frozen_idx], 'sqeuclidean'), axis=1)
        p[np.ix_(np.array(f['test_i']), np.array(f['test_s']))] = fp.reshape(len(f['test_i']), len(f['test_s']))
    assert np.all(p >= 0)
    return record(p, y, CLASSES), p


def control_predictions(x_train, y, plan, control_vectors, frozen_idx):
    """Classify control conditions (never trained on) with frozen outer models."""
    frozen_idx = np.asarray(frozen_idx, int)
    out = {}
    for name, vec in control_vectors.items():  # vec: (seeds, feat)
        votes = np.zeros(NCLASS, int)
        for f in plan:
            outer = PreparedFold.make(x_train, f)
            means, _ = outer.fit(y, NCLASS)
            pred = np.argmin(cdist(vec[:, frozen_idx], means[:, frozen_idx], 'sqeuclidean'), axis=1)
            np.add.at(votes, pred, 1)
        out[name] = {CLASSES[c]: int(votes[c]) for c in range(NCLASS)}
    return out


def distance_decomp(x, y):
    n, s, _ = x.shape
    unequal = ~np.eye(s, dtype=bool)
    noise = []; within = []; between = []
    for i in range(n):
        for j in range(i, n):
            v = float(cdist(x[i], x[j])[unequal].mean())
            (noise if i == j else within if y[i] == y[j] else between).append(v)
    return dict(same_instance_new_seed=float(np.mean(noise)),
                new_instance_same_class=float(np.mean(within)),
                different_class=float(np.mean(between)),
                margin=float(np.mean(between) - np.mean(within)))


def load_engineered(primary_ids, seeds):
    """Experiment-6 engineered-pathway DN counts on the identical instances/seeds."""
    raw = json.loads((EXP6 / "trials.json").read_text())["trials"]
    x = []
    for iid in primary_ids:
        byseed = {t['seed']: t for t in raw[iid]}
        x.append([byseed[s]['dn_counts'] for s in seeds])
    return np.asarray(x, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args()
    started = time.monotonic()
    CAPTURES.mkdir(parents=True, exist_ok=True)

    npz = np.load(RESULTS / "stage-vectors.npz", allow_pickle=False)
    cond = npz["conditions"].astype(str)
    seeds = npz["seeds"].tolist()
    primary_ids = npz["primary_ids"].astype(str).tolist()
    audit = json.loads((RESULTS / "connectivity-audit.json").read_text())

    labels_all = json.loads((EXP6 / "dataset-manifest.json").read_text())
    id_label = {r['instance_id']: r['class_label'] for r in labels_all['instances']}
    prim_pos = {c: k for k, c in enumerate(cond)}
    primary_cols = np.array([prim_pos[i] for i in primary_ids])
    y = np.array([CLASSES.index(id_label[i]) for i in primary_ids])

    dn_brain = npz["idx_descending"]
    dn_slot = {int(b): k for k, b in enumerate(dn_brain)}
    vp_brain = npz["idx_visual_projection"]
    vp_slot = {int(b): k for k, b in enumerate(vp_brain)}
    dn_types = npz["dn_types"].astype(str)

    # (instance, seed, feature) arrays per stage, primary instances only
    stage_x = {k: npz[f"stage_{k}"][primary_cols].astype(np.float64) for k in STAGES}
    x_dn = stage_x["descending"]

    plan = split_plan(y, nseed=len(seeds))
    # split_plan guarantees disjoint instances and seed blocks; verify explicitly.
    for f in plan:
        assert not (set(f['train_i']) & set(f['test_i']))
        assert not (set(f['train_s']) & set(f['test_s']))
        for inner in f['inner']:
            assert not (set(inner['train_i']) & set(inner['test_i']))
            assert not (set(inner['train_s']) & set(inner['test_s']))
    audit_checks = {"instance_overlap": False, "seed_overlap": False,
                    "inner_containment": True, "note": "verified disjoint by split_plan"}

    # ---- Stage-wise held-out accuracy (all-feature nearest centroid) ----
    stage_results = {}
    for k in STAGES:
        p = unselected(stage_x[k], y, plan)
        stage_results[k] = dict(record(p, y, CLASSES), **{"distance": distance_decomp(stage_x[k], y)})
        print(f"stage {k:<18} held-out acc {stage_results[k]['accuracy']:.3f}", flush=True)

    # ---- DN-level biological ----
    prepared_dn = prepare_plan(x_dn, plan)
    all_dn = record(unselected(x_dn, y, plan), y, CLASSES)

    # anatomically selected DN subsets (label-free frozen)
    sel_idx = {name: np.array([dn_slot[b] for b in idx], int)
               for name, idx in audit["selected_dn_indices"].items()}
    # Secondary descriptive: each fixed anatomical subset evaluated on its own.
    bio_frozen = {}
    for name, idx in sel_idx.items():
        rec, _ = frozen_eval(x_dn, y, plan, idx, prepared=prepared_dn)
        bio_frozen[name] = rec
        print(f"biological {name:<14} held-out acc {rec['accuracy']:.3f} (secondary/fixed)", flush=True)

    # PRIMARY endpoint: nested inner-CV choice of K in {10,20,50}, no test leakage.
    subset_order = ["bio_dn_top10", "bio_dn_top20", "bio_dn_top50"]
    primary_subsets = {n: sel_idx[n] for n in subset_order}
    primary_pred, primary_choices = nested_fixed_subsets(
        y, plan, prepared_dn, primary_subsets, subset_order, x_dn.shape[:2], NCLASS)
    primary_rec = record(primary_pred, y, CLASSES)
    chosen_counts = {n: sum(c['chosen'] == n for c in primary_choices) for n in subset_order}
    print(f"PRIMARY nested anatomical-DN held-out acc {primary_rec['accuracy']:.3f} "
          f"(chosen K per fold: {chosen_counts})", flush=True)

    # nested training-selected DN (leak-free nesting)
    nested_res = nested(x_dn, y, plan)
    nested_rec = record(nested_res['predictions'], y, CLASSES)

    # P20 (== frozen Experiment-4/6 informative_dn_top20) and named controller cells
    p20_idx = np.array([dn_slot[int(nr['brain_index'])] for nr in audit["p20"]], int)
    p20_rec, _ = frozen_eval(x_dn, y, plan, p20_idx, prepared=prepared_dn)

    def type_idx(name):
        return np.array([dn_slot[int(b)] for b in dn_brain[dn_types == name]], int)
    named = {}
    for nm in ("DNa02", "DNg100", "DNg13", "DNp01"):
        idx = type_idx(nm)
        named[nm], _ = frozen_eval(x_dn, y, plan, idx, prepared=prepared_dn)
        print(f"named {nm:<7} (n={len(idx)}) held-out acc {named[nm]['accuracy']:.3f}", flush=True)

    # ---- Visual-projection subsets ----
    x_vp = stage_x["visual_projection"]
    prepared_vp = prepare_plan(x_vp, plan)
    vp_all = record(unselected(x_vp, y, plan), y, CLASSES)
    vp_frozen = {}
    # map selected VP (by anatomical rank) using vp_ranked brain indices
    vp_ranked = audit["vp_ranked"]
    for K in (50, 100, 250):
        idx = np.array([vp_slot[int(vp_ranked[i]["brain_index"])] for i in range(K)], int)
        rec, _ = frozen_eval(x_vp, y, plan, idx, prepared=prepared_vp)
        vp_frozen[f"bio_vp_top{K}"] = rec
    # T5-reader VP (ranked by direct T5 inputs)
    t5_sorted = sorted(vp_ranked, key=lambda r: -r["t5_inputs"])
    for K in (50, 100):
        idx = np.array([vp_slot[int(t5_sorted[i]["brain_index"])] for i in range(K)], int)
        rec, _ = frozen_eval(x_vp, y, plan, idx, prepared=prepared_vp)
        vp_frozen[f"t5_reader_vp_top{K}"] = rec
    for name, rec in vp_frozen.items():
        print(f"VP subset {name:<18} held-out acc {rec['accuracy']:.3f}", flush=True)

    # ---- Engineered pathway A (Experiment 6 DN counts, identical plan) ----
    x_eng = load_engineered(primary_ids, seeds)
    prepared_eng = prepare_plan(x_eng, plan)
    eng_all = record(unselected(x_eng, y, plan), y, CLASSES)
    eng_nested = record(nested(x_eng, y, plan)['predictions'], y, CLASSES)
    eng_p20, _ = frozen_eval(x_eng, y, plan, p20_idx, prepared=prepared_eng)
    eng_bio20, _ = frozen_eval(x_eng, y, plan, sel_idx["bio_dn_top20"], prepared=prepared_eng)
    print(f"engineered all-DN {eng_all['accuracy']:.3f} nested {eng_nested['accuracy']:.3f} "
          f"P20 {eng_p20['accuracy']:.3f} bio20 {eng_bio20['accuracy']:.3f}", flush=True)

    # ---- Controls (never trained on) ----
    def cond_vec(stage, name):
        return npz[f"stage_{stage}"][prim_pos[name]].astype(np.float64)
    controls = {}
    for name in cond:
        if name == "baseline_none" or name.startswith("shuffled-") or name == "uniform_gray":
            controls[name] = cond_vec("descending", name)
    # Control classification uses the modal inner-CV-chosen subset (label-free choice).
    control_subset_name = max(chosen_counts, key=lambda n: (chosen_counts[n], -subset_order.index(n)))
    control_dn = control_predictions(x_dn, y, plan, controls, sel_idx[control_subset_name])
    # also shuffle retention on R1-6 (does destroyed structure still separate?)
    shuffle_r16 = {n: cond_vec("r1_6", n) for n in controls if n.startswith("shuffled-")}
    control_r16 = control_predictions(stage_x["r1_6"], y, plan,
                                      shuffle_r16, np.arange(stage_x["r1_6"].shape[-1]))

    # ---- Permutation test on the PRIMARY (nested) biological DN endpoint ----
    # The full nested selection is repeated under each permutation, so K selection
    # is part of the null. prepared_dn is label-independent, so this is cheap.
    blocks = instance_blocks(y)
    rng = np.random.default_rng(15062026)
    observed = primary_rec['accuracy']
    null = []
    for _ in range(args.permutations):
        yp = permutation_labels(y, blocks, rng)
        pred_p, _ = nested_fixed_subsets(yp, plan, prepared_dn, primary_subsets,
                                         subset_order, x_dn.shape[:2], NCLASS)
        null.append(float((pred_p == yp[:, None]).mean()))
    null = np.array(null)
    pval = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    boot = bootstrap(primary_pred, y)

    # ---- assemble ----
    out = dict(
        seeds=seeds, primary_ids=primary_ids, chance=1.0 / NCLASS,
        n_permutations=args.permutations,
        stage_accuracy={k: stage_results[k]['accuracy'] for k in STAGES},
        stage_balanced={k: stage_results[k]['balanced_accuracy'] for k in STAGES},
        stage_distance={k: stage_results[k]['distance'] for k in STAGES},
        biological=dict(all_dn=all_dn, nested_selected=nested_rec,
                        anatomical_dn_fixed=bio_frozen, p20=p20_rec, named=named,
                        vp_all=vp_all, vp_subsets=vp_frozen,
                        primary_endpoint=dict(
                            method="nested inner-CV choice of K in {10,20,50}; label-free anatomical subsets",
                            candidates=subset_order, accuracy=observed,
                            balanced=primary_rec['balanced_accuracy'],
                            chosen_K_counts=chosen_counts,
                            per_fold_choices=primary_choices,
                            permutation_p=pval, null_mean=float(null.mean()),
                            null_95=float(np.quantile(null, 0.95)),
                            bootstrap95=boot['interval95'],
                            confusion=primary_rec['confusion'], per_class_recall=primary_rec['per_class_recall'])),
        engineered=dict(all_dn=eng_all, nested_selected=eng_nested, p20=eng_p20,
                        bio_dn_top20=eng_bio20),
        controls=dict(dn_subset=control_dn, r16_shuffle=control_r16,
                      subset_used=control_subset_name),
        nested_choices=[{k: c[k] for k in ('instance_block', 'seed_block', 'population')}
                        for c in nested_res['choices']],
        leakage_checks=audit_checks,
    )
    write_json(RESULTS / "classification.json", out)

    # ---- plots ----
    stage_labels = ["R1-6", "lamina", "T4", "T5", "visual_proj", "all DN"]
    bar_svg(CAPTURES / "stage-accuracy.svg", stage_labels,
            [stage_results[k]['accuracy'] for k in STAGES],
            "Experiment 15 — held-out screen accuracy by biological stage", ref=0.2)
    comp_labels = ["bio all-DN", "bio anat (nested)", "bio P20", "bio nested-sel",
                   "eng all-DN", "eng P20", "eng nested"]
    comp_vals = [all_dn['accuracy'], observed, p20_rec['accuracy'], nested_rec['accuracy'],
                 eng_all['accuracy'], eng_p20['accuracy'], eng_nested['accuracy']]
    bar_svg(CAPTURES / "pathway-comparison.svg", comp_labels, comp_vals,
            "Experiment 15 — biological vs engineered DN pathways", ref=0.2)
    vp_labels = ["broad VP"] + list(vp_frozen)
    vp_vals = [vp_all['accuracy']] + [vp_frozen[k]['accuracy'] for k in vp_frozen]
    bar_svg(CAPTURES / "vp-subsets.svg", vp_labels, vp_vals,
            "Experiment 15 — visual-projection subset held-out accuracy", ref=0.2)

    print(f"\nPRIMARY nested anatomical-DN endpoint: {observed:.3f} "
          f"(balanced {primary_rec['balanced_accuracy']:.3f}, p={pval:.4f}, "
          f"boot95 {boot['interval95']}); biological all-DN {all_dn['accuracy']:.3f}; "
          f"engineered all-DN {eng_all['accuracy']:.3f}", flush=True)
    print(f"analysis wall {time.monotonic()-started:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    main()
