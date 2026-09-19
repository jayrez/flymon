"""Experiment 6: freeze pixel dataset/protocol, then run the unchanged E3 pathway."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
import time
import numpy as np
from scipy.spatial.distance import cdist
from flymon.dataset import (ROOT, RESULTS, CAPTURES, CLASSES, SEEDS, SIZES,
                            load_dataset, encode, regression, digest, write_json)
from flymon.spatiotemporal import SpatiotemporalConfig, SpatialProjection, spatial_grid


def matrix_svg(path, matrix, labels, title):
    from xml.sax.saxutils import escape
    a=np.asarray(matrix);n=len(a);cell=22;left=130;top=145
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{left+n*cell+30}" height="{top+n*cell+30}">',
           '<rect width="100%" height="100%" fill="white"/>',f'<text x="10" y="20" font-size="13">{escape(title)}</text>']
    scale=max(float(a.max()),1e-9)
    for i,label in enumerate(labels):
        parts.append(f'<text x="{left-5}" y="{top+i*cell+15}" text-anchor="end" font-size="10">{escape(label)}</text>')
        parts.append(f'<text transform="translate({left+i*cell+15},{top-5}) rotate(-65)" font-size="10">{escape(label)}</text>')
        for j in range(n):
            q=float(a[i,j])/scale;r=int(245-215*q);g=int(250-155*q);b=int(255-65*q)
            parts.append(f'<rect x="{left+j*cell}" y="{top+i*cell}" width="{cell}" height="{cell}" fill="rgb({r},{g},{b})"><title>{a[i,j]:.4f}</title></rect>')
    parts.append('</svg>');path.write_text('\n'.join(parts))


def summarize_distance(matrix, labels):
    i,j=np.triu_indices(len(labels),1);same=labels[i]==labels[j]
    within=float(matrix[i[same],j[same]].mean());between=float(matrix[i[~same],j[~same]].mean())
    return dict(different_instance_same_class=within,different_class=between,generalization_margin=between-within)


def prepare(brain, projection):
    records,sequences=load_dataset()
    # Visual inspection found this attempted capture had already closed the menu.
    excluded={'menu-06':'Framebuffer has no pause menu; capture label failed manual inspection.'}
    primary=[r for r in records if r['role']=='primary' and r['instance_id'] not in excluded]
    counts=Counter(r['class_label'] for r in primary)
    if min(counts.get(c,0) for c in CLASSES)<5:
        raise RuntimeError(f'Need at least five valid instances/class: {counts}')
    encoded={r['instance_id']:encode(s,projection) for r,s in zip(records,sequences)}
    regression_hashes=regression(projection)
    chosen=[(r,s) for r,s in zip(records,sequences) if r in primary]
    labels=np.array([r['class_label'] for r,s in chosen]);ids=[r['instance_id'] for r,s in chosen]
    pixels=np.stack([np.stack([spatial_grid(f,projection.config)[0] for f in s]).mean(axis=0) for r,s in chosen])
    enc=np.stack([encoded[r['instance_id']].ravel() for r,s in chosen])
    hist=np.stack([np.histogram(p,bins=16,range=(0,1),density=False)[0]/p.size for p in pixels])
    stats=np.stack([np.r_[p.mean(),p.std(),(p<.5).mean(),p.mean(axis=0),p.mean(axis=1)] for p in pixels])
    dm=cdist(pixels.reshape(len(pixels),-1),pixels.reshape(len(pixels),-1));em=cdist(enc,enc)
    # First-frame changed pixel fractions expose near duplicates even when sequence hashes differ.
    changed=np.zeros_like(dm)
    for i,(r,s) in enumerate(chosen):
        for j,(r2,s2) in enumerate(chosen):
            changed[i,j]=np.mean(np.any(s[0,:,:,:3]!=s2[0,:,:,:3],axis=-1))
    diagnostics=dict(instance_ids=ids,image_distance=summarize_distance(dm,labels),
        encoder_distance=summarize_distance(em,labels),pixel_euclidean=dm.tolist(),
        encoder_euclidean=em.tolist(),luminance_histogram_distance=cdist(hist,hist).tolist(),
        first_frame_changed_pixel_fraction=changed.tolist(),
        near_duplicate_pairs=[dict(a=ids[i],b=ids[j],changed_fraction=float(changed[i,j]))
                              for i in range(len(ids)) for j in range(i+1,len(ids)) if changed[i,j]<.01],
        pixel_statistics=stats.tolist(),
        encoder=[dict(instance_id=r['instance_id'],sha256=digest(encoded[r['instance_id']].tobytes()),
                      vector=encoded[r['instance_id']].ravel().tolist(),
                      class_centroid_distance=float(np.linalg.norm(enc[i]-enc[labels==labels[i]].mean(axis=0))))
                 for i,(r,s) in enumerate(chosen)],regression_checks=regression_hashes)
    try:
        from skimage.metrics import structural_similarity
        diagnostics['ssim']=[[float(structural_similarity(a,b,data_range=1.0)) for b in pixels] for a in pixels]
    except ImportError:
        diagnostics['ssim']=None
        diagnostics['ssim_note']='scikit-image unavailable; pixel, histogram and encoder distances recorded.'
    manifest=dict(classes=list(CLASSES),instances_per_class=dict(counts),seeds=list(SEEDS),
                  primary_ids=ids,excluded=excluded,instances=records,
                  total_unique_first_images=len({r['frame_sha256'][0] for r in primary}),
                  total_unique_framebuffers=len({h for r in primary for h in r['frame_sha256']}),
                  encoder_source_sha256=digest((ROOT/'flymon/spatiotemporal.py').read_bytes()),
                  config=asdict(projection.config),regression_checks=regression_hashes)
    RESULTS.mkdir(parents=True,exist_ok=True);CAPTURES.mkdir(parents=True,exist_ok=True)
    target=RESULTS/'dataset-manifest.json'
    if target.exists() and json.loads(target.read_text())!=manifest:
        raise RuntimeError('Frozen dataset manifest differs; do not overwrite')
    write_json(target,manifest);write_json(RESULTS/'dataset-diagnostics.json',diagnostics)
    matrix_svg(CAPTURES/'image-space-distance-matrix.svg',dm,ids,'Mean-frame pixel Euclidean distance')
    matrix_svg(CAPTURES/'encoder-space-distance-matrix.svg',em,ids,'Unchanged E3 sequence encoder distance')
    design=RESULTS/'design.md'
    text=f'''# Experiment 6 — frozen visual-instance generalization protocol

Frozen before neural simulation or outer-test evaluation. Dataset manifest SHA-256:
`{digest(target.read_bytes())}`. E3 encoder source SHA-256:
`{manifest['encoder_source_sha256']}`. No Experiments 1–5 result is changed.

## Dataset and independence

Classes and accepted sequence counts: {dict(counts)}. Each instance contains ten actual
RGBA framebuffer samples, 4 emulator frames apart; 20 neural steps/sample, 200 total.
An instance is a deterministic natural capture trajectory, not an augmented image.
All its frames and all neural repeats stay grouped. Distinct instances share no
identical framebuffer hashes, even across classes. First-frame changed fractions
below 1% are flagged, not silently removed based on neural results.

Bedroom means the canonical `states/bedroom.state` room view, with naturally changed
player position/facing and camera framing. The savestate filename is retained as
the class label; it is not independently inferred from RAM. Dialogue means Oak's
opening text-box pages (both Oak and Pokémon illustration backgrounds). Menu means
the in-room pause-menu family, varying cursor and underlying room framing. It does
not pool options, new-game menus, inventory or unrelated interfaces. Intro means
natural poses in the opening Gengar/Nidorino animation; title means the fixed logo
with naturally changing Pokémon sprites. These are narrow visual families, not all
possible Pokémon dialogue, rooms or menus. All captures are deterministic and can
be temporally/visually correlated: distinct pixels do not imply independent game
sessions or broad semantic coverage. One attempted menu capture (`menu-06`) had no
menu on inspection and is excluded before neural evaluation, retained for audit.
The options screen is an OOD probe only. No ROM bytes or RAM-derived labels/input.

## Pixel-only preflight

Mean-frame pixel distance: {diagnostics['image_distance']}.
Encoder sequence distance: {diagnostics['encoder_distance']}.
Near-duplicate pairs below 1% first-frame changed pixels: {len(diagnostics['near_duplicate_pairs'])}.
Inspect the contact sheet and recorded pixel/histogram/encoder distances. Means
may hide local changes; all original frames and hashes are retained. No label is
changed after neural results. Diagnostic image classifiers use mean, standard
deviation, dark-pixel fraction and horizontal/vertical luminance profiles only.

## Model and readouts

Unchanged Experiment 3 18×20 spatial_grid / temporal_features / SpatialProjection,
same LC10a/LPLC2 indices, gains, quantization and ten-frame timing. Bit-exact encoder
regression on all five original E3 sequences must pass. MaleCNS v1.0, 166700 neurons,
25582938 connections, flybrain 0.1.0, CUDA P40. Primary DN readout is 200-step counts.
LC10a/LPLC2 counts are recorded separately. No gameplay policy, reward or action decoder.
Twenty new CNS seeds per image: {list(SEEDS)}. No outcome-based replay filtering.

## Outer and inner splits (fixed before testing)

Within each class, sort accepted instance IDs. Aligned instance fold j holds out
the jth instance of every class having one. With unequal class counts, later folds
have fewer classes; every instance is tested exactly once per seed block. Cross
these instance folds with four seed blocks: 201–205, 206–210, 211–215, 216–220.
Outer training excludes ALL held-out instances at ANY seed and ALL held-out seeds
at ANY instance. Every primary (instance,seed) trial receives one outer prediction.
Report trial-weighted accuracy and balanced accuracy, per-class recall, confusion.
Chance is 20% for five-way classification even in smaller test folds.

Within each outer training set, leave out each remaining aligned instance fold,
and hold out one of the three remaining seed blocks, cycling by inner fold order.
Re-rank DNs on that inner training subset, and score candidates {list(SIZES)}.
Choose maximal pooled inner accuracy; ties prefer the earlier/larger size. Re-rank
on all outer training data, freeze size and IDs, then predict once. ANOVA-like
between-class / within-class residual score with epsilon 1e-6, stable index tie
break. Euclidean nearest centroid on raw counts; no normalization or fitted metric.
The all-DN and each fixed candidate-size curve use the same outer splits.
Frozen E4 diagnostic uses the exact 20 FlyWire IDs in its previously recorded
`informative_dn_top20` table; no E6-driven choice of that subset.

## Other diagnostics

Exact-instance identity: seed-block outer holdout with all instance identities in
training; inner seed-block CV repeats DN ranking and population-size choice. This
diagnostic intentionally sees the same images and is not category generalization.
Encoder/pixel statistics: same outer splits, no DN selection. Projection counts:
same outer splits and raw Euclidean nearest centroid. Four-class no-title nested
analysis repeats selection from scratch; interpret as secondary.

Distance conditions use DIFFERENT seeds: same image, different image/same class,
different class. Margin = mean different-class minus mean within-class/different-
instance distance. Encoder margin is deterministic over image pairs. DN/projection
raw distances include all seed pairs with unequal seeds. Selected-DN geometry uses
a preregistered fixed top 50 to compare distances in equal dimension: for each
instance pair and each seed block, rank using data excluding BOTH aligned instance
folds and that entire seed block, then measure distances only on the withheld
seeds/images. This separate cross-fitted geometry cannot tune the primary model.
Its matrix, margin and noise distance explicitly refer to top50, not an adaptive
union of outer-selected IDs. Stability: outer selected-set and top50 Jaccard.

Leave-one-class-out geometry is exploratory: select top50 using four classes only,
then describe held-class within/between distances; no prediction of unseen labels.
No-vision baseline at every seed; spatial destruction control for first accepted
instance of each class at every seed, one fixed pixel permutation (seed 606), same
permutation for all frames, preserves exact per-frame luminance histograms. Controls
and the OOD options screen never enter training, selection, permutations or bootstrap.
Evaluate their distances to outer-training centroids with frozen selected features.

## Inference and leakage audits

1000 permutations, RNG 6062026: permute class labels among instances WITHIN each
aligned instance-fold block. This preserves block class composition and every
seed replicate of an image receives the same permuted label. This restricted,
conditional instance-level null preserves the prespecified stratified fold design;
later incomplete blocks have fewer exchangeable labels. Repeat the FULL inner
ranking, size selection, outer ranking and prediction for every permutation.
p=(1+null accuracy >= observed)/(1001); report mean and 95th percentile.
10000 bootstrap resamples, RNG 6062027: resample whole image instances WITHIN class,
carrying all their out-of-fold seed predictions. Report percentile 95% accuracy CI;
this is conditional on fitted folds, not a retraining/independent-game-session CI.
Assert zero image-hash, instance and seed overlap in every outer AND inner split.
Audit ranking inputs; perturb outer-test activity/labels and verify unchanged fold
feature selection. Verify permutation labels are constant within instance and
full nesting executes again. Verify bootstrap resamples instance units.

## Interpretation frozen before test results

PASS needs substantial above-chance nested double-holdout accuracy, instance-level
p<.05, most classes informative without title dependence, positive cross-fitted
selected-DN margin, and either frozen E4 transfer or meaningful cross-fold stability.
70% is a strong guide, not an arbitrary statistical cutoff. WEAK means modest,
class-dependent or uncertain generalization or unstable selected neurons. FAIL
means near-chance generalization or representation failure, even if image identity
works. Report A (fixed-image recognition), B (encoder succeeds/projection fails),
C (projection succeeds/DNs fail), D (some DN transfer/unstable subset), or E (stable
DN generalization), choosing the best-supported case and stating limitations.
If encoder itself fails, A is the closest outcome with explicit upstream caveat.
Stop at Experiment 6; only discuss whether an Experiment 7 study is justified.
'''
    if design.exists() and design.read_text()!=text:
        raise RuntimeError('Frozen design differs; do not overwrite')
    design.write_text(text)
    write_json(RESULTS/'protocol-lock.json',dict(design_sha256=digest(design.read_bytes()),manifest_sha256=digest(target.read_bytes())))
    print('Prepared/frozen',dict(counts),'regression PASS',flush=True)
    return manifest,records,sequences,encoded


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
    from flybrain import FlyBrain
    from run_spatiotemporal_experiment import run_trial
    brain=FlyBrain(data=ROOT/'redfly-benchmark/data',device='cuda',batch=1)
    assert brain.device=='cuda' and brain.n==166700 and brain._W.nnz==25582938
    projection=SpatialProjection(brain,SpatiotemporalConfig())
    manifest,records,sequences,encoded=prepare(brain,projection)
    if args.prepare:return
    dn=brain.cells(['descending_neuron']);assert len(dn)==1314
    dn_slot=np.full(brain.n,-1,np.int32);dn_slot[dn]=np.arange(len(dn))
    vp_slot=np.full(brain.n,-1,np.int32);vp_slot[projection.ids]=np.arange(len(projection.ids))
    meta=np.load(ROOT/'redfly-benchmark/data/brain.npz')
    conditions={name:encoded[name] for name in manifest['primary_ids']}
    conditions['baseline_none']=None
    permutation=np.random.default_rng(606).permutation(144*160)
    for label in CLASSES:
        r=next(r for r in records if r['instance_id'] in manifest['primary_ids'] and r['class_label']==label)
        seq=sequences[records.index(r)]
        shuffled=seq.reshape(10,-1,4)[:,permutation].reshape(seq.shape)
        assert all(np.array_equal(np.sort(a.reshape(-1,4),axis=0),np.sort(b.reshape(-1,4),axis=0)) for a,b in zip(seq,shuffled))
        conditions['shuffled-'+r['instance_id']]=encode(shuffled,projection)
    for r in records:
        if r['role']=='ood':conditions[r['instance_id']]=encoded[r['instance_id']]
    target=RESULTS/'trials.json';lock=json.loads((RESULTS/'protocol-lock.json').read_text())
    output=dict(protocol_lock=lock,model='flybrain 0.1.0 MaleCNS v1.0',device='cuda',
                neurons=brain.n,connections=brain._W.nnz,seeds=list(SEEDS),steps=200,
                dn_indices=dn.tolist(),flywire_ids=[int(x) for x in meta['ids'][dn]],
                cell_types=brain.cell_type[dn].tolist(),projection_indices=projection.ids.tolist(),
                trials={},shuffle_pixel_permutation_sha256=digest(permutation.tobytes()))
    if target.exists():
        old=json.loads(target.read_text());assert old['protocol_lock']==lock
        output['trials']=old['trials']
    brain.reset(seed=0)
    for _ in range(20):brain.step(eye_drive=None)
    started=time.monotonic()
    for name,vectors in conditions.items():
        group=output['trials'].setdefault(name,[])
        for seed in SEEDS:
            if any(t['seed']==seed for t in group):continue
            trial=run_trial(brain,seed,vectors,projection,dn_slot,vp_slot,projection.config)
            trial['instance_id']=name
            group.append(trial)
        write_json(target,output)
        print(f'Completed {name}: {len(group)} seeds; elapsed {time.monotonic()-started:.1f}s',flush=True)
    print('All neural captures complete. No outer-test classification performed here.',flush=True)


if __name__=='__main__':main()
