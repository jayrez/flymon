"""Frozen Experiment 6 analysis. Never trains on controls or held-out images/seeds."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
import multiprocessing as mp
import time
import numpy as np
from scipy.spatial.distance import cdist
from flymon.dataset import ROOT, RESULTS, CAPTURES, CLASSES, SEEDS, SIZES, digest, write_json, load_dataset
from flymon.generalization import (PreparedFold, split_plan, instance_blocks, prepare_plan, nested,
                                   unselected, record, audit_plan, bootstrap, permutation_labels)
from run_generalization_experiment import matrix_svg


def bar_svg(path,labels,values,title):
    from xml.sax.saxutils import escape
    width=700;height=80+len(labels)*32;scale=max(max(values),1e-9)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',f'<text x="15" y="25">{escape(title)}</text>']
    for i,(label,value) in enumerate(zip(labels,values)):
        y=50+i*32
        parts += [f'<text x="145" y="{y+15}" text-anchor="end" font-size="12">{escape(str(label))}</text>',
                  f'<rect x="155" y="{y}" width="{460*value/scale}" height="22" fill="#275e85"/>',
                  f'<text x="625" y="{y+15}" font-size="12">{value:.4f}</text>']
    path.write_text('\n'.join(parts+['</svg>']))


def check_lock():
    design=RESULTS/'design.md'
    if not design.exists():raise RuntimeError('Freeze design before evaluation')
    lock=json.loads((RESULTS/'protocol-lock.json').read_text())
    assert digest(design.read_bytes())==lock['design_sha256'],'Design changed'
    assert digest((RESULTS/'dataset-manifest.json').read_bytes())==lock['manifest_sha256'],'Dataset changed'
    records,_=load_dataset()
    manifest=json.loads((RESULTS/'dataset-manifest.json').read_text())
    assert records==manifest['instances']
    assert digest((ROOT/'flymon/spatiotemporal.py').read_bytes())==manifest['encoder_source_sha256']
    return lock,manifest


def trial_arrays(trials,ids,key):
    values=[]
    for name in ids:
        group=trials[name]
        assert sorted(t['seed'] for t in group)==list(SEEDS),f'Incomplete/duplicate seeds: {name}'
        byseed={t['seed']:t for t in group}
        values.append([byseed[s][key] for s in SEEDS])
    return np.asarray(values,np.float64)


def distance_diagnostics(x,y):
    n,s,d=x.shape;matrix=np.zeros((n,n));noise=[];within=[];between=[]
    unequal=~np.eye(s,dtype=bool)
    for i in range(n):
        for j in range(i,n):
            values=cdist(x[i],x[j])[unequal]
            v=float(values.mean());matrix[i,j]=matrix[j,i]=v
            (noise if i==j else within if y[i]==y[j] else between).append(v)
    return dict(same_instance_different_seed=float(np.mean(noise)),
                different_instance_same_class=float(np.mean(within)),different_class=float(np.mean(between)),
                generalization_margin=float(np.mean(between)-np.mean(within)),matrix=matrix.tolist(),
                distance='Euclidean; only unequal seed pairs; pair means equally weighted')


def selected_geometry(x,y):
    blocks=instance_blocks(y);assignment={int(i):j for j,b in enumerate(blocks) for i in b}
    seedblocks=np.array_split(np.arange(x.shape[1]),4);matrix=np.zeros((len(x),len(x)))
    ranks={}
    for i in range(len(x)):
        for j in range(i,len(x)):
            pair=tuple(sorted({assignment[i],assignment[j]}));values=[]
            for k,ss in enumerate(seedblocks):
                key=(pair,k)
                if key not in ranks:
                    excluded=np.concatenate([blocks[b] for b in pair]);ti=np.setdiff1d(np.arange(len(x)),excluded)
                    ts=np.setdiff1d(np.arange(x.shape[1]),ss)
                    f=PreparedFold.make(x,dict(train_i=ti,train_s=ts,test_i=[i,j],test_s=ss))
                    ranks[key]=f.fit(y,len(CLASSES))[1][:50]
                selected=ranks[key]
                a=x[i,ss][:,selected];b=x[j,ss][:,selected]
                values.append(float(cdist(a,b)[~np.eye(len(ss),dtype=bool)].mean()))
            matrix[i,j]=matrix[j,i]=np.mean(values)
    within=[];between=[]
    for i,j in combinations(range(len(x)),2):(within if y[i]==y[j] else between).append(matrix[i,j])
    return dict(same_instance_different_seed=float(np.diag(matrix).mean()),
                different_instance_same_class=float(np.mean(within)),different_class=float(np.mean(between)),
                generalization_margin=float(np.mean(between)-np.mean(within)),matrix=matrix.tolist(),
                population=50,method='Pairwise cross-fitted top50: exclude both instance blocks and test seed block before ranking')


def stability(choices,key):
    sets=[set(c[key]) for c in choices]
    values=[len(a&b)/len(a|b) for a,b in combinations(sets,2)]
    # Same-seed-block, different-image-block comparison separates image variation.
    across_images=[len(sets[i]&sets[j])/len(sets[i]|sets[j]) for i,j in combinations(range(len(sets)),2)
                   if choices[i]['seed_block']==choices[j]['seed_block']]
    frequency=Counter(n for s in sets for n in s)
    return dict(mean_jaccard=float(np.mean(values)),min_jaccard=float(min(values)),
                same_seed_block_different_instance_block_jaccard=float(np.mean(across_images)),
                selection_frequency={str(k):v for k,v in frequency.most_common()})


def instance_identity(x):
    # Here image identity IS the label, deliberately shared across train/test.
    y=np.arange(len(x));blocks=np.array_split(np.arange(x.shape[1]),4)
    p=np.full(x.shape[:2],-1,int);allp=p.copy();choices=[]
    for k,ss in enumerate(blocks):
        ts=np.setdiff1d(np.arange(x.shape[1]),ss);correct=np.zeros(len(SIZES),int);total=0
        for j,vs in enumerate(blocks):
            if j==k:continue
            f=PreparedFold.make(x,dict(train_i=y,train_s=np.setdiff1d(ts,vs),test_i=y,test_s=vs))
            means,ranking=f.fit(y,len(y));pred=f.predictions(means,ranking,SIZES)
            correct+=(pred==f.truth(y)[None]).sum(axis=1);total+=pred.shape[1]
        chosen=int(np.argmax(correct));f=PreparedFold.make(x,dict(train_i=y,train_s=ts,test_i=y,test_s=ss))
        means,ranking=f.fit(y,len(y));pred=f.predictions(means,ranking,SIZES)
        p[:,ss]=pred[chosen].reshape(len(y),len(ss));allp[:,ss]=pred[0].reshape(len(y),len(ss))
        choices.append(dict(seed_block=k,population=SIZES[chosen],inner_accuracy=(correct/total).tolist()))
    return allp,p,choices


def exploratory_class_geometry(x,y):
    out={};allseeds=np.arange(x.shape[1]);seedblocks=np.array_split(allseeds,4)
    for held,name in enumerate(CLASSES):
        train_i=np.flatnonzero(y!=held);test_i=np.flatnonzero(y==held);remap=y.copy()
        for k,c in enumerate(sorted(set(y[train_i]))):remap[y==c]=k
        summaries=[]
        for ss in seedblocks:
            f=PreparedFold.make(x,dict(train_i=train_i,train_s=np.setdiff1d(allseeds,ss),test_i=test_i,test_s=ss))
            means,ranking=f.fit(remap,4);sel=ranking[:50]
            held_means=x[test_i][:,ss][:,:,sel].mean(axis=1)
            other_means=x[train_i][:,ss][:,:,sel].mean(axis=1)
            within=cdist(held_means,held_means)[np.triu_indices(len(test_i),1)].mean()
            between=cdist(held_means,other_means).mean()
            summaries.append(dict(within=float(within),between=float(between),margin=float(between-within)))
        out[name]=dict(within=float(np.mean([a['within'] for a in summaries])),
                       between=float(np.mean([a['between'] for a in summaries])),
                       margin=float(np.mean([a['margin'] for a in summaries])))
    return dict(exploratory=True,method='Held class absent from top50 ranking; seed-block holdout; distances between instance seed means',classes=out)


def control_diagnostics(x,y,plan,choices,raw,ids):
    names=[n for n in raw if n=='baseline_none' or n.startswith(('shuffled-','ood-'))]
    data=trial_arrays(raw,names,'dn_counts');out={}
    for name,control in zip(names,data):
        nearest=[];reference=[];pred=[];source_pred=[]
        source_index=ids.index(name[len('shuffled-'):]) if name.startswith('shuffled-') else None
        for f,choice in zip(plan,choices):
            if source_index is not None and source_index not in f['test_i']:
                continue
            selected=choice['selected'];ss=f['test_s']
            train=x[f['train_i']][:,f['train_s']][:,:,selected]
            centroids=np.stack([train[y[f['train_i']]==c].mean(axis=(0,1)) for c in range(len(CLASSES))])
            dist=cdist(control[ss][:,selected],centroids);nearest.extend(dist.min(axis=1));pred.extend(dist.argmin(axis=1).tolist())
            test=x[f['test_i']][:,ss][:,:,selected].reshape(-1,len(selected))
            reference.extend(cdist(test,centroids).min(axis=1))
            if source_index is not None:
                source_pred.extend(cdist(x[source_index,ss][:,selected],centroids).argmin(axis=1).tolist())
        out[name]=dict(mean_nearest_training_centroid_distance=float(np.mean(nearest)),
                       primary_test_mean_nearest_centroid_distance=float(np.mean(reference)),
                       distance_ratio=float(np.mean(nearest)/np.mean(reference)),
                       assigned_labels_descriptive_only=dict(Counter(CLASSES[k] for k in pred)),
                       note='Repeated across outer models; not independent trials; no OOD accuracy or forced correctness')
        out[name]['evaluation_count']=len(pred)
        if source_index is not None:
            assert len(pred)==x.shape[1]
            out[name]['source_class_assignment_fraction']=float(np.mean(np.array(pred)==y[source_index]))
            out[name]['unshuffled_source_class_accuracy']=float(np.mean(np.array(source_pred)==y[source_index]))
            out[name]['note']='Only folds holding out the original source image AND test seeds; source-label retention is diagnostic, not OOD correctness.'
    return out


_PERM_CONTEXT=None


def perm_worker(item):
    index,labels=item;x,y,plan,prepared=_PERM_CONTEXT
    result=nested(x,labels,plan,prepared=prepared,details=False)
    return index,result['accuracy'],result['ranking_fits']


def permutations(x,y,plan,observed,count=1000,workers=4):
    global _PERM_CONTEXT
    rng=np.random.default_rng(6062026);blocks=instance_blocks(y)
    labels=[permutation_labels(y,blocks,rng) for _ in range(count)]
    prepared=prepare_plan(x,plan);_PERM_CONTEXT=(x,y,plan,prepared)
    null=[None]*count;fits=[None]*count;target=RESULTS/'permutation-progress.json'
    fingerprint=digest(x.tobytes()+y.tobytes()+json.dumps(plan,sort_keys=True).encode())
    if target.exists():
        old=json.loads(target.read_text())
        if old['fingerprint']==fingerprint:
            for i,v in enumerate(old['null_accuracies'][:count]):null[i]=v;fits[i]=old['ranking_fits'][i]
    todo=[(i,l) for i,l in enumerate(labels) if null[i] is None];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        for done,(i,value,nfit) in enumerate(pool.map(perm_worker,todo,chunksize=2),1):
            null[i]=value;fits[i]=nfit
            if done%20==0 or done==len(todo):
                write_json(target,dict(fingerprint=fingerprint,null_accuracies=null,ranking_fits=fits))
                print(f'Permutation {sum(v is not None for v in null)}/{count}; elapsed {time.monotonic()-started:.1f}s',flush=True)
    expected=len(plan)+sum(len(f['inner']) for f in plan)
    assert all(f==expected for f in fits)
    return dict(observed_accuracy=observed,null_mean=float(np.mean(null)),null_95th_percentile=float(np.quantile(null,.95)),
                empirical_p=float((1+np.count_nonzero(np.asarray(null)>=observed))/(count+1)),
                permutations=count,null_accuracies=null,ranking_fits_per_permutation=expected,
                shuffle_unit='visual instance; permute within aligned instance-fold blocks; one label for all seeds',
                repeats_full_nested_selection=True,seed=6062026,
                first_permutation_instance_labels=labels[0].tolist())


def intervention_audit(x,y,plan):
    """Change all outer-test pixels' repeats AND all held-out seeds, then refit."""
    checks=[]
    for f in plan:
        outer=PreparedFold.make(x,f);means,rank=outer.fit(y,len(CLASSES))
        altered=x.copy();altered[f['test_i']]=1e7;altered[:,f['test_s']]=-1e7
        yy=y.copy();yy[f['test_i']]=(yy[f['test_i']]+1)%len(CLASSES)
        m2,r2=PreparedFold.make(altered,f).fit(yy,len(CLASSES))
        assert np.array_equal(rank,r2) and np.array_equal(means,m2)
        for inner in f['inner']:
            a=PreparedFold.make(x,inner);b=PreparedFold.make(altered,inner)
            ma,ra=a.fit(y,len(CLASSES));mb,rb=b.fit(yy,len(CLASSES))
            assert np.array_equal(ra,rb) and np.array_equal(ma,mb)
            assert np.array_equal(a.truth(y),b.truth(yy))
            assert np.array_equal(a.predictions(ma,ra,SIZES),b.predictions(mb,rb,SIZES))
        checks.append(True)
    return all(checks)


def analyze(count=1000,workers=4):
    lock,manifest=check_lock();raw=json.loads((RESULTS/'trials.json').read_text())
    assert raw['protocol_lock']==lock
    ids=manifest['primary_ids'];byid={r['instance_id']:r for r in manifest['instances']};records=[byid[n] for n in ids]
    y=np.array([CLASSES.index(r['class_label']) for r in records]);plan=split_plan(y)
    x=trial_arrays(raw['trials'],ids,'dn_counts');vp=trial_arrays(raw['trials'],ids,'vp_counts')
    # Require completion of every declared control before reporting final results.
    controls=['baseline_none',*[f'shuffled-{next(n for n in ids if byid[n]["class_label"]==c)}' for c in CLASSES],
              *[r['instance_id'] for r in records if r['role']=='ood']]
    controls += [r['instance_id'] for r in manifest['instances'] if r['role']=='ood']
    trial_arrays(raw['trials'],controls,'dn_counts')
    assert x.shape==(len(ids),20,1314)
    checks=audit_plan(plan,records,SEEDS)
    write_json(RESULTS/'split-manifest.json',dict(protocol_lock=lock,instance_ids=ids,
        instance_frame_hashes={r['instance_id']:r['frame_sha256'] for r in records},seeds=list(SEEDS),
        indexing='train_i/test_i index instance_ids; train_s/test_s index seeds; Cartesian products',folds=plan))
    checks['dn_selection_leakage']=intervention_audit(x,y,plan)
    old=json.loads((ROOT/'results/experiment-04-temporal-dn/classification.json').read_text())
    stable=old['informative_dn_top20'];id_to_slot={v:i for i,v in enumerate(raw['flywire_ids'])}
    frozen=np.array([id_to_slot[t['flywire_id']] for t in stable]);assert len(frozen)==20
    result=nested(x,y,plan,frozen=frozen)
    c=dict(protocol_lock=lock,all_dn=record(result['fixed'][0],y,CLASSES),
           nested_selected_dn=record(result['predictions'],y,CLASSES),
           frozen_experiment4_dn=record(result['frozen'],y,CLASSES),frozen_experiment4_ids=stable,
           by_population={str(s):record(p,y,CLASSES) for s,p in zip(SIZES,result['fixed'])},choices=result['choices'],
           selection_stability=stability(result['choices'],'selected'),top50_stability=stability(result['choices'],'top50'))
    c['bootstrap']=bootstrap(result['predictions'],y)
    diag=json.loads((RESULTS/'dataset-diagnostics.json').read_text())
    encoder=np.array([d['vector'] for d in diag['encoder']]);pixels=np.array(diag['pixel_statistics'])
    # Post-hoc sensitivity only: group the visually flagged near-duplicate pair
    # together so neither can train a model evaluating the other. Primary stays frozen.
    blocks=instance_blocks(y)
    for pair in diag['near_duplicate_pairs']:
        a=ids.index(pair['a']);b=ids.index(pair['b'])
        touched=[i for i,block in enumerate(blocks) if a in block or b in block]
        merged=np.sort(np.concatenate([blocks[i] for i in touched]))
        blocks=[block for i,block in enumerate(blocks) if i not in touched]+[merged]
    robust_plan=split_plan(y,blocks=blocks)
    audit_plan(robust_plan,records,SEEDS)
    robust=nested(x,y,robust_plan)
    c['near_duplicate_sensitivity']=dict(post_hoc=True,
        method='Merge aligned instance blocks containing flagged near-duplicate pairs; both excluded together at every seed. Rerun complete nested selection. No primary labels or results changed.',
        flagged_pairs=diag['near_duplicate_pairs'],folds=robust_plan,
        result=record(robust['predictions'],y,CLASSES))
    c['encoder']=record(unselected(np.repeat(encoder[:,None],20,axis=1),y,plan),y,CLASSES)
    c['pixel_statistics']=record(unselected(np.repeat(pixels[:,None],20,axis=1),y,plan),y,CLASSES)
    c['projection']=record(unselected(vp,y,plan),y,CLASSES)
    allp,selectedp,ic=instance_identity(x)
    c['instance_all_dn']=record(allp,np.arange(len(ids)),ids)
    c['instance_selected_dn']=record(selectedp,np.arange(len(ids)),ids);c['instance_choices']=ic
    four=y!=CLASSES.index('title');four_names=[n for n in CLASSES if n!='title']
    fy=np.array([four_names.index(CLASSES[v]) for v in y[four]])
    fr=nested(x[four],fy,split_plan(fy));c['without_title']=record(fr['predictions'],fy,four_names)
    c['distances']=dict(encoder=diag['encoder_distance'],projection=distance_diagnostics(vp,y),
                        all_dn=distance_diagnostics(x,y),selected_dn=selected_geometry(x,y))
    c['leave_one_class_out']=exploratory_class_geometry(x,y)
    c['controls']=control_diagnostics(x,y,plan,result['choices'],raw['trials'],ids)
    checks['population_size_training_only']=checks['dn_selection_leakage'] and checks['inner_containment']
    checks['bootstrap_instance_unit']=c['bootstrap']['number_of_instance_units']==len(ids)
    write_json(RESULTS/'classification.json',c)
    print('Observed nested accuracy',c['nested_selected_dn']['accuracy'],'all DN',c['all_dn']['accuracy'],flush=True)
    perm=permutations(x,y,plan,c['nested_selected_dn']['accuracy'],count,workers)
    checks['permutation_leakage']=perm['repeats_full_nested_selection'] and perm['ranking_fits_per_permutation']==len(plan)*8
    checks['permutation_instance_unit']=all(np.all(np.repeat(v,20)==v) for v in perm['first_permutation_instance_labels'])
    c['leakage_checks']={k:'PASS' if v else 'FAIL' for k,v in checks.items()};assert all(checks.values())
    write_json(RESULTS/'classification.json',c);write_json(RESULTS/'permutation-results.json',perm)
    for key,filename,title in [('projection','projection-neuron-distance-matrix.svg','Projection neurons: different-seed distances'),
                              ('selected_dn','selected-DN-distance-matrix.svg','Cross-fitted top50 DN distances')]:
        matrix_svg(CAPTURES/filename,c['distances'][key]['matrix'],ids,title)
    matrix_svg(CAPTURES/'classification-confusion-matrix.svg',c['nested_selected_dn']['confusion'],CLASSES,'Nested double-holdout confusion (rows=true)')
    bar_svg(CAPTURES/'accuracy-by-DN-population-size.svg',[*map(str,SIZES),'nested'],
            [c['by_population'][str(s)]['accuracy'] for s in SIZES]+[c['nested_selected_dn']['accuracy']], 'Outer accuracy; fixed-size curves are diagnostic')
    bar_svg(CAPTURES/'per-class-recall.svg',CLASSES,list(c['nested_selected_dn']['per_class_recall'].values()),'Nested double-holdout per-class recall')
    hist,bins=np.histogram(perm['null_accuracies'],bins=20)
    bar_svg(CAPTURES/'permutation-null.svg',[f'{a:.3f}–{b:.3f}' for a,b in zip(bins[:-1],bins[1:])],hist.tolist(),
            f'Instance permutation null; observed={perm["observed_accuracy"]:.3f}; p={perm["empirical_p"]:.4f}')
    report(c,perm,manifest)


def report(c,p,m):
    accuracy=c['nested_selected_dn']['accuracy'];recall=c['nested_selected_dn']['per_class_recall']
    significant=p['empirical_p']<.05;positive=c['distances']['selected_dn']['generalization_margin']>0
    stable=c['top50_stability']['mean_jaccard']>=.25
    broad=sum(v>.4 for v in recall.values())>=4 and c['without_title']['accuracy']>.5
    strong=accuracy>=.7 and significant and positive and broad and (stable or c['frozen_experiment4_dn']['accuracy']>.4)
    verdict='PASS' if strong else 'WEAK' if accuracy>.3 and significant else 'FAIL'
    outcome='E' if strong and stable else 'D' if accuracy>.3 and significant else 'C' if c['projection']['accuracy']>.4 else 'B' if c['encoder']['accuracy']>.4 else 'A'
    # Descriptive thresholds above operationalize the prespecified qualitative guide,
    # not additional significance tests; all underlying values are printed.
    answer='YES' if verdict=='PASS' else 'PARTIALLY' if verdict=='WEAK' else 'NO'
    confused=[];conf=np.array(c['nested_selected_dn']['confusion'])
    for i,j in combinations(range(5),2):confused.append((int(conf[i,j]+conf[j,i]),f'{CLASSES[i]} / {CLASSES[j]}'))
    confused.sort(reverse=True);d=c['distances']['selected_dn'];ci=c['bootstrap']['interval95']
    summary=f'''Experiment 6 verdict: {verdict}

Classes: {', '.join(CLASSES)}
Instances per class: {m['instances_per_class']}
Total unique images: {m['total_unique_first_images']} instance first frames; {m['total_unique_framebuffers']} unique sequence framebuffers

Primary outer protocol: unseen image instance + unseen neural seed
Chance accuracy: 20.0% (uniform five-way); largest-class baseline 25.0%

All-DN held-out-instance accuracy: {c['all_dn']['accuracy']:.2%}
Nested selected-DN held-out-instance accuracy: {accuracy:.2%}
Frozen Experiment-4-DN accuracy: {c['frozen_experiment4_dn']['accuracy']:.2%}

95% instance-bootstrap interval: {ci[0]:.2%}–{ci[1]:.2%}
Permutation p-value: {p['empirical_p']:.6f}

Exact-image instance classification: all-DN {c['instance_all_dn']['accuracy']:.2%}; selected-DN {c['instance_selected_dn']['accuracy']:.2%}; chance {1/len(m['primary_ids']):.3%}
Semantic class classification: {accuracy:.2%}; balanced accuracy {c['nested_selected_dn']['balanced_accuracy']:.2%}

Same-instance / different-seed distance: {d['same_instance_different_seed']:.4f}
Different-instance / same-class distance: {d['different_instance_same_class']:.4f}
Different-class distance: {d['different_class']:.4f}
(Distances above: independent pairwise cross-fitted top50 DNs.)

Encoder generalization margin: {c['distances']['encoder']['generalization_margin']:.4f}
Projection-neuron generalization margin: {c['distances']['projection']['generalization_margin']:.4f}
All-DN generalization margin: {c['distances']['all_dn']['generalization_margin']:.4f}
Selected-DN generalization margin: {d['generalization_margin']:.4f}

DN population size selected by inner CV: {dict(Counter(q['population'] for q in c['choices']))}
DN selection stability across folds: selected-set Jaccard {c['selection_stability']['mean_jaccard']:.4f}; top50 Jaccard {c['top50_stability']['mean_jaccard']:.4f}

Most confused classes: {confused[0][1]} ({confused[0][0]} errors)
Best-generalizing class: {max(recall,key=recall.get)} ({max(recall.values()):.2%})
Worst-generalizing class: {min(recall,key=recall.get)} ({min(recall.values()):.2%})

Primary result: {outcome}

Leakage checks:
image overlap: {c['leakage_checks']['image_overlap']}
instance overlap: {c['leakage_checks']['instance_overlap']}
seed overlap: {c['leakage_checks']['seed_overlap']}
DN-selection leakage: {c['leakage_checks']['dn_selection_leakage']}
permutation leakage: {c['leakage_checks']['permutation_leakage']}
population-size selection: {c['leakage_checks']['population_size_training_only']}
bootstrap instance unit: {c['leakage_checks']['bootstrap_instance_unit']}

Does the Experiment 4 DN signal generalize beyond the five exact screens? {answer}
'''
    lines=['# Experiment 6 — visual-instance generalization','', '```text',summary,'```','',
           '## Inference and limits','',
           f"The nested classifier scored {accuracy:.2%} on 640 image/seed trials from 32 visual instances. Every image and seed was excluded from its outer training fold. The restricted instance-level null mean was {p['null_mean']:.2%}, its 95th percentile {p['null_95th_percentile']:.2%}, and p={p['empirical_p']:.6f} over {p['permutations']} complete nested repetitions. The null preserves aligned-block composition; it is not an unrestricted image-label permutation. The bootstrap uses images, not the 640 neural repeats, and is conditional on the fitted folds.", '',
           f"Encoder class accuracy was {c['encoder']['accuracy']:.2%}, projection-population accuracy {c['projection']['accuracy']:.2%}. Excluding Title and repeating nested selection gave {c['without_title']['accuracy']:.2%} (25% chance). Trivial image-statistics accuracy was {c['pixel_statistics']['accuracy']:.2%}; high performance here limits claims about sophisticated or semantic neural representation.", '',
           'These captures cover narrow visual families: one canonical room, Oak introduction text, one pause-menu layout, one intro animation and changing title sprites. They are naturally different images, not independent game sessions or diverse world states. Distinct hashes prove pixel separation, not semantic independence. The preflight flags one near-duplicate pair; see dataset-diagnostics.json. Menu here differs from the original New Game menu, so the frozen E4 subset is a transfer diagnostic across that shift as well.', '',
           'Euclidean distances depend on dimension. The selected-DN margin uses separately cross-fitted fixed top50 features, excluding both compared instance blocks and test seeds. It is not the distance in the variable-size primary classifier. Stability overlaps are descriptive and partly reflect overlap among training folds. Exact-image identification uses 32 labels, so its absolute accuracy is not directly comparable to five-way class accuracy.', '',
           f"Post-hoc near-duplicate sensitivity: merging both aligned blocks containing menu-03 and menu-09, so neither image can train a model testing the other, gives {c['near_duplicate_sensitivity']['result']['accuracy']:.2%} nested class accuracy. This stricter diagnostic does not change the frozen primary analysis or its p-value. The five menu captures should not be described as five independent background layouts: this pair changes only a small cursor region.", '',
           '## Per-class recall','', '| Class | Recall |','| --- | ---: |']
    lines += [f'| {n} | {v:.2%} |' for n,v in recall.items()]
    lines += ['', '## Distance decomposition','', '| Stage | Same image/new seed | New image/same class | Different class | Margin |', '| --- | ---: | ---: | ---: | ---: |']
    for name,dist in c['distances'].items():
        lines.append(f"| {name} | {dist.get('same_instance_different_seed',0):.4f} | {dist['different_instance_same_class']:.4f} | {dist['different_class']:.4f} | {dist['generalization_margin']:.4f} |")
    lines += ['', '## Controls and exploratory geometry','', '| Probe | Nearest-centroid distance / primary test distance |','| --- | ---: |']
    lines += [f"| {name} | {v['distance_ratio']:.3f} |" for name,v in c['controls'].items()]
    lines += ['', '| Shuffled source | Original class accuracy | Shuffled source-label retention |', '| --- | ---: | ---: |']
    lines += [f"| {name} | {v['unshuffled_source_class_accuracy']:.2%} | {v['source_class_assignment_fraction']:.2%} |" for name,v in c['controls'].items() if 'source_class_assignment_fraction' in v]
    lines += ['', 'Shuffled-source comparisons use only models that exclude the original source instance and the test seeds. Label retention is descriptive, not correctness on a semantically valid screen.', '', 'These ratios use frozen outer models. They are descriptive, not a calibrated OOD detector; no closest-class assignment is counted as correct. The shuffled controls preserve each frame’s pixel histogram, and none of the controls entered training. Full assigned-label distributions are in classification.json.', '',
              'Spatial destruction preserved the dialogue, title and intro source labels for all 20 test seeds each, while bedroom and menu source-label retention fell to zero. Thus the strongest class separation does not require intact screen structure and may largely reflect luminance statistics; the controls do not support a claim of sophisticated spatial or semantic recognition. The options probe has a nearest-centroid distance ratio of about 1.007, so this diagnostic provides no clear evidence of novelty separation from all trained classes.', '',
              '| Class excluded from feature ranking | Held-class within distance | Distance to other classes | Margin |', '| --- | ---: | ---: | ---: |']
    lines += [f"| {name} | {v['within']:.3f} | {v['between']:.3f} | {v['margin']:.3f} |" for name,v in c['leave_one_class_out']['classes'].items()]
    lines += ['', 'This leave-one-class-out geometry is exploratory and predicts no unseen label.', '', '## Scientific conclusion','']
    conclusion=(f"Outcome {outcome}: "+('A stable selected DN representation transfers to unseen images and unseen CNS seeds within these narrow capture families. The earlier 98.3% result is therefore not solely recognition of five fixed images, but this experiment does not establish broad Pokémon-state understanding.' if verdict=='PASS' else
                 'DN activity shows limited transfer to unseen images and seeds, but the evidence does not establish robust category generalization. The earlier 98.3% fixed-stimulus result should not be treated as a general-state recognition rate.' if verdict=='WEAK' else
                 'This dataset does not establish useful descending-neuron category generalization beyond fixed images. The earlier 98.3% remains a fixed-stimulus result; the present failure does not by itself prove that all possible DN readouts lack generalization.'))
    lines += [conclusion,'',('The evidence supports considering a separate, carefully scoped Experiment 7 behavioral-interface study, with the simple-statistics and narrow-dataset limitations above.' if verdict=='PASS' else 'A behavioral-interface study is not yet justified as a robust visual-state interface; improve and independently validate visual coverage first.'),
              'No gameplay control, action decoding, reward modeling or reinforcement learning was implemented. Experiments 1–5 were not modified.','',
              '## Reproduction','', '```bash','POKEMON_ROM=/external/pokemon-red.gb redfly-benchmark/.venv/bin/python capture_visual_instance.py --collect',
              'OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python run_generalization_experiment.py',
              'OPENBLAS_NUM_THREADS=1 redfly-benchmark/.venv/bin/python analyze_generalization_experiment.py --permutations 1000 --workers 4','```',
              '', 'Dataset/protocol hashes, frame hashes, full inner/outer index splits, per-trial counts, exact frozen FlyWire IDs, prediction arrays, permutation draws, selection frequencies and diagnostics are stored beside this report.']
    (RESULTS/'analysis.md').write_text('\n'.join(lines)+'\n');(RESULTS/'summary.txt').write_text(summary)
    print(summary,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--permutations',type=int,default=1000);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.permutations<1000:p.error('Final protocol requires at least 1000 permutations')
    analyze(args.permutations,args.workers)


if __name__=='__main__':main()
