"""Grouped analysis and stage diagnosis for Experiment 5."""
from __future__ import annotations
from itertools import combinations
import json, statistics, time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

from run_visual_experiment import ROOT, SEEDS
from run_spatiotemporal_experiment import NAMES
from analyze_temporal_dn import (accuracy, classification_record, distances, gram, grouped_gram,
                                  grouped_selected, nested_population_cv, representation)

RESULTS=ROOT/"results/experiment-05-retinotopic"; PERMUTATIONS=1000
POPS=(1314,500,250,100,50)

def stage_matrix(archive, all_conditions, conditions, group):
    indices=[all_conditions.index(c) for c in conditions]
    return archive["stage_"+group][indices].transpose(1,0,2).astype(np.float32)[:,:,None,:]

def classify(data,names):
    labels=np.tile(np.arange(len(names)),(len(SEEDS),1))
    return classification_record(grouped_gram(gram(data),labels),names)

def nested_selected(data,names):
    labels=np.tile(np.arange(len(names)),(len(SEEDS),1)); reps={200:data}
    # population selector expects a prior bin selection; aggregate is prespecified.
    bins=[{"bin_steps":200} for _ in SEEDS]
    confusion,choices,freq,informative=nested_population_cv(reps,bins)
    return classification_record(confusion,names),choices,freq,informative

def permutation(data,names,selected,observed,count=PERMUTATIONS):
    rng=np.random.default_rng(20260918+(1 if selected else 0)); labels=np.tile(np.arange(len(names)),(len(SEEDS),1)); null=[]
    for _ in range(count):
        shuffled=np.stack([rng.permutation(row) for row in labels])
        permuted=np.empty_like(data)
        for seed in range(len(SEEDS)):
            for item in range(len(names)): permuted[seed,shuffled[seed,item]]=data[seed,item]
        score=nested_selected(permuted,names)[0]["accuracy"] if selected else classify(permuted,names)["accuracy"]
        null.append(score)
    x=np.asarray(null); return {"observed":observed,"null_mean":float(x.mean()),"null_95th":float(np.quantile(x,.95)),
        "empirical_p":float((1+np.sum(x>=observed))/(count+1)),"permutations":count,"null_accuracies":x.tolist(),
        "shuffle":"condition labels shuffled independently within seed; all grouped folds, training-only ranking and population-size selection repeated"}

def main():
    started=time.perf_counter(); raw=json.loads((RESULTS/"trials.json").read_text()); trials=raw["trials"]; archive=np.load(RESULTS/"stage-counts.npz"); all_conditions=list(archive["conditions"])
    pokemon={}; stage={}; selected={}; choices={}
    for mode in ("direct","temporal"):
        cond=[f"pokemon_{mode}_{n}" for n in NAMES]
        pokemon[mode]={}
        for group in raw["group_sizes"]:
            data=stage_matrix(archive,all_conditions,cond,group); pokemon[mode][group]=classify(data,NAMES)
        data=stage_matrix(archive,all_conditions,cond,"descending"); selected[mode],choices[mode],_,_=nested_selected(data,NAMES)
        pokemon[mode]["distance"]=distances(data)
        fixed=stage_matrix(archive,all_conditions,cond,"experiment4_fixed_dn"); pokemon[mode]["experiment4_fixed_dn_diagnostic"]=classify(fixed,NAMES)
    synth_pairs={"left_vs_right":["left","right"],"vertical_vs_horizontal":["vertical","horizontal"],"motion_left_vs_right":["motion_left","motion_right"]}
    for label,pair in synth_pairs.items():
        cond=[f"synthetic_{x}" for x in pair]; stage[label]={}
        for group in raw["group_sizes"]: stage[label][group]=classify(stage_matrix(archive,all_conditions,cond,group),pair)
    primary=stage_matrix(archive,all_conditions,[f"pokemon_temporal_{n}" for n in NAMES],"descending")
    p_all=permutation(primary,NAMES,False,pokemon["temporal"]["descending"]["accuracy"])
    p_selected=permutation(primary,NAMES,True,selected["temporal"]["accuracy"])
    exp1=json.loads((ROOT/"results/experiment-01-photoreceptors/trials.json").read_text())
    exp3=json.loads((ROOT/"results/experiment-03-spatiotemporal/trials.json").read_text())
    comparison={"experiment1":{"all_dn_accuracy":None,"within":178.8,"between":168.4,"ratio":.942},
                "experiment3":{"all_dn_accuracy":exp3["dn_classifier"]["accuracy"],"selected_dn_accuracy":.983,
                               "within":177.0,"between":169.3,"ratio":.957},
                "experiment5":{"all_dn_accuracy":pokemon["temporal"]["descending"]["accuracy"],"selected_dn_accuracy":selected["temporal"]["accuracy"],**pokemon["temporal"]["distance"]}}
    # Earliest stage with perfect binary grouped calibration; otherwise explicit collapse point.
    order=["r1_6","lamina","t4","t5","visual_projection","descending"]
    distinguish={k:next((g for g in order if stage[k][g]["accuracy"]>.5),"none") for k in stage}
    pstage=pokemon["temporal"]
    if pstage["r1_6"]["accuracy"]<=.2: loss="A"
    elif pstage["lamina"]["accuracy"]<=.2: loss="B"
    elif pstage["visual_projection"]["accuracy"]<=.2: loss="C"
    elif pstage["descending"]["accuracy"]<=.2: loss="D"
    else: loss="E"
    verdict="PASS" if selected["temporal"]["accuracy"]>=.8 and pokemon["temporal"]["descending"]["accuracy"]>.4 and loss=="E" else ("WEAK" if selected["temporal"]["accuracy"]>.4 or pokemon["temporal"]["descending"]["accuracy"]>.4 else "FAIL")
    result={"protocol":{"held_out_group":"seed","conditions":list(NAMES),"chance":.2,"primary_transform":"existing Flymon temporal blend","population_sizes":POPS},
            "pokemon":pokemon,"nested_selected":selected,"population_choices":choices,"synthetic":stage,"synthetic_first_distinguishable_stage":distinguish,
            "comparison":comparison,"primary_information_loss_stage":loss,"verdict":verdict,"offline_classification_seconds":time.perf_counter()-started}
    (RESULTS/"classification.json").write_text(json.dumps(result,indent=2)+"\n")
    perms={"all_dn":p_all,"nested_selected_dn":p_selected}; (RESULTS/"permutation-results.json").write_text(json.dumps(perms,indent=2)+"\n")
    make_report(raw,result,perms)
    d=raw["mapping_diagnostics"]
    print(f"R1–R6 total: {d['r1_6_total']}\nR1–R6 mapped: {d['r1_6_mapped']}\nR1–R6 unmapped: {d['r1_6_unmapped']}\nleft/right mapped: {d['left_mapped']}/{d['right_mapped']}\nmedian mapping confidence: {d['median_mapping_confidence']:.3f}")
    print(f"\nSynthetic calibration:\nleft vs right distinguishable at: {distinguish['left_vs_right']}\nvertical vs horizontal distinguishable at: {distinguish['vertical_vs_horizontal']}\nmotion-left vs motion-right distinguishable at: {distinguish['motion_left_vs_right']}")
    print(f"\nPokémon all-DN accuracy: {pokemon['temporal']['descending']['accuracy']:.1%}\nPokémon selected-DN accuracy: {selected['temporal']['accuracy']:.1%}\nChance: 20.0%\nPermutation p: {p_selected['empirical_p']:.4f}")
    print(f"\nExperiment 1 selected/all-DN comparison: unavailable/{comparison['experiment1']['all_dn_accuracy']}\nExperiment 3 selected/all-DN comparison: 98.3%/{comparison['experiment3']['all_dn_accuracy']:.1%}\nExperiment 5 selected/all-DN comparison: {selected['temporal']['accuracy']:.1%}/{pokemon['temporal']['descending']['accuracy']:.1%}\n\nPrimary information-loss stage: {loss}\n\nExperiment 5 verdict: {verdict}")

def make_report(raw,c,p):
    d=raw["mapping_diagnostics"]; q=c["pokemon"]["temporal"]; sel=c["nested_selected"]["temporal"]; sep=q["distance"]
    text=f"""# Experiment 5 — inferred 2-D R1–R6 retinal geometry

**Verdict: {c['verdict']}. Primary information-loss stage: {c['primary_information_loss_stage']}.** The primary encoder uses the existing Flymon temporal blend on bilinearly sampled, linear-light Rec.709 luminance. Grouped leave-one-seed-out all-DN accuracy was **{q['descending']['accuracy']:.1%}** and nested training-selected DN accuracy was **{sel['accuracy']:.1%}**, versus 20% chance.

## Retinal mapping

`brain.visual` contains 6,006 cells: 3,377 R1–R6, 1,300 R7, and 1,329 R8. Experiment 5 maps only R1–R6. From local MaleCNS edges, every receptor's contacts onto column-annotated L1/L2/L3 cells were accumulated and the maximum-weight optic column selected. {d['r1_6_mapped']:,}/{d['r1_6_total']:,} mapped ({d['left_mapped']:,} left, {d['right_mapped']:,} right); {d['r1_6_unmapped']:,} were unmapped. Median confidence was {d['median_mapping_confidence']:.3f}; {d['below_0_8']:,} were below 0.8.

The axial embedding is `x=h1-0.5*h2`, `y=sqrt(3)/2*h2`. Each eye is normalized independently. Left maps to u=0..0.6; right is horizontally mirrored into u=0.4..1; v reverses hex y so larger y is screen-up. The 0.4..0.6 overlap is engineered and was not tuned against Pokémon images. R7, R8, and unmapped visual cells receive zero drive.

## Calibration and stage diagnosis

| Pair | R1–R6 | Lamina | T4 | T5 | Visual projection | DNs | First > chance |
|---|---:|---:|---:|---:|---:|---:|---|
"""
    for key,label in (("left_vs_right","left/right"),("vertical_vs_horizontal","vertical/horizontal"),("motion_left_vs_right","motion left/right")):
        row=c['synthetic'][key]; text+=f"| {label} | {row['r1_6']['accuracy']:.1%} | {row['lamina']['accuracy']:.1%} | {row['t4']['accuracy']:.1%} | {row['t5']['accuracy']:.1%} | {row['visual_projection']['accuracy']:.1%} | {row['descending']['accuracy']:.1%} | {c['synthetic_first_distinguishable_stage'][key]} |\n"
    text+=f"""

## Pokémon results

| Transform | R1–R6 | Lamina | T4 | T5 | Visual projection | All DNs | Nested selected DNs | Fixed Experiment 4 DN types |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for mode in ('direct','temporal'):
        x=c['pokemon'][mode]; text+=f"| {mode} | {x['r1_6']['accuracy']:.1%} | {x['lamina']['accuracy']:.1%} | {x['t4']['accuracy']:.1%} | {x['t5']['accuracy']:.1%} | {x['visual_projection']['accuracy']:.1%} | {x['descending']['accuracy']:.1%} | {c['nested_selected'][mode]['accuracy']:.1%} | {x['experiment4_fixed_dn_diagnostic']['accuracy']:.1%} |\n"
    text+=f"""

Primary all-DN distance: within {sep['mean_within']:.1f}, between {sep['mean_between']:.1f}, ratio {sep['between_within_ratio']:.3f}. The fixed-DN result is exploratory and separate from the nested classifier. Every primary outer fold reranked DNs using training spikes and selected among 1,314/500/250/100/50 using inner grouped validation.

## Permutation tests

| Result | Observed | Null mean | Null 95th | p |
|---|---:|---:|---:|---:|
| All DNs | {p['all_dn']['observed']:.1%} | {p['all_dn']['null_mean']:.1%} | {p['all_dn']['null_95th']:.1%} | {p['all_dn']['empirical_p']:.4f} |
| Nested selected DNs | {p['nested_selected_dn']['observed']:.1%} | {p['nested_selected_dn']['null_mean']:.1%} | {p['nested_selected_dn']['null_95th']:.1%} | {p['nested_selected_dn']['empirical_p']:.4f} |

## Comparison and performance

Experiment 1: all-DN classification was not reported; distance ratio 0.942. Experiment 3: all-DN 40.0%, nested selected-DN 98.3%, distance ratio 0.957. Experiment 5 values are reported above. Mean luminance conversion was {raw['encoder_performance']['mean_luminance_ms']:.3f} ms, bilinear sampling {raw['encoder_performance']['mean_sampling_ms']:.3f} ms, and eye-drive construction {raw['encoder_performance']['mean_drive_ms']:.3f} ms. Per-trial neural and aggregation timing is retained in `trials.json`; offline classification took {c['offline_classification_seconds']:.2f} s before report writing.

## Limitations

This is inferred connectome geometry projected onto an arbitrary display, not measured fly vision. Display projection is not calibrated to fly viewing angle; eye pose is unknown; viewport overlap is engineered; photoreceptor physiology is simplified; FlyBrain's spiking lamina may not preserve graded biological signals; and Pokémon pixels are not natural fly stimuli.

## Answer

The result is judged from grouped discrimination, calibration propagation, and comparison with Experiments 1 and 3. No gameplay controls, reward, reinforcement learning, action decoder, or Pokémon RAM sensory input was added.
"""
    (RESULTS/"analysis.md").write_text(text)

if __name__=="__main__": main()
