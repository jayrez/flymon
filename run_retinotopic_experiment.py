"""Experiment 5: inferred 2-D R1-R6 retina through FlyBrain's eye-drive path."""
from __future__ import annotations

import hashlib, json, time
from pathlib import Path
import cupy as cp
import numpy as np
from PIL import Image, ImageDraw
from flybrain import FlyBrain

from flymon.emulator import PokemonEmulator
from flymon.retina import (bilinear_sample, full_eye_drive, infer_retina, linear_luminance,
                           load_mapping, save_mapping, synthetic_frame)
from run_spatiotemporal_experiment import move_to_condition, NAMES
from run_visual_experiment import DATA, ROOT, SEEDS, STEPS

RESULTS = ROOT / "results/experiment-05-retinotopic"
CAPTURES = ROOT / "captures/experiment-05"
RAW = DATA / "raw"
MAPPING = RESULTS / "retina-mapping.json"
FRAME_COUNT = 10
STEPS_PER_FRAME = STEPS // FRAME_COUNT
SYNTHETIC = ("black", "white", "left", "right", "vertical", "horizontal", "motion_left", "motion_right")


def capture_sequences():
    out = {}
    for name in NAMES:
        with PokemonEmulator() as game:
            move_to_condition(game, name)
            frames = []
            for i in range(FRAME_COUNT):
                if i: game.tick(2)
                frames.append(game.framebuffer())
            out[name] = frames
    return out


def overlay(frame, records, path):
    image = Image.fromarray(frame, "RGBA").convert("RGB"); draw = ImageDraw.Draw(image)
    for r in records:
        x, y = r.u * 159, r.v * 143
        color = (0, 220, 255) if r.eye_side == "L" else (255, 80, 120)
        draw.ellipse((x-1, y-1, x+1, y+1), fill=color)
    image.save(path)


def eye_space(records, path):
    x = np.asarray([r.x for r in records]); y = np.asarray([r.y for r in records])
    canvas = Image.new("RGB", (1000, 600), "white"); draw = ImageDraw.Draw(canvas)
    for side, xoff, color in (("L", 20, (0,120,180)), ("R", 520, (210,40,80))):
        take = np.asarray([r.eye_side == side for r in records]); xx=x[take]; yy=y[take]
        xx=(xx-xx.min())/max(np.ptp(xx),1); yy=(yy-yy.min())/max(np.ptp(yy),1)
        for a,b in zip(xx,yy): draw.ellipse((xoff+a*450-2, 20+(1-b)*550-2, xoff+a*450+2, 20+(1-b)*550+2),fill=color)
    canvas.save(path)


def encode(frames, records, visual_count, temporal):
    drives=[]; previous=np.full(len(records), .9, np.float32); timings=[]
    uv=np.asarray([(r.u,r.v) for r in records],np.float32)
    for frame in frames:
        t=time.perf_counter(); lum=linear_luminance(frame); lum_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter(); sample=bilinear_sample(lum,uv); sample_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter(); drive=full_eye_drive(visual_count,records,sample,previous,temporal); drive_ms=(time.perf_counter()-t)*1000
        drives.append(drive); timings.append((lum_ms,sample_ms,drive_ms)); previous=sample
    return drives,timings


def groups(brain):
    ct=brain.cell_type.astype(str)
    return {"r1_6":np.flatnonzero(ct=="R1-6"), "lamina":np.flatnonzero(np.isin(ct,["L1","L2","L3","L5"])),
            "t4":np.flatnonzero(np.char.startswith(ct,"T4")), "t5":np.flatnonzero(np.char.startswith(ct,"T5")),
            "visual_projection":brain.cells(["visual_projection"]), "descending":brain.cells(["descending_neuron"]),
            "experiment4_fixed_dn":np.flatnonzero(np.isin(ct,["DNae002","DNg111","DNb01","DNg13","DNae001","DNp04","DNp54","DNp71","DNp31","pIP1","DNpe056"]))}


def run_trial(brain, seed, drives, group_slots, group_sizes, dn_slot):
    brain.reset(seed=seed); totals={k:np.zeros(n,np.int32) for k,n in group_sizes.items()}
    bins=np.zeros((40,group_sizes["descending"]),np.uint8); step_time=agg_time=0.
    for fi in range(FRAME_COUNT):
        drive=None if drives is None else drives[fi]
        for si in range(STEPS_PER_FRAME):
            t=time.perf_counter(); fired=brain.step(eye_drive=drive); step_time+=time.perf_counter()-t
            t=time.perf_counter()
            for name,slot in group_slots.items():
                active=slot[fired]; active=active[active>=0]
                totals[name]+=np.bincount(active,minlength=group_sizes[name]).astype(np.int32)
            active=dn_slot[fired]; active=active[active>=0]
            bins[(fi*STEPS_PER_FRAME+si)//5]+=np.bincount(active,minlength=group_sizes["descending"]).astype(np.uint8)
            agg_time+=time.perf_counter()-t
    return totals,bins,dict(step_ms=step_time*1000/STEPS,aggregation_ms=agg_time*1000/STEPS)


def main():
    RESULTS.mkdir(parents=True,exist_ok=True); CAPTURES.mkdir(parents=True,exist_ok=True)
    ann=RAW/"body-annotations-male-cns-v1.0-minconf-0.5.feather"; edges=RAW/"connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    if MAPPING.exists(): records,diagnostics=load_mapping(MAPPING)
    else:
        records,diagnostics=infer_retina(DATA/"brain.npz",ann,edges); save_mapping(MAPPING,records,diagnostics)
    print(json.dumps(diagnostics,indent=2),flush=True)
    brain=FlyBrain(data=DATA,device="cuda",batch=1)
    if brain.n!=166700 or brain._W.nnz!=25582938: raise RuntimeError("unexpected MaleCNS graph")
    meta=np.load(DATA/"brain.npz"); ids=meta["ids"]
    for r in records:
        if ids[r.brain_index]!=r.flywire_id or brain.visual[r.visual_index]!=r.brain_index: raise RuntimeError("receptor ID/index mismatch")
    pokemon=capture_sequences()
    overlay(pokemon["bedroom"][0],records,CAPTURES/"retina-map.png"); eye_space(records,CAPTURES/"retina-eye-space.png")
    encoded={}; encoder_timing={}
    for mode in ("direct","temporal"):
        for name,frames in pokemon.items(): encoded[f"pokemon_{mode}_{name}"],encoder_timing[f"pokemon_{mode}_{name}"]=encode(frames,records,len(brain.visual),mode=="temporal")
    for name in SYNTHETIC:
        frames=[synthetic_frame(name,i) for i in range(FRAME_COUNT)]
        encoded[f"synthetic_{name}"],encoder_timing[f"synthetic_{name}"]=encode(frames,records,len(brain.visual),True)
    gray=[np.full((144,160,4),128,np.uint8) for _ in range(FRAME_COUNT)]; [f.__setitem__((...,3),255) for f in gray]
    encoded["uniform_gray"],encoder_timing["uniform_gray"]=encode(gray,records,len(brain.visual),False)
    encoded["baseline_none"]=None; encoder_timing["baseline_none"]=[]
    pop=groups(brain); group_sizes={k:len(v) for k,v in pop.items()}
    slots={};
    for name,idx in pop.items():
        slot=np.full(brain.n,-1,np.int32); slot[idx]=np.arange(len(idx)); slots[name]=slot
    dn_slot=slots["descending"]
    brain.reset(seed=0)
    for _ in range(20): brain.step()
    trials={k:[] for k in encoded}; all_bins=np.empty((len(encoded),len(SEEDS),40,group_sizes["descending"]),np.uint8)
    all_stages={k:np.empty((len(encoded),len(SEEDS),n),np.int32) for k,n in group_sizes.items()}
    for sj,seed in enumerate(SEEDS):
        for ci,(condition,drives) in enumerate(encoded.items()):
            totals,bins,timing=run_trial(brain,seed,drives,slots,group_sizes,dn_slot); all_bins[ci,sj]=bins
            for group,values in totals.items(): all_stages[group][ci,sj]=values
            trials[condition].append({"seed":seed,"group_spikes":{k:int(v.sum()) for k,v in totals.items()},**timing})
        print("completed seed",seed,flush=True)
    np.savez_compressed(RESULTS/"stage-counts.npz",dn_bins=all_bins,conditions=np.asarray(list(encoded)),seeds=np.asarray(SEEDS),
                        dn_indices=pop["descending"],dn_ids=ids[pop["descending"]],dn_types=brain.cell_type[pop["descending"]],dn_side=brain.side[pop["descending"]],
                        **{"stage_"+k:v for k,v in all_stages.items()})
    timing_values=[x for values in encoder_timing.values() for x in values]
    output={"model":"flybrain 0.1.0 MaleCNS v1.0","steps":STEPS,"seeds":SEEDS,"conditions":list(encoded),
            "mapping_diagnostics":diagnostics,"group_sizes":group_sizes,"neutral_visual_value":0.0,
            "drive_transforms":{"direct":"linear-light Rec.709 luminance","temporal":{"luminance_gain":.45,"change_gain":1.6,"initial_background":.9}},
            "encoder_performance":{"mean_luminance_ms":float(np.mean([x[0] for x in timing_values])),"mean_sampling_ms":float(np.mean([x[1] for x in timing_values])),"mean_drive_ms":float(np.mean([x[2] for x in timing_values]))},
            "bedroom_state":{"path":"states/bedroom.state","sha256":hashlib.sha256((ROOT/'states/bedroom.state').read_bytes()).hexdigest()},"trials":trials}
    (RESULTS/"trials.json").write_text(json.dumps(output,indent=2)+"\n")
    print("saved Experiment 5 trials",flush=True)

if __name__=="__main__": main()
