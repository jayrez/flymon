"""Experiment 9 controlled DNg100/MDN calibration and threshold freezing."""
from dataclasses import asdict
import json,time
import numpy as np
from PIL import Image
from flybrain import FlyBrain
from flymon.locomotion import estimate_population_baseline
from flymon.motor import E7_CONFIG,OnlineE3Encoder,run_neural_window,synthetic_frame,verify_motor_neurons
from flymon.spatiotemporal import SpatialProjection
from run_visual_experiment import DATA,ROOT

RESULTS=ROOT/"results/experiment-09-locomotion";CAPTURES=ROOT/"captures/experiment-09"
SEEDS=list(range(501,521));SELECT=set(range(501,511));VALIDATE=set(range(511,521));BASE=10;ACTIVE=20;CAND=(1.,1.5,2.)

def rgba(gray):
    x=np.empty((144,160,4),np.uint8);x[...,:3]=gray[...,None];x[...,3]=255;return x
def sequences():
    gray=np.full((144,160),192,np.uint8);out={"no_vision":[None]*ACTIVE,"uniform":[synthetic_frame("uniform")]*ACTIVE,
      "left_biased":[synthetic_frame("left")]*ACTIVE,"right_biased":[synthetic_frame("right")]*ACTIVE}
    vertical=[];horizontal=[];loom=[]
    positions=list(np.linspace(8,120,10,dtype=int))+list(np.linspace(120,8,10,dtype=int))
    for p in positions:
        g=gray.copy();g[p:p+16,:]=16;vertical.append(rgba(g));g=gray.copy();g[:,p:p+20]=16;horizontal.append(rgba(g))
    for size in np.linspace(6,100,ACTIVE,dtype=int):
        g=gray.copy();y0=72-size//2;x0=80-size//2;g[max(0,y0):min(144,y0+size),max(0,x0):min(160,x0+size)]=16;loom.append(rgba(g))
    out.update(vertical_motion=vertical,horizontal_motion=horizontal,looming=loom,receding=list(reversed(loom)))
    out["bedroom"]=[np.asarray(Image.open(ROOT/"captures/vision/bedroom-original.png").convert("RGBA"))]*ACTIVE
    out["menu"]=[np.asarray(Image.open(ROOT/"captures/vision/new_game_menu-original.png").convert("RGBA"))]*ACTIVE
    return out

def run_condition(brain,projection,pops,seed,name,frames):
    brain.reset(seed=seed);cache={};base_rates=[];base_cells=[]
    for _ in range(BASE):
        n,_=run_neural_window(brain,projection,None,pops,E7_CONFIG.neural_steps_per_frame,cache);base_rates.append(n["controller_rates_hz"]);base_cells.append(n["per_cell_counts"])
    baseline=estimate_population_baseline(base_rates,base_cells);encoder=OnlineE3Encoder(projection);rows=[]
    for i,frame in enumerate(frames):
        if frame is None:vec=None;enc={"encoder_sha256":None,"change_mean":None}
        else:vec,enc=encoder.encode(frame)
        n,vp=run_neural_window(brain,projection,vec,pops,E7_CONFIG.neural_steps_per_frame,cache);rates=n["controller_rates_hz"]
        rows.append({"seed":seed,"condition":name,"decision":i,"rates_hz":rates,"per_cell_counts":n["per_cell_counts"],
          "forward_delta_hz":rates["DNg100"]-baseline.dng100_mean_hz,"backward_delta_hz":rates["MDN"]-baseline.mdn_mean_hz,
          "DNa02_L_minus_R_hz":rates["DNa02_L"]-rates["DNa02_R"],"DNp01_hz":rates["DNp01"],"encoder":enc,"visual_projection_spikes":vp})
    return asdict(baseline),base_rates,rows

def sign_test(values):
    v=np.asarray(values,float);obs=float(v.mean());null=[]
    for bits in range(1<<len(v)):
        s=np.array([1 if bits&(1<<i) else -1 for i in range(len(v))]);null.append(float((v*s).mean()))
    return {"effect_hz":obs,"p_value":float(sum(abs(x)>=abs(obs)-1e-12 for x in null)/len(null)),"enumerations":len(null)}

def choose(rows,baselines,base_windows,pop,key,floor):
    scores=[]
    for mult in CAND:
        no=[];visual=[]
        for r in rows:
            if r["seed"] not in SELECT:continue
            b=baselines[f"{r['seed']}:{r['condition']}"];sd=max(b[key+"_variance_hz2"]**.5,floor);cross=r[("forward" if pop=="DNg100" else "backward")+"_delta_hz"]>mult*sd
            (no if r["condition"]=="no_vision" else visual).append(cross)
        nv=float(np.mean(no));vv=float(np.mean(visual));scores.append({"multiplier":mult,"visual_crossing":vv,"no_vision_crossing":nv,"objective":vv-nv})
    eligible=[x for x in scores if x["no_vision_crossing"]<=.10]
    picked=max(eligible,key=lambda x:(x["objective"],x["multiplier"])) if eligible else min(scores,key=lambda x:(x["no_vision_crossing"],-x["objective"],-x["multiplier"]))
    return scores,picked

def main():
    RESULTS.mkdir(parents=True,exist_ok=True);CAPTURES.mkdir(parents=True,exist_ok=True)
    brain=FlyBrain(data=DATA,device="cuda");meta,pops=verify_motor_neurons(brain,DATA);projection=SpatialProjection(brain,E7_CONFIG);seq=sequences()
    records=[];baselines={};base_windows={}
    for seed in SEEDS:
        for name,frames in seq.items():
            b,w,r=run_condition(brain,projection,pops,seed,name,frames);baselines[f"{seed}:{name}"]=b;base_windows[f"{seed}:{name}"]=w;records+=r
        print("locomotion calibration seed",seed,flush=True)
    floors={}
    for pop,label in (("DNg100","dng100"),("MDN","mdn")):
        residual=[]
        for k,w in base_windows.items():
            if int(k.split(":")[0]) in SELECT:
                x=np.array([z[pop] for z in w]);residual.extend(x-x.mean())
        floors[pop]=max(.5,float(np.std(residual,ddof=1)))
    fs,fp=choose(records,baselines,base_windows,"DNg100","dng100",floors["DNg100"]);bs,bp=choose(records,baselines,base_windows,"MDN","mdn",floors["MDN"])
    validation={}
    for pop,delta in (("DNg100","forward_delta_hz"),("MDN","backward_delta_hz")):
        vals=[]
        for seed in sorted(VALIDATE):
            visual=np.mean([r[delta] for r in records if r["seed"]==seed and r["condition"]!="no_vision"]);no=np.mean([r[delta] for r in records if r["seed"]==seed and r["condition"]=="no_vision"]);vals.append(visual-no)
        validation[pop]=sign_test(vals)|{"seed_effects_hz":vals}
    thresholds={"calibration_seeds":SEEDS,"selection_seeds":sorted(SELECT),"validation_seeds":sorted(VALIDATE),"primary_seeds":list(range(521,541)),
      "candidates":list(CAND),"forward":{"scores":fs,"selected_multiplier":fp["multiplier"],"sd_floor_hz":floors["DNg100"]},
      "backward":{"scores":bs,"selected_multiplier":bp["multiplier"],"sd_floor_hz":floors["MDN"]},"validation":validation,"frozen_before_primary":True}
    motor={"model":"MaleCNS v1.0","populations":{k:[x for x in meta["neurons"] if x["cell_type"]==k] for k in ("DNg100","MDN","DNp01")},
      "aggregation":{"DNg100":"sum spikes across 2 cells / 0.2 s","MDN":"sum spikes across 4 cells / 0.2 s","DNp01":"diagnostic only"}}
    (RESULTS/"motor-populations.json").write_text(json.dumps(motor,indent=2)+"\n");(RESULTS/"thresholds.json").write_text(json.dumps(thresholds,indent=2)+"\n")
    (RESULTS/"calibration.json").write_text(json.dumps({"metadata":{"conditions":list(seq),"baseline_windows":BASE,"active_windows":ACTIVE},"baselines":baselines,"baseline_windows":base_windows,"records":records},indent=2)+"\n")
    print(json.dumps(thresholds,indent=2),flush=True)
if __name__=="__main__":main()
