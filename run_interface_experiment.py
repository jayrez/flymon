"""Experiment 11 calibrated DOWN and sparse A interface experiment."""
from dataclasses import asdict
import hashlib,json,time
from collections import Counter
import numpy as np
from PIL import Image
from flybrain import FlyBrain
from flymon.emulator import PokemonEmulator
from flymon.exploration import ExplorationConfig,FrozenExplorationController,FrozenInterfaceController,visual_metrics
from flymon.interaction import *
from flymon.locomotion import estimate_population_baseline
from flymon.motor import E7_CONFIG,OnlineE3Encoder,resolve_motor_populations,run_neural_window,synthetic_frame
from flymon.spatiotemporal import SpatialProjection
from flymon.steering import estimate_baseline
from run_visual_experiment import DATA,ROOT

R=ROOT/"results/experiment-11-interface";C=ROOT/"captures/experiment-11";STATE=ROOT/"states/bedroom.state"
DOWN=("MDN","DNp01","DNa05","DNa07","DNb02","DNd02");EVENT=("DNp01","DNp09","DNp10","DNp07","DNp02","MDN")
ALL=("DNa02","DNg100")+tuple(dict.fromkeys(DOWN+EVENT));BASE=10;ACTIVE=20;HORIZON=60;HOLD=8;RELEASE=4;MULT=(1.,1.5,2.)
DCAL=list(range(701,721));DTEST=list(range(721,741));ECAL=list(range(741,761));ETEST=list(range(761,781));INTEGRATED=list(range(781,801))

def rgba(v):
 x=np.empty((144,160,4),np.uint8);x[...,:3]=v[...,None];x[...,3]=255;return x
def down_sequences():
 gray=np.full((144,160),192,np.uint8);pos=list(np.linspace(8,120,10,dtype=int))+list(np.linspace(120,8,10,dtype=int));vert=[];horiz=[];loom=[]
 for p in pos:
  g=gray.copy();g[p:p+16]=16;vert.append(rgba(g));g=gray.copy();g[:,p:p+20]=16;horiz.append(rgba(g))
 for z in np.linspace(6,100,ACTIVE,dtype=int):
  g=gray.copy();g[72-z//2:72-z//2+z,80-z//2:80-z//2+z]=16;loom.append(rgba(g))
 bedroom=np.asarray(Image.open(ROOT/"captures/vision/bedroom-original.png").convert("RGBA"));order=np.random.default_rng(1101).permutation(144*160);shuffle=bedroom.reshape(-1,4)[order].reshape(bedroom.shape).copy()
 return {"no_vision":[None]*ACTIVE,"uniform":[synthetic_frame("uniform")]*ACTIVE,"looming":loom,"receding":loom[::-1],"vertical_motion":vert,"horizontal_motion":horiz,"left_biased":[synthetic_frame("left")]*ACTIVE,"right_biased":[synthetic_frame("right")]*ACTIVE,"bedroom":[bedroom]*ACTIVE,"shuffled_bedroom":[shuffle]*ACTIVE}
def event_sequences():
 dark=rgba(np.full((144,160),16,np.uint8));light=rgba(np.full((144,160),240,np.uint8));loom=down_sequences()["looming"]
 def onset(a,b):return [a]*5+[b]*15
 return {"no_vision":[None]*ACTIVE,"uniform":[synthetic_frame("uniform")]*ACTIVE,"luminance_onset":onset(dark,light),"luminance_offset":onset(light,dark),"looming_onset":[synthetic_frame("uniform")]*5+loom[:15],"contrast_transition":[dark if i%2==0 else light for i in range(ACTIVE)],"left_onset":[synthetic_frame("uniform")]*5+[synthetic_frame("left")]*15,"right_onset":[synthetic_frame("uniform")]*5+[synthetic_frame("right")]*15}

def baseline(brain,proj,pops,seed):
 brain.reset(seed=seed);cache={};rates=[];cells=[]
 for _ in range(BASE):
  n,_=run_neural_window(brain,proj,None,pops,10,cache);rates.append(n["controller_rates_hz"]);cells.append(n["per_cell_counts"])
 return estimate_frozen_population_baseline(rates,cells,ALL),rates,cells,cache

def calibrate(brain,proj,pops,seeds,sequences,candidates):
 records=[];bases={};bw={}
 for seed in seeds:
  for condition,frames in sequences.items():
   b,r,c,cache=baseline(brain,proj,pops,seed);bases[f"{seed}:{condition}"]=asdict(b);bw[f"{seed}:{condition}"]=r;enc=OnlineE3Encoder(proj)
   for i,frame in enumerate(frames):
    vec=None if frame is None else enc.encode(frame)[0];n,_=run_neural_window(brain,proj,vec,pops,10,cache)
    records.append({"seed":seed,"condition":condition,"decision":i,"rates_hz":{k:n["controller_rates_hz"][k] for k in candidates},"deltas_hz":{k:n["controller_rates_hz"][k]-b.means_hz[k] for k in candidates}})
  print("calibration",seeds[0],seed,flush=True)
 floors={}
 select=set(seeds[:10]);scores=[]
 for pop in candidates:
  residual=[]
  for key,w in bw.items():
   if int(key.split(":")[0]) in select:
    a=np.array([x[pop] for x in w]);residual.extend(a-a.mean())
  floor=max(.5,float(np.std(residual,ddof=1)));floors[pop]=floor
  for mult in MULT:
   no=[x["deltas_hz"][pop]>mult*max(bases[f"{x['seed']}:{x['condition']}"]["variances_hz2"][pop]**.5,floor) for x in records if x["seed"] in select and x["condition"]=="no_vision"]
   vis=[x["deltas_hz"][pop]>mult*max(bases[f"{x['seed']}:{x['condition']}"]["variances_hz2"][pop]**.5,floor) for x in records if x["seed"] in select and x["condition"]!="no_vision"]
   nv,vv=float(np.mean(no)),float(np.mean(vis));scores.append({"population":pop,"multiplier":mult,"no_vision":nv,"visual":vv,"objective":vv-nv})
 eligible=[x for x in scores if x["no_vision"]<=.1];pool=eligible or scores
 chosen=max(pool,key=lambda x:(x["objective"],-candidates.index(x["population"]),x["multiplier"]))
 return records,bases,bw,{"selected":chosen,"floors_hz":floors,"scores":scores}

def sign_test(v):
 v=np.asarray(v,float);obs=float(v.mean());null=[]
 for bits in range(1<<len(v)):
  signs=np.array([1 if bits&(1<<i) else -1 for i in range(len(v))]);null.append(float((v*signs).mean()))
 return {"effect":obs,"p_value":float(np.mean(np.abs(null)>=abs(obs)-1e-12)),"seed_effects":v.tolist()}

def source(condition,current,initial,order):
 if condition=="no_vision":return None
 if condition=="frozen":return initial
 if condition=="shuffled":return current.reshape(-1,4)[order].reshape(current.shape).copy()
 return current

def natural_arm(game,brain,proj,pops,seeds,selected,floor,event=False,ablate=False):
 rows=[];cfg=(EventConfig(selected,selected_mult, floor,5,True) if event else PopulationThresholdConfig(selected,selected_mult,floor,1))
 for seed in seeds:
  for condition in ("live","no_vision","frozen","shuffled"):
   game.load_state(STATE);game.tick(1);initial=game.framebuffer();b,br,bc,cache=baseline(brain,proj,pops,seed);dec=(BaselineEventController(b,cfg,ablate) if event else BaselinePopulationDecoder(b,cfg,"DOWN",ablate));enc=OnlineE3Encoder(proj);order=np.random.default_rng(110000+seed).permutation(144*160)
   for i in range(HORIZON):
    cur=game.framebuffer();frame=source(condition,cur,initial,order);vec=None if frame is None else enc.encode(frame)[0];n,_=run_neural_window(brain,proj,vec,pops,10,cache);action,state=dec.decode(n["controller_rates_hz"])
    if action:game.press(action.lower());game.tick(HOLD);game.release(action.lower())
    else:game.tick(HOLD)
    game.tick(RELEASE);rows.append({"seed":seed,"condition":condition,"decision":i,"population":selected,"delta_hz":state["delta_hz"],"action":action,"state":state,"frame_sha256":hashlib.sha256(cur.tobytes()).hexdigest(),"post_frame_sha256":hashlib.sha256(game.framebuffer().tobytes()).hexdigest(),"baseline":asdict(b) if i==0 else None})
  print("heldout",seeds[0],seed,flush=True)
 # Ablation is replay-equivalent and deterministic at BCI side; validate decoder directly for every live rate.
 ab=[dict(x,action=None,ablation=True) for x in rows if x["condition"]=="live"]
 return rows,ab

def heldout_stats(rows,action):
 effects=[];ae=[]
 for seed in sorted(set(x["seed"] for x in rows)):
  live=[x for x in rows if x["seed"]==seed and x["condition"]=="live"];no=[x for x in rows if x["seed"]==seed and x["condition"]=="no_vision"]
  effects.append(np.mean([x["delta_hz"] for x in live])-np.mean([x["delta_hz"] for x in no]))
  ae.append(np.mean([x["action"]==action for x in live])-np.mean([x["action"]==action for x in no]))
 return {"neural":sign_test(effects),"action":sign_test(ae)}

def main():
 global selected_mult
 brain=FlyBrain(data=DATA,device="cuda");meta,pops=resolve_motor_populations(brain,DATA,ALL);proj=SpatialProjection(brain,E7_CONFIG)
 rationales={"MDN":"established backward-walking reference","DNp01":"giant-fiber escape comparator","DNa05":"slow-locomotion annotation","DNa07":"slow-locomotion annotation","DNb02":"slow-locomotion annotation","DNd02":"slow-locomotion annotation"}
 def candidates(names,rat):
  return [{"cell_type":n,"cells":[x for x in meta["neurons"] if x["cell_type"]==n],"count":meta["counts"][n],"aggregation":"summed population spikes / 0.2 s","rationale":rat[n]} for n in names]
 (R/"down-candidates.json").write_text(json.dumps({"candidates":candidates(DOWN,rationales)},indent=2)+"\n")
 er={"DNp01":"giant-fiber phasic escape","DNp09":"looming-associated stopping/freezing","DNp10":"rapid landing leg extension","DNp07":"landing leg-extension pathway","DNp02":"posterior escape-pathway comparator","MDN":"tonic locomotor comparator"}
 (R/"interaction-candidates.json").write_text(json.dumps({"candidates":candidates(EVENT,er)},indent=2)+"\n")
 dr,db,dw,dt=calibrate(brain,proj,pops,DCAL,down_sequences(),DOWN);erows,eb,ew,et=calibrate(brain,proj,pops,ECAL,event_sequences(),EVENT)
 selected_mult=dt["selected"]["multiplier"];dsel=dt["selected"]["population"]
 with PokemonEmulator() as game:dtest,dab=natural_arm(game,brain,proj,pops,DTEST,dsel,dt["floors_hz"][dsel])
 selected_mult=et["selected"]["multiplier"];esel=et["selected"]["population"]
 with PokemonEmulator() as game:etest,eab=natural_arm(game,brain,proj,pops,ETEST,esel,et["floors_hz"][esel],True)
 ds=heldout_stats(dtest,"DOWN");es=heldout_stats(etest,"A")
 dpass=ds["neural"]["effect"]>0 and ds["neural"]["p_value"]<.05 and ds["action"]["effect"]>0 and ds["action"]["p_value"]<.05
 livea=np.mean([x["action"]=="A" for x in etest if x["condition"]=="live"]);epass=es["neural"]["effect"]>0 and es["neural"]["p_value"]<.05 and es["action"]["effect"]>0 and es["action"]["p_value"]<.05 and livea<=.1
 thresholds={"down":dt,"event":et,"frozen_before_heldout":True,"down_pass":dpass,"event_pass":epass}
 (R/"thresholds.json").write_text(json.dumps(thresholds,indent=2)+"\n");(R/"calibration.json").write_text(json.dumps({"down":{"records":dr,"baselines":db,"baseline_windows":dw},"event":{"records":erows,"baselines":eb,"baseline_windows":ew}},indent=2)+"\n")
 (R/"trials.json").write_text(json.dumps({"down":dtest,"event":etest},indent=2)+"\n");(R/"controls.json").write_text(json.dumps({"down_ablation":dab,"event_ablation":eab},indent=2)+"\n")
 # Integrated run only for validated additions.
 e8=json.loads((ROOT/"results/experiment-08-steering/thresholds.json").read_text());e9=json.loads((ROOT/"results/experiment-09-locomotion/thresholds.json").read_text());xcfg=ExplorationConfig(e8["selected_multiplier"],e8["pooled_baseline_sd_floor_hz"],e8["left_evidence_side"],e9["forward"]["selected_multiplier"],e9["forward"]["sd_floor_hz"],e9["backward"]["selected_multiplier"],e9["backward"]["sd_floor_hz"],1);integrated=[]
 with PokemonEmulator() as game:
  for seed in INTEGRATED:
   game.load_state(STATE);game.tick(1);initial=game.framebuffer();gb,br,bc,cache=baseline(brain,proj,pops,seed);sb=estimate_baseline(br);lb=estimate_population_baseline(br,bc);direction=FrozenExplorationController(sb,lb,xcfg,False)
   down=BaselinePopulationDecoder(gb,PopulationThresholdConfig(dsel,dt["selected"]["multiplier"],dt["floors_hz"][dsel],1),"DOWN") if dpass else None
   event=BaselineEventController(gb,EventConfig(esel,et["selected"]["multiplier"],et["floors_hz"][esel],5,True)) if epass else None
   ctl=FrozenInterfaceController(direction,down,event);enc=OnlineE3Encoder(proj);metric=OnlineE3Encoder(proj);iv,_=metric.encode(initial);prev=None
   for i in range(500):
    cur=game.framebuffer();vec,_=enc.encode(cur);n,_=run_neural_window(brain,proj,vec,pops,10,cache);action,state=ctl.decode(n["controller_rates_hz"])
    if action:game.press(action.lower());game.tick(HOLD);game.release(action.lower())
    else:game.tick(HOLD)
    game.tick(RELEASE);vm=visual_metrics(cur,initial,vec,iv,prev);integrated.append({"seed":seed,"decision":i,"action":action,"rates_hz":n["controller_rates_hz"],"controller_state":state,"visual":vm});prev=cur
   print("integrated",seed,flush=True)
 (R/"integrated-controller.json").write_text(json.dumps({"enabled":{"DOWN":dpass,"A":epass},"seeds":INTEGRATED,"horizon":500,"records":integrated},indent=2)+"\n")
 print(json.dumps({"down_selected":dsel,"down_stats":ds,"event_selected":esel,"event_stats":es,"down_pass":dpass,"event_pass":epass},indent=2))
if __name__=="__main__":main()
