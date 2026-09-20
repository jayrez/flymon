"""Experiment 12 focused DNp01 transparent event-decoder validation."""
from __future__ import annotations
from dataclasses import asdict
import hashlib,json,math
import numpy as np
from PIL import Image
from flybrain import FlyBrain
from flymon.emulator import PokemonEmulator
from flymon.exploration import ExplorationConfig,FrozenExplorationController,FrozenInterfaceController,visual_metrics
from flymon.interaction import EventDecoderConfig,TransparentEventController,estimate_frozen_population_baseline
from flymon.locomotion import estimate_population_baseline
from flymon.motor import E7_CONFIG,OnlineE3Encoder,resolve_motor_populations,run_neural_window,synthetic_frame
from flymon.spatiotemporal import SpatialProjection
from flymon.steering import estimate_baseline
from run_visual_experiment import DATA,ROOT

R=ROOT/"results/experiment-12-dnp01-event";C=ROOT/"captures/experiment-12";STATE=ROOT/"states/bedroom.state"
CAL=list(range(801,821));TEST=list(range(821,841));INTEGRATED=list(range(841,861))
POPS=("DNa02","DNg100","MDN","DNp01");BASE=10;CONTROLLED=20;HORIZON=60;HOLD=8;RELEASE=4

def rgba(value):
 x=np.empty((144,160,4),np.uint8);x[...,:3]=value[...,None];x[...,3]=255;return x

def controlled_sequences():
 dark=rgba(np.full((144,160),16,np.uint8));light=rgba(np.full((144,160),240,np.uint8))
 gray=np.full((144,160),192,np.uint8);loom=[]
 for z in np.linspace(6,100,CONTROLLED,dtype=int):
  g=gray.copy();g[72-z//2:72-z//2+z,80-z//2:80-z//2+z]=16;loom.append(rgba(g))
 onset=lambda a,b:[a]*5+[b]*15
 out={"controlled_no_vision":[None]*CONTROLLED,"uniform":[synthetic_frame("uniform")]*CONTROLLED,
 "luminance_onset":onset(dark,light),"luminance_offset":onset(light,dark),
 "looming_onset":[synthetic_frame("uniform")]*5+loom[:15],
 "left_onset":[synthetic_frame("uniform")]*5+[synthetic_frame("left")]*15,
 "right_onset":[synthetic_frame("uniform")]*5+[synthetic_frame("right")]*15}
 for name,file in (("menu","new_game_menu-original.png"),("dialogue","oak_dialogue-original.png"),("title","title-original.png"),("intro","intro-original.png")):
  p=ROOT/"captures/vision"/file
  if p.exists():out["secondary_"+name]=[np.asarray(Image.open(p).convert("RGBA"))]*CONTROLLED
 return out

def baseline(brain,proj,pops,seed):
 brain.reset(seed=seed);cache={};rates=[];cells=[]
 for _ in range(BASE):
  n,_=run_neural_window(brain,proj,None,pops,10,cache);rates.append(n["controller_rates_hz"]);cells.append(n["per_cell_counts"])
 return estimate_frozen_population_baseline(rates,cells,POPS),rates,cells,cache

def cell_record(n,b,seconds=.2):
 counts=n["per_cell_counts"]["DNp01"];rates=[float(x/seconds) for x in counts];means=b.per_neuron_means_hz["DNp01"];delta=[rates[i]-means[i] for i in range(2)]
 return {"counts":counts,"rates_hz":rates,"baseline_means_hz":list(means),"deltas_hz":delta,
  "aggregates_hz":{"sum":sum(rates),"mean":float(np.mean(rates)),"max":max(rates),"R_minus_L":rates[0]-rates[1]},
  "aggregate_deltas_hz":{"sum":sum(rates)-b.means_hz["DNp01"],"mean":float(np.mean(delta)),"max":max(rates)-max(means),"R_minus_L":delta[0]-delta[1]}}

def transform(condition,current,initial,order):
 if condition in ("bedroom_no_vision","no_vision"):return None
 if condition=="bedroom_frozen":return initial
 if condition=="bedroom_shuffled":return current.reshape(-1,4)[order].reshape(current.shape).copy()
 return current

def acquire_calibration(brain,proj,pops):
 records=[];baselines={};baseline_windows={}
 # Controlled and existing natural static sequences.
 for seed in CAL:
  for condition,frames in controlled_sequences().items():
   b,br,bc,cache=baseline(brain,proj,pops,seed);baselines[f"{seed}:{condition}"]=asdict(b);baseline_windows[f"{seed}:{condition}"]=br;enc=OnlineE3Encoder(proj)
   for i,frame in enumerate(frames):
    vec=None if frame is None else enc.encode(frame)[0];n,_=run_neural_window(brain,proj,vec,pops,10,cache)
    records.append({"seed":seed,"condition":condition,"decision":i,"cells":cell_record(n,b)})
  print("controlled calibration",seed,flush=True)
 # Natural bedroom is advanced without button input; outcome cannot enter selection.
 with PokemonEmulator() as game:
  for seed in CAL:
   for condition in ("bedroom_live","bedroom_no_vision","bedroom_frozen","bedroom_shuffled"):
    game.load_state(STATE);game.tick(1);initial=game.framebuffer();order=np.random.default_rng(120000+seed).permutation(144*160)
    b,br,bc,cache=baseline(brain,proj,pops,seed);baselines[f"{seed}:{condition}"]=asdict(b);baseline_windows[f"{seed}:{condition}"]=br;enc=OnlineE3Encoder(proj)
    for i in range(HORIZON):
     cur=game.framebuffer();src=transform(condition,cur,initial,order);vec=None if src is None else enc.encode(src)[0];n,_=run_neural_window(brain,proj,vec,pops,10,cache)
     records.append({"seed":seed,"condition":condition,"decision":i,"cells":cell_record(n,b),"frame_sha256":hashlib.sha256(cur.tobytes()).hexdigest()});game.tick(HOLD+RELEASE)
   print("natural calibration",seed,flush=True)
 return records,baselines,baseline_windows

def candidate_grid():
 out=[];order=("rising_edge","level","recovery","derivative","cumulative")
 specs={"rising_edge":([1],[1,1.5,2]),"level":([1,3,5],[-.5,0,.5]),"recovery":([3,5],[.5,1,1.5]),"derivative":([1,3],[.5,1,1.5]),"cumulative":([3,5],[0,.5,1])}
 for family in order:
  for window in specs[family][0]:
   for threshold in specs[family][1]:out.append({"strategy":family,"window":window,"threshold_z":threshold})
 return out

def replay(records,baselines,spec,floor,shuffle=False,drop=None,ablated=False):
 outputs=[]
 for seed,condition in sorted(set((x["seed"],x["condition"]) for x in records)):
  rows=[x for x in records if x["seed"]==seed and x["condition"]==condition]
  if shuffle:rows=[rows[i] for i in np.random.default_rng(1200+seed).permutation(len(rows))]
  b=baselines[f"{seed}:{condition}"];scale=max(math.sqrt(max(0,b["variances_hz2"]["DNp01"])),floor)
  cfg=EventDecoderConfig(spec["strategy"],"sum",b["means_hz"]["DNp01"],scale,spec["threshold_z"],spec["window"],.5,5);ctl=TransparentEventController(cfg,ablated=ablated)
  for j,row in enumerate(rows):
   cells=list(row["cells"]["rates_hz"])
   if drop is not None:cells[drop]=0.
   action,state=ctl.decode({"DNp01_cells_hz":cells});outputs.append({"seed":seed,"condition":condition,"decision":j,"action":action,"state":state})
 return outputs

def select_decoder(records,baselines,base_windows):
 residual=[]
 for key,w in base_windows.items():
  x=np.asarray([z["DNp01"] for z in w]);residual.extend(x-x.mean())
 floor=max(.5,float(np.std(residual,ddof=1)));scores=[]
 controlled={"luminance_onset","luminance_offset","looming_onset","left_onset","right_onset"}
 for rank,spec in enumerate(candidate_grid()):
  out=replay(records,baselines,spec,floor);seed_stats=[]
  for seed in CAL:
   def rate(conds):
    x=[r for r in out if r["seed"]==seed and r["condition"] in conds];return float(np.mean([r["action"]=="A" for r in x]))
   live=rate({"bedroom_live"});no=rate({"bedroom_no_vision"});ctrl=rate(controlled)
   suppress=max([r["state"]["refractory_suppressions"] for r in out if r["seed"]==seed],default=0)/max(1,len([r for r in out if r["seed"]==seed]))
   seed_stats.append((live,no,ctrl,suppress))
  a=np.asarray(seed_stats);live,no,ctrl,supp=a.mean(axis=0)
  max_burst=0
  for seed in CAL:
   for condition in sorted(set(r["condition"] for r in out if r["seed"]==seed)):
    run=0
    for row in (r for r in out if r["seed"]==seed and r["condition"]==condition):
     run=run+1 if row["action"]=="A" else 0;max_burst=max(max_burst,run)
  objective=live-no+.25*(ctrl-no)-2*max(0,no-.05)-max(0,live-.10)-.01*supp
  scores.append(spec|{"live_rate":float(live),"no_vision_rate":float(no),"controlled_rate":float(ctrl),"suppression_fraction":float(supp),"max_event_burst":max_burst,"objective":float(objective),"eligible":bool(no<=.05 and live<=.10 and max_burst<=1),"grid_rank":rank})
 eligible=[x for x in scores if x["eligible"]];pool=eligible or scores
 family_order={x:i for i,x in enumerate(("rising_edge","level","recovery","derivative","cumulative"))}
 selected=max(pool,key=lambda x:(x["objective"],-family_order[x["strategy"]],-x["window"],x["threshold_z"]))
 return floor,scores,selected,replay(records,baselines,selected,floor)

def sign_test(values):
 v=np.asarray(values,float);obs=float(v.mean());extreme=0;total=1<<len(v);chunk=32768
 for start in range(0,total,chunk):
  bits=np.arange(start,min(total,start+chunk),dtype=np.uint32)[:,None];signs=1-2*((bits>>np.arange(len(v),dtype=np.uint32))&1).astype(np.int8)
  null=(signs*v).mean(axis=1);extreme+=int(np.count_nonzero(np.abs(null)>=abs(obs)-1e-12))
 return {"effect":obs,"p_value":extreme/total,"seed_effects":v.tolist(),"permutations":total}

def heldout_trial(game,brain,proj,pops,seed,condition,spec,floor):
 game.load_state(STATE);game.tick(1);initial=game.framebuffer();order=np.random.default_rng(120000+seed).permutation(144*160)
 b,br,bc,cache=baseline(brain,proj,pops,seed);scale=max(math.sqrt(max(0,b.variances_hz2["DNp01"])),floor)
 cfg=EventDecoderConfig(spec["strategy"],"sum",b.means_hz["DNp01"],scale,spec["threshold_z"],spec["window"],.5,5);ctl=TransparentEventController(cfg);enc=OnlineE3Encoder(proj);metric=OnlineE3Encoder(proj);iv,_=metric.encode(initial);previous=None;rows=[]
 for i in range(HORIZON):
  cur=game.framebuffer();src=transform(condition,cur,initial,order)
  if src is None:vec=np.zeros(len(proj.ids),np.float32);inject=None
  else:vec,_=enc.encode(src);inject=vec
  n,_=run_neural_window(brain,proj,inject,pops,10,cache);cells=cell_record(n,b);action,state=ctl.decode({"DNp01_cells_hz":cells["rates_hz"]})
  if action:game.press("a");game.tick(HOLD);game.release("a")
  else:game.tick(HOLD)
  game.tick(RELEASE);post=game.framebuffer();vm=visual_metrics(cur,initial,vec,iv,previous)
  rows.append({"seed":seed,"condition":condition,"decision":i,"cells":cells,"action":action,"controller_state":state,"visual":vm,"post_frame_sha256":hashlib.sha256(post.tobytes()).hexdigest(),"baseline":asdict(b) if i==0 else None});previous=cur
 return rows

def compare(rows,key):
 effects=[]
 for seed in TEST:
  live=[x for x in rows if x["seed"]==seed and x["condition"]=="bedroom_live"];other=[x for x in rows if x["seed"]==seed and x["condition"]==key]
  effects.append(np.mean([x["action"]=="A" for x in live])-np.mean([x["action"]=="A" for x in other]))
 return sign_test(effects)

def neural_compare(rows):
 effects=[]
 for seed in TEST:
  def mean(c):return np.mean([x["cells"]["aggregate_deltas_hz"]["sum"] for x in rows if x["seed"]==seed and x["condition"]==c])
  effects.append(mean("bedroom_live")-mean("bedroom_no_vision"))
 return sign_test(effects)

def main():
 R.mkdir(parents=True,exist_ok=True);C.mkdir(parents=True,exist_ok=True)
 brain=FlyBrain(data=DATA,device="cuda");meta,pops=resolve_motor_populations(brain,DATA,POPS);proj=SpatialProjection(brain,E7_CONFIG)
 cells=[x for x in meta["neurons"] if x["cell_type"]=="DNp01"]
 cal,bases,bw=acquire_calibration(brain,proj,pops);floor,scores,selected,selected_out=select_decoder(cal,bases,bw)
 thresholds={"calibration_seeds":CAL,"heldout_seeds":TEST,"integrated_seeds":INTEGRATED,"baseline_windows":BASE,"neural_window_seconds":.2,"sd_floor_hz":floor,"aggregation":"sum","hysteresis_z":.5,"refractory_decisions":5,"a_hold_frames":HOLD,"a_release_frames":RELEASE,"selected":selected,"grid_scores":scores,"frozen_before_heldout":True}
 (R/"calibration.json").write_text(json.dumps({"DNp01_cells":cells,"records":cal,"baselines":bases,"baseline_windows":bw,"selected_decoder_outputs":selected_out},indent=2)+"\n");(R/"thresholds.json").write_text(json.dumps(thresholds,indent=2)+"\n")
 trials=[]
 with PokemonEmulator() as game:
  for seed in TEST:
   for condition in ("bedroom_live","bedroom_no_vision","bedroom_frozen","bedroom_shuffled"):trials+=heldout_trial(game,brain,proj,pops,seed,condition,selected,floor)
   print("heldout",seed,flush=True)
 neural=neural_compare(trials);event_no=compare(trials,"bedroom_no_vision");event_frozen=compare(trials,"bedroom_frozen");event_shuffle=compare(trials,"bedroom_shuffled")
 # Offline falsifications preserve each trial's marginal cell-rate distribution.
 raw=[];hb={}
 for x in trials:
  raw.append({"seed":x["seed"],"condition":x["condition"],"decision":x["decision"],"cells":{"rates_hz":x["cells"]["rates_hz"]}})
  if x["decision"]==0:hb[f"{x['seed']}:{x['condition']}"]=x["baseline"]
 temporal=replay(raw,hb,selected,floor,shuffle=True);drop0=replay(raw,hb,selected,floor,drop=0);drop1=replay(raw,hb,selected,floor,drop=1);ablated=replay(raw,hb,selected,floor,ablated=True)
 live=[x for x in trials if x["condition"]=="bedroom_live"];live_rate=float(np.mean([x["action"]=="A" for x in live]));ablation_events=sum(x["action"]=="A" for x in ablated)
 passed=neural["effect"]>0 and neural["p_value"]<.05 and event_no["effect"]>0 and event_no["p_value"]<.05 and live_rate<=.10 and ablation_events==0
 (R/"heldout-trials.json").write_text(json.dumps({"metadata":{"seeds":TEST,"conditions":["bedroom_live","bedroom_no_vision","bedroom_frozen","bedroom_shuffled"],"horizon":HORIZON,"DNp01_cells":cells},"records":trials},indent=2)+"\n")
 (R/"controls.json").write_text(json.dumps({"ablation":ablated,"temporal_shuffle":temporal,"drop_cell_0":drop0,"drop_cell_1":drop1},indent=2)+"\n")
 preliminary={"continuous":neural,"live_vs_no_vision":event_no,"live_vs_frozen":event_frozen,"live_vs_shuffled":event_shuffle,"live_rate":live_rate,"pass":passed}
 # Conditional integrated trial.
 integrated=[]
 if passed:
  e8=json.loads((ROOT/"results/experiment-08-steering/thresholds.json").read_text());e9=json.loads((ROOT/"results/experiment-09-locomotion/thresholds.json").read_text())
  xcfg=ExplorationConfig(e8["selected_multiplier"],e8["pooled_baseline_sd_floor_hz"],e8["left_evidence_side"],e9["forward"]["selected_multiplier"],e9["forward"]["sd_floor_hz"],e9["backward"]["selected_multiplier"],e9["backward"]["sd_floor_hz"],1)
  with PokemonEmulator() as game:
   for seed in INTEGRATED:
    game.load_state(STATE);game.tick(1);initial=game.framebuffer();gb,br,bc,cache=baseline(brain,proj,pops,seed);sb=estimate_baseline(br);lb=estimate_population_baseline(br,bc);scale=max(math.sqrt(max(0,gb.variances_hz2["DNp01"])),floor)
    ev=TransparentEventController(EventDecoderConfig(selected["strategy"],"sum",gb.means_hz["DNp01"],scale,selected["threshold_z"],selected["window"],.5,5));ctl=FrozenInterfaceController(FrozenExplorationController(sb,lb,xcfg,False),None,ev);enc=OnlineE3Encoder(proj);metric=OnlineE3Encoder(proj);iv,_=metric.encode(initial);prev=None
    for i in range(500):
     cur=game.framebuffer();vec,_=enc.encode(cur);n,_=run_neural_window(brain,proj,vec,pops,10,cache);cells=cell_record(n,gb);rates=n["controller_rates_hz"]|{"DNp01_cells_hz":cells["rates_hz"]};action,state=ctl.decode(rates)
     if action:game.press(action.lower());game.tick(HOLD);game.release(action.lower())
     else:game.tick(HOLD)
     game.tick(RELEASE);integrated.append({"seed":seed,"decision":i,"action":action,"controller_state":state,"cells":cells,"visual":visual_metrics(cur,initial,vec,iv,prev)});prev=cur
    print("integrated",seed,flush=True)
 (R/"integrated-controller.json").write_text(json.dumps({"enabled":passed,"seeds":INTEGRATED if passed else [],"horizon":500,"records":integrated},indent=2)+"\n")
 print(json.dumps({"selected":selected,"floor":floor,"heldout":preliminary},indent=2))
if __name__=="__main__":main()
