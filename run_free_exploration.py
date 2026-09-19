"""Experiment 10 long-horizon autonomous pixel→MaleCNS→DN exploration."""
from dataclasses import asdict
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from flybrain import FlyBrain
from flymon.emulator import PokemonEmulator
from flymon.exploration import ExplorationConfig,FrozenExplorationController,visual_metrics
from flymon.locomotion import estimate_population_baseline
from flymon.motor import E7_CONFIG,OnlineE3Encoder,run_neural_window,verify_motor_neurons
from flymon.spatiotemporal import SpatialProjection
from flymon.steering import estimate_baseline
from run_visual_experiment import DATA,ROOT

RESULTS=ROOT/"results/experiment-10-exploration";CAPTURES=ROOT/"captures/experiment-10";STATE=ROOT/"states/bedroom.state"
PRIMARY=list(range(601,621));SECONDARY=list(range(621,641));HORIZON=300;LONG_SEED=650;LONG_HORIZON=1000;BASE=10;HOLD=8;RELEASE=4

def baseline(brain,projection,pops,seed):
    brain.reset(seed=seed);cache={};rates=[];cells=[]
    for _ in range(BASE):
        n,_=run_neural_window(brain,projection,None,pops,E7_CONFIG.neural_steps_per_frame,cache);rates.append(n["controller_rates_hz"]);cells.append(n["per_cell_counts"])
    return estimate_baseline(rates),estimate_population_baseline(rates,cells),rates,cells,cache

def sheet(frames,path):
    ims=[Image.fromarray(x,"RGBA").convert("RGB") for x in frames];canvas=Image.new("RGB",(160*len(ims),144),"white")
    for i,im in enumerate(ims):canvas.paste(im,(160*i,0))
    canvas.save(path)

def trial(game,brain,projection,pops,cfg,seed,condition,horizon,down=False,repeat=0,long=False):
    game.load_state(STATE);game.tick(1);initial=game.framebuffer();sb,lb,base_rates,base_cells,cache=baseline(brain,projection,pops,seed)
    controller=FrozenExplorationController(sb,lb,cfg,down);encoder=OnlineE3Encoder(projection);metric_encoder=OnlineE3Encoder(projection)
    if condition=="no_vision":initial_vector=np.zeros(len(projection.ids),np.float32)
    else:initial_vector,_=metric_encoder.encode(initial)
    records=[];previous=None;start=time.perf_counter();selected_frames=[];max_frame=initial.copy();max_dist=-1
    checkpoints=set(np.linspace(0,horizon-1,5,dtype=int).tolist())
    for i in range(horizon):
        loop=time.perf_counter();current=game.framebuffer();source=None if condition=="no_vision" else initial if condition=="frozen_vision" else current
        te=time.perf_counter()
        if source is None:vector=np.zeros(len(projection.ids),np.float32);enc={"encoder_sha256":None,"change_mean":None};inject=None
        else:vector,enc=encoder.encode(source);inject=vector
        encode_s=time.perf_counter()-te;tn=time.perf_counter();n,vp=run_neural_window(brain,projection,inject,pops,E7_CONFIG.neural_steps_per_frame,cache);neural_s=time.perf_counter()-tn
        tc=time.perf_counter();action,state=controller.decode(n["controller_rates_hz"]);decode_s=time.perf_counter()-tc;tg=time.perf_counter()
        if action:game.press(action.lower());game.tick(HOLD);game.release(action.lower())
        else:game.tick(HOLD)
        game.tick(RELEASE);post=game.framebuffer();game_s=time.perf_counter()-tg
        vm=visual_metrics(current,initial,vector,initial_vector,previous);post_mae=float(np.mean(np.abs(post[...,:3].astype(float)-initial[...,:3]))/255)
        if vm["frame_to_initial_mae"]>max_dist:max_dist=vm["frame_to_initial_mae"];max_frame=current.copy()
        transition=vm["frame_to_previous_mae"]>.20
        if transition:
            Image.fromarray(current,"RGBA").save(CAPTURES/f"transition-{condition}-{seed}-{repeat}-{i:04d}.png")
            Image.fromarray(previous if previous is not None else initial,"RGBA").save(CAPTURES/f"transition-before-{condition}-{seed}-{repeat}-{i:04d}.png")
        if i in checkpoints:selected_frames.append(current.copy())
        records.append({"trial":f"{condition}-{seed}-{repeat}-{down}","seed":seed,"condition":condition,"repeat":repeat,"down_enabled":down,"decision":i,
          "visual":vm,"post_frame_sha256":hashlib.sha256(post.tobytes()).hexdigest(),"post_frame_to_initial_mae":post_mae,"scene_transition":transition,
          "rates_hz":n["controller_rates_hz"],"controller_state":state,"action":action,"encoder":enc,"visual_projection_spikes":vp,
          "baseline":{"steering":asdict(sb),"locomotion":asdict(lb)},"baseline_rates":base_rates if i==0 else None,"baseline_cells":base_cells if i==0 else None,
          "timing_seconds":{"visual_encode":encode_s,"MaleCNS":neural_s,"motor_decode":decode_s,"PyBoy_update":game_s,"total_loop":time.perf_counter()-loop}})
        previous=current
    selected_frames.append(max_frame);sheet(selected_frames,CAPTURES/f"contact-sheet-{condition}-{seed}-{repeat}-{down}.png")
    return {"seed":seed,"condition":condition,"repeat":repeat,"down_enabled":down,"horizon":horizon,"wall_seconds":time.perf_counter()-start,"records":records}

def main():
    e8=json.loads((ROOT/"results/experiment-08-steering/thresholds.json").read_text());e9=json.loads((ROOT/"results/experiment-09-locomotion/thresholds.json").read_text())
    cfg=ExplorationConfig(e8["selected_multiplier"],e8["pooled_baseline_sd_floor_hz"],e8["left_evidence_side"],e9["forward"]["selected_multiplier"],e9["forward"]["sd_floor_hz"],e9["backward"]["selected_multiplier"],e9["backward"]["sd_floor_hz"],1)
    RESULTS.mkdir(parents=True,exist_ok=True);CAPTURES.mkdir(parents=True,exist_ok=True);brain=FlyBrain(data=DATA,device="cuda");_,pops=verify_motor_neurons(brain,DATA);projection=SpatialProjection(brain,E7_CONFIG)
    primary=[];repeats=[]
    with PokemonEmulator() as game:
        for seed in PRIMARY:
            for condition in ("live","frozen_vision","no_vision"):primary.append(trial(game,brain,projection,pops,cfg,seed,condition,HORIZON))
            if seed<606:repeats.append(trial(game,brain,projection,pops,cfg,seed,"live",HORIZON,repeat=1))
            print("primary exploration seed",seed,flush=True)
    (RESULTS/"trials.json").write_text(json.dumps({"metadata":{"seeds":PRIMARY,"horizon":HORIZON,"config":asdict(cfg)},"trials":primary},indent=2)+"\n")
    (RESULTS/"controls.json").write_text(json.dumps({"metadata":{"repeat_seeds":PRIMARY[:5]},"repeat_trials":repeats},indent=2)+"\n")
    secondary=[]
    with PokemonEmulator() as game:
        for seed in SECONDARY:
            secondary.append(trial(game,brain,projection,pops,cfg,seed,"live",HORIZON,down=True));print("secondary exploration seed",seed,flush=True)
    (RESULTS/"secondary-controller.json").write_text(json.dumps({"label":"MDN DOWN is BCI-causal but not validated as naturally vision-dependent","config":asdict(cfg),"trials":secondary},indent=2)+"\n")
    with PokemonEmulator() as game:long_trial=trial(game,brain,projection,pops,cfg,LONG_SEED,"live",LONG_HORIZON,long=True)
    (RESULTS/"long-run.json").write_text(json.dumps({"label":"showcase only; excluded from primary statistics","trial":long_trial},indent=2)+"\n")
    primary_desc={"actions":{"LEFT":"Experiment 8 DNa02 baseline-relative","RIGHT":"Experiment 8 DNa02 baseline-relative","UP":"Experiment 9 DNg100 baseline-relative","DOWN":"disabled","NONE":"deadband/cooldown"},"config":asdict(cfg)}
    (RESULTS/"primary-controller.json").write_text(json.dumps(primary_desc,indent=2)+"\n")
    print("saved Experiment 10 autonomous records",flush=True)
if __name__=="__main__":main()
