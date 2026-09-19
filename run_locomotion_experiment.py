"""Experiment 9 locomotion-only primary trials and exploratory integration."""
from dataclasses import asdict
import hashlib,json,time
import numpy as np
from flybrain import FlyBrain
from flymon.emulator import PokemonEmulator
from flymon.locomotion import BaselineLocomotionController,LocomotionConfig,estimate_population_baseline
from flymon.motor import E7_CONFIG,OnlineE3Encoder,run_neural_window,sha256_array,verify_motor_neurons
from flymon.spatiotemporal import SpatialProjection
from flymon.steering import BaselineRelativeSteeringController,BaselineSteeringConfig,estimate_baseline
from run_visual_experiment import DATA,ROOT

RESULTS=ROOT/"results/experiment-09-locomotion";CAPTURES=ROOT/"captures/experiment-09";STATE=ROOT/"states/bedroom.state"
SEEDS=list(range(521,541));BASE=10;HORIZON=60;HOLD=8;RELEASE=4;CONDS=("live_vision","no_vision","frozen_frame","shuffled_vision")

def source(condition,current,initial,order):
    if condition in ("no_vision","title_no_vision"):return None
    if condition=="frozen_frame":return initial
    if condition=="shuffled_vision":
        x=current.reshape(-1,4)[order].reshape(current.shape).copy();x[...,3]=255;return x
    return current

def baseline(brain,proj,pops,seed):
    brain.reset(seed=seed);cache={};rates=[];cells=[]
    for _ in range(BASE):
        n,_=run_neural_window(brain,proj,None,pops,E7_CONFIG.neural_steps_per_frame,cache);rates.append(n["controller_rates_hz"]);cells.append(n["per_cell_counts"])
    return estimate_population_baseline(rates,cells),estimate_baseline(rates),rates,cells,cache

def run_trial(game,brain,proj,pops,lcfg,seed,condition,environment="bedroom",ablate=None,integrated=False,sconfig=None,capture=False):
    if environment=="bedroom":game.load_state(STATE);game.tick(1)
    else:game.tick(1500)
    initial=game.framebuffer();lb,sb,br,bc,cache=baseline(brain,proj,pops,seed);loc=BaselineLocomotionController(lb,lcfg,ablate);steer=BaselineRelativeSteeringController(sb,sconfig) if integrated else None
    enc=OnlineE3Encoder(proj);order=np.random.default_rng(90000+seed).permutation(144*160);rows=[];global_cooldown=0
    for i in range(HORIZON):
        t0=time.perf_counter();current=game.framebuffer();frame=source(condition,current,initial,order);te=time.perf_counter()
        if frame is None:vec=None;est={"encoder_sha256":None,"change_mean":None}
        else:vec,est=enc.encode(frame)
        encode_s=time.perf_counter()-te;tn=time.perf_counter();n,vp=run_neural_window(brain,proj,vec,pops,E7_CONFIG.neural_steps_per_frame,cache);neural_s=time.perf_counter()-tn;rates=n["controller_rates_hz"]
        tc=time.perf_counter();action,lstate=loc.decode(rates);sstate=None
        if integrated:
            _,sstate=steer.decode(rates);candidates={}
            if sstate["steering_signal_hz"]>sstate["threshold_hz"]:candidates["LEFT"]=sstate["steering_signal_hz"]/sstate["threshold_hz"]-1
            if sstate["steering_signal_hz"]<-sstate["threshold_hz"]:candidates["RIGHT"]=-sstate["steering_signal_hz"]/sstate["threshold_hz"]-1
            if lstate["forward_crossed"]:candidates["UP"]=lstate["forward_delta_hz"]/lstate["forward_threshold_hz"]-1
            if lstate["backward_crossed"]:candidates["DOWN"]=lstate["backward_delta_hz"]/lstate["backward_threshold_hz"]-1
            order_actions=("LEFT","RIGHT","UP","DOWN");raw=max(order_actions,key=lambda a:(candidates.get(a,-1),-order_actions.index(a))) if candidates else None
            if global_cooldown:action=None;global_cooldown-=1
            else:action=raw;global_cooldown=1 if action else 0
        controller_s=time.perf_counter()-tc;tg=time.perf_counter()
        if action:game.press(action.lower());game.tick(HOLD);game.release(action.lower())
        else:game.tick(HOLD)
        game.tick(RELEASE);post=game.framebuffer();game_s=time.perf_counter()-tg
        rows.append({"trial":f"{environment}-{condition}-{seed}-{ablate}-{integrated}","seed":seed,"environment":environment,"condition":condition,"ablation":ablate,"integrated":integrated,"decision_index":i,
          "frame_sha256":sha256_array(current),"source_sha256":None if frame is None else sha256_array(frame),"post_frame_sha256":sha256_array(post),
          "population_baseline":asdict(lb),"baseline_window_rates":br if i==0 else None,"baseline_per_cell_counts":bc if i==0 else None,
          "rates_hz":rates,"per_cell_counts":n["per_cell_counts"],"locomotion_state":lstate,"steering_state":sstate,"selected_action":action,
          "encoder":est,"visual_projection_spikes":vp,"timing_seconds":{"encoding":encode_s,"MaleCNS_and_aggregation":neural_s,"controller":controller_s,"PyBoy":game_s,"total":time.perf_counter()-t0}})
    return rows

def main():
    th=json.loads((RESULTS/"thresholds.json").read_text());assert th["frozen_before_primary"] and th["primary_seeds"]==SEEDS
    lcfg=LocomotionConfig(th["forward"]["selected_multiplier"],th["backward"]["selected_multiplier"],th["forward"]["sd_floor_hz"],th["backward"]["sd_floor_hz"],1)
    e8=json.loads((ROOT/"results/experiment-08-steering/thresholds.json").read_text());sconfig=BaselineSteeringConfig(e8["selected_multiplier"],e8["pooled_baseline_sd_floor_hz"],e8["left_evidence_side"],1)
    brain=FlyBrain(data=DATA,device="cuda");_,pops=verify_motor_neurons(brain,DATA);proj=SpatialProjection(brain,E7_CONFIG);trials=[];ablations=[];dynamic=[];integrated=[]
    with PokemonEmulator() as game:
        for seed in SEEDS:
            for condition in CONDS:trials+=run_trial(game,brain,proj,pops,lcfg,seed,condition)
            if seed<531:
                ablations+=run_trial(game,brain,proj,pops,lcfg,seed,"live_vision",ablate="DNg100")
                ablations+=run_trial(game,brain,proj,pops,lcfg,seed,"live_vision",ablate="MDN")
            if seed<526:integrated+=run_trial(game,brain,proj,pops,lcfg,seed,"live_vision",integrated=True,sconfig=sconfig)
            print("locomotion bedroom seed",seed,flush=True)
    for seed in SEEDS:
        for condition in ("title_live","title_no_vision"):
            with PokemonEmulator() as game:dynamic+=run_trial(game,brain,proj,pops,lcfg,seed,condition,environment="title")
        print("locomotion title seed",seed,flush=True)
    meta={"seeds":SEEDS,"conditions":CONDS,"baseline_windows":BASE,"horizon":HORIZON,"neural_steps":10,"hold_frames":HOLD,"release_frames":RELEASE,
      "locomotion_config":asdict(lcfg),"experiment8_steering_config":asdict(sconfig),"state_sha256":hashlib.sha256(STATE.read_bytes()).hexdigest()}
    (RESULTS/"trials.json").write_text(json.dumps({"metadata":meta,"records":trials},indent=2)+"\n");(RESULTS/"controls.json").write_text(json.dumps({"metadata":meta,"ablations":ablations,"dynamic":dynamic},indent=2)+"\n")
    (RESULTS/"integrated-controller.json").write_text(json.dumps({"metadata":meta,"status":"exploratory; excluded from primary verdict","records":integrated},indent=2)+"\n")
    print("saved Experiment 9 records",flush=True)
if __name__=="__main__":main()
