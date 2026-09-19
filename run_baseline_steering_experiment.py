"""Experiment 8 baseline-relative DNa02 LEFT/RIGHT closed-loop trials."""
from __future__ import annotations

from dataclasses import asdict
import hashlib, json, time

import numpy as np
from PIL import Image
from flybrain import FlyBrain

from flymon.emulator import PokemonEmulator
from flymon.motor import E7_CONFIG, OnlineE3Encoder, run_neural_window, sha256_array, verify_motor_neurons
from flymon.spatiotemporal import SpatialProjection
from flymon.steering import BaselineRelativeSteeringController, BaselineSteeringConfig, estimate_baseline
from run_visual_experiment import DATA, ROOT

RESULTS=ROOT/"results/experiment-08-steering"; CAPTURES=ROOT/"captures/experiment-08"; STATE=ROOT/"states/bedroom.state"
SEEDS=list(range(421,441)); BASELINE_WINDOWS=10; HORIZON=60; HOLD_FRAMES=8; RELEASE_FRAMES=4
BEDROOM_CONDITIONS=("live_vision","no_vision","frozen_frame","shuffled_vision","mirrored_vision")


def encoder_sides(projection,vector):
    offset=0;sums={"L":0.,"R":0.}
    for (_,side),ids in projection.populations.items(): sums[side]+=float(vector[offset:offset+len(ids)].sum());offset+=len(ids)
    return {"left_activity":sums["L"],"right_activity":sums["R"],"left_minus_right":sums["L"]-sums["R"]}


def transform(condition,current,initial,seed,shuffle_order=None):
    if condition in ("no_vision","title_no_vision"): return None
    if condition=="frozen_frame": return initial
    if condition=="shuffled_vision":
        out=current.reshape(-1,4)[shuffle_order].reshape(current.shape).copy();out[...,3]=255;return out
    if condition=="mirrored_vision": return np.ascontiguousarray(current[:,::-1])
    return current


def baseline_phase(brain,projection,populations,seed):
    brain.reset(seed=seed);cache={};rates=[];seconds=[]
    for _ in range(BASELINE_WINDOWS):
        t=time.perf_counter();neural,_=run_neural_window(brain,projection,None,populations,E7_CONFIG.neural_steps_per_frame,cache);seconds.append(time.perf_counter()-t)
        rates.append(neural["controller_rates_hz"])
    return estimate_baseline(rates),rates,cache,float(np.mean(seconds))


def run_trial(game,brain,projection,populations,cfg,seed,condition,environment="bedroom",swap=False,capture=False):
    if environment=="bedroom": game.load_state(STATE);game.tick(1)
    elif environment=="title": game.tick(1500)
    else: raise ValueError(environment)
    initial=game.framebuffer();baseline,baseline_rates,cache,baseline_step_s=baseline_phase(brain,projection,populations,seed)
    controller=BaselineRelativeSteeringController(baseline,cfg,swap_mapping=swap);encoder=OnlineE3Encoder(projection)
    order=np.random.default_rng(80000+seed).permutation(144*160);records=[]
    for decision in range(HORIZON):
        loop_t=time.perf_counter();current=game.framebuffer();source=transform(condition,current,initial,seed,order)
        t=time.perf_counter()
        if source is None: vector=None;enc={"grid_sha256":None,"grid_mean":None,"grid_std":None,"change_mean":None,"encoder_sha256":None,"encoder_mean":0.,"encoder_nonzero":0};sides={"left_activity":0.,"right_activity":0.,"left_minus_right":0.}
        else: vector,enc=encoder.encode(source);sides=encoder_sides(projection,vector)
        encode_s=time.perf_counter()-t
        t=time.perf_counter();neural,vp=run_neural_window(brain,projection,vector,populations,E7_CONFIG.neural_steps_per_frame,cache);neural_s=time.perf_counter()-t
        t=time.perf_counter();rates=neural["controller_rates_hz"];dna_access_s=time.perf_counter()-t
        t=time.perf_counter();action,state=controller.decode(rates);controller_s=time.perf_counter()-t
        t=time.perf_counter()
        if action:game.press(action.lower());game.tick(HOLD_FRAMES);game.release(action.lower())
        else:game.tick(HOLD_FRAMES)
        game.tick(RELEASE_FRAMES);post=game.framebuffer();pyboy_s=time.perf_counter()-t
        record={"trial":f"{environment}-{condition}-seed-{seed}-swap-{swap}","seed":seed,"environment":environment,"condition":condition,
          "mapping_swap":swap,"decision_index":decision,"frame_sha256":sha256_array(current),"source_frame_sha256":None if source is None else sha256_array(source),
          "post_action_frame_sha256":sha256_array(post),"baseline":asdict(baseline),"baseline_window_rates_hz":baseline_rates if decision==0 else None,
          "DNa02_rates_hz":{"L":rates["DNa02_L"],"R":rates["DNa02_R"]},"diagnostic_rates_hz":{"DNg100":rates["DNg100"],"MDN":rates["MDN"],"DNp01":rates["DNp01"]},
          "controller_state":state,"selected_action":action,"encoder":enc,"encoder_lateral":sides,"visual_projection_spikes":vp,
          "action_hold_frames":HOLD_FRAMES if action else 0,"release_frames":RELEASE_FRAMES,
          "timing_seconds":{"visual_encoding":encode_s,"MaleCNS_and_spike_aggregation":neural_s,"DNa02_access":dna_access_s,"controller":controller_s,
                            "PyBoy_action_update":pyboy_s,"total_decision_loop":time.perf_counter()-loop_t,"baseline_MaleCNS_per_window":baseline_step_s}}
        records.append(record)
        if capture and decision%6==0:Image.fromarray(post,"RGBA").save(CAPTURES/f"{environment}-{condition}-seed-{seed}-frame-{decision:02d}.png")
    return records


def main():
    thresholds=json.loads((RESULTS/"thresholds.json").read_text())
    if thresholds["primary_test_seeds"]!=SEEDS or not thresholds["frozen_before_primary"]:raise RuntimeError("threshold protocol is not frozen")
    cfg=BaselineSteeringConfig(thresholds["selected_multiplier"],thresholds["pooled_baseline_sd_floor_hz"],thresholds["left_evidence_side"],1)
    brain=FlyBrain(data=DATA,device="cuda");_,populations=verify_motor_neurons(brain,DATA);projection=SpatialProjection(brain,E7_CONFIG)
    CAPTURES.mkdir(parents=True,exist_ok=True);trials=[];swaps=[];dynamic=[]
    with PokemonEmulator() as game:
        for seed in SEEDS:
            for condition in BEDROOM_CONDITIONS:trials.extend(run_trial(game,brain,projection,populations,cfg,seed,condition,capture=(seed==421 and condition=="live_vision")))
            if seed<431:swaps.extend(run_trial(game,brain,projection,populations,cfg,seed,"live_vision",swap=True))
            print("bedroom steering seed",seed,flush=True)
    for seed in SEEDS:
        for condition in ("title_live","title_no_vision"):
            with PokemonEmulator() as game:dynamic.extend(run_trial(game,brain,projection,populations,cfg,seed,condition,environment="title",capture=(seed==421 and condition=="title_live")))
        print("title steering seed",seed,flush=True)
    metadata={"model":"flybrain 0.1.0 MaleCNS v1.0","seeds":SEEDS,"conditions":BEDROOM_CONDITIONS,"baseline_windows":BASELINE_WINDOWS,
      "horizon":HORIZON,"neural_steps_per_decision":E7_CONFIG.neural_steps_per_frame,"simulated_seconds_per_decision":E7_CONFIG.neural_steps_per_frame*brain.dt,
      "hold_frames":HOLD_FRAMES,"release_frames":RELEASE_FRAMES,"controller_config":asdict(cfg),"state_sha256":hashlib.sha256(STATE.read_bytes()).hexdigest()}
    (RESULTS/"trials.json").write_text(json.dumps({"metadata":metadata,"records":trials},indent=2)+"\n")
    (RESULTS/"controls.json").write_text(json.dumps({"metadata":metadata,"mapping_swap":swaps,"dynamic_condition":dynamic},indent=2)+"\n")
    print("saved Experiment 8 primary records",flush=True)


if __name__=="__main__":main()
