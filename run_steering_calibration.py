"""Experiment 8 calibration and frozen threshold selection."""
from __future__ import annotations

from dataclasses import asdict
import json, math, time

import numpy as np
from flybrain import FlyBrain

from flymon.motor import E7_CONFIG, OnlineE3Encoder, run_neural_window, synthetic_frame, verify_motor_neurons
from flymon.spatiotemporal import SpatialProjection
from flymon.steering import (BaselineRelativeSteeringController, BaselineSteeringConfig,
                             estimate_baseline)
from run_visual_experiment import DATA, ROOT

RESULTS=ROOT/"results/experiment-08-steering"; CAPTURES=ROOT/"captures/experiment-08"
SEEDS=list(range(401,421)); SELECT=set(range(401,411)); VALIDATE=set(range(411,421))
BASELINE_WINDOWS=10; ACTIVE_WINDOWS=30; CANDIDATES=(1.0,1.5,2.0)


def encoder_sides(projection, vector):
    offset=0; sums={"L":0.0,"R":0.0}
    for (_,side),ids in projection.populations.items():
        sums[side]+=float(vector[offset:offset+len(ids)].sum()); offset+=len(ids)
    return {"left_activity":sums["L"],"right_activity":sums["R"],
            "left_minus_right":sums["L"]-sums["R"]}


def one_condition(brain, projection, populations, seed, condition, frame):
    brain.reset(seed=seed); cache={}; baseline_rows=[]; timings=[]
    for window in range(BASELINE_WINDOWS):
        t=time.perf_counter(); neural,_=run_neural_window(brain,projection,None,populations,E7_CONFIG.neural_steps_per_frame,cache)
        timings.append(time.perf_counter()-t); baseline_rows.append(neural["controller_rates_hz"])
    baseline=estimate_baseline(baseline_rows); encoder=OnlineE3Encoder(projection); rows=[]
    for decision in range(ACTIVE_WINDOWS):
        if frame is None: vector=None; enc={"encoder_sha256":None,"encoder_mean":0.,"encoder_nonzero":0}; sides={"left_activity":0.,"right_activity":0.,"left_minus_right":0.}
        else: vector,enc=encoder.encode(frame); sides=encoder_sides(projection,vector)
        t=time.perf_counter(); neural,vp=run_neural_window(brain,projection,vector,populations,E7_CONFIG.neural_steps_per_frame,cache); neural_s=time.perf_counter()-t
        rates=neural["controller_rates_hz"]; rows.append({"seed":seed,"condition":condition,"decision":decision,
            "rates_hz":rates,"raw_L_minus_R_hz":rates["DNa02_L"]-rates["DNa02_R"],
            "encoder":enc,"encoder_lateral":sides,"visual_projection_spikes":vp,
            "diagnostic_rates_hz":{"DNg100":rates["DNg100"],"MDN":rates["MDN"],"DNp01":rates["DNp01"]},
            "neural_seconds":neural_s})
    return asdict(baseline),baseline_rows,rows,{"baseline_neural_seconds_mean":float(np.mean(timings))}


def replay(rows, baselines, multiplier, floor, polarity):
    outputs=[]
    for seed in SEEDS:
        for condition in ("left_biased","right_biased","neutral","no_vision"):
            cfg=BaselineSteeringConfig(multiplier,floor,polarity,1)
            controller=BaselineRelativeSteeringController(type_baseline(baselines[f"{seed}:{condition}"]),cfg)
            for row in [r for r in rows if r["seed"]==seed and r["condition"]==condition]:
                action,state=controller.decode(row["rates_hz"])
                outputs.append({"seed":seed,"condition":condition,"decision":row["decision"],
                                "action":action,"steering_signal_hz":state["steering_signal_hz"],
                                "threshold_hz":state["threshold_hz"]})
    return outputs


def type_baseline(d):
    from flymon.steering import SteeringBaseline
    return SteeringBaseline(**d)


def score(outputs,seeds):
    subset=[x for x in outputs if x["seed"] in seeds]
    def frac(condition,action=None):
        x=[r for r in subset if r["condition"]==condition]
        return np.mean([(r["action"] is not None) if action is None else r["action"]==action for r in x])
    correct=(frac("left_biased","LEFT")+frac("right_biased","RIGHT"))/2
    wrong=(frac("left_biased","RIGHT")+frac("right_biased","LEFT"))/2
    false=(frac("neutral")+frac("no_vision"))/2
    return {"correct_lateral_fraction":float(correct),"incorrect_lateral_fraction":float(wrong),
            "neutral_no_vision_action_fraction":float(false),"objective":float(correct-wrong)}


def sign_paired(values):
    values=np.asarray(values,float); observed=float(values.mean()); null=[]
    for bits in range(1<<len(values)):
        signs=np.array([1 if bits&(1<<i) else -1 for i in range(len(values))]); null.append(float((values*signs).mean()))
    return {"observed":observed,"p_value":float(sum(abs(x)>=abs(observed)-1e-12 for x in null)/len(null)),"enumerations":len(null)}


def main():
    RESULTS.mkdir(parents=True,exist_ok=True); CAPTURES.mkdir(parents=True,exist_ok=True)
    brain=FlyBrain(data=DATA,device="cuda"); _,populations=verify_motor_neurons(brain,DATA); projection=SpatialProjection(brain,E7_CONFIG)
    frames={"left_biased":synthetic_frame("left"),"right_biased":synthetic_frame("right"),"neutral":synthetic_frame("uniform"),"no_vision":None}
    rows=[]; baselines={}; baseline_windows={}; timing={}
    for seed in SEEDS:
        for condition,frame in frames.items():
            base,raw,active,t=one_condition(brain,projection,populations,seed,condition,frame)
            baselines[f"{seed}:{condition}"]=base; baseline_windows[f"{seed}:{condition}"]=raw; rows.extend(active); timing[f"{seed}:{condition}"]=t
        print("steering calibration seed",seed,flush=True)
    residual=[]
    for key,windows in baseline_windows.items():
        if int(key.split(":")[0]) in SELECT:
            values=np.array([x["DNa02_L"]-x["DNa02_R"] for x in windows]); residual.extend(values-values.mean())
    floor=float(np.std(residual,ddof=1)); floor=max(floor,0.5)
    left=np.mean([r["raw_L_minus_R_hz"] for r in rows if r["seed"] in SELECT and r["condition"]=="left_biased"])
    right=np.mean([r["raw_L_minus_R_hz"] for r in rows if r["seed"] in SELECT and r["condition"]=="right_biased"])
    polarity="L" if left-right>0 else "R"
    candidates=[]; replayed={}
    for multiplier in CANDIDATES:
        out=replay(rows,baselines,multiplier,floor,polarity); replayed[str(multiplier)]=out
        candidates.append({"multiplier":multiplier,"selection":score(out,SELECT),"validation":score(out,VALIDATE)})
    eligible=[x for x in candidates if x["selection"]["neutral_no_vision_action_fraction"]<=.10]
    if eligible: chosen=max(eligible,key=lambda x:(x["selection"]["objective"],x["multiplier"]))
    else: chosen=min(candidates,key=lambda x:(x["selection"]["neutral_no_vision_action_fraction"],-x["selection"]["objective"],-x["multiplier"]))
    outputs=replayed[str(chosen["multiplier"])]
    validation=[]
    for seed in sorted(VALIDATE):
        l=np.mean([x["steering_signal_hz"] for x in outputs if x["seed"]==seed and x["condition"]=="left_biased"])
        r=np.mean([x["steering_signal_hz"] for x in outputs if x["seed"]==seed and x["condition"]=="right_biased"])
        validation.append({"seed":seed,"left_mean":float(l),"right_mean":float(r),"opposite_effect":float(l-r)})
    test=sign_paired([x["opposite_effect"] for x in validation])
    threshold={"calibration_seeds":SEEDS,"selection_seeds":sorted(SELECT),"validation_seeds":sorted(VALIDATE),
        "candidate_multipliers":list(CANDIDATES),"pooled_baseline_sd_floor_hz":floor,"left_evidence_side":polarity,
        "candidate_scores":candidates,"selected_multiplier":chosen["multiplier"],"selection_rule":"preregistered design.md",
        "validation_lateral_test":test,"validation_opposite_signed_means":bool(np.mean([x['left_mean'] for x in validation])>0 and np.mean([x['right_mean'] for x in validation])<0),
        "primary_test_seeds":list(range(421,441)),"frozen_before_primary":True}
    (RESULTS/"thresholds.json").write_text(json.dumps(threshold,indent=2)+"\n")
    (RESULTS/"calibration.json").write_text(json.dumps({"metadata":{"baseline_windows":BASELINE_WINDOWS,"active_windows":ACTIVE_WINDOWS,
        "neural_steps_per_window":E7_CONFIG.neural_steps_per_frame},"baselines":baselines,"baseline_windows":baseline_windows,
        "records":rows,"selected_controller_outputs":outputs,"validation":validation,"timing":timing},indent=2)+"\n")
    print(json.dumps(threshold,indent=2),flush=True)


if __name__=="__main__": main()
