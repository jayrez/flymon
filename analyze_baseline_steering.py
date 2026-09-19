"""Experiment 8 preregistered statistics, figures, and report."""
from collections import Counter, defaultdict
import json, math
import numpy as np
from PIL import Image, ImageDraw
from run_visual_experiment import ROOT

RESULTS=ROOT/"results/experiment-08-steering"; CAPTURES=ROOT/"captures/experiment-08"

def signal(r): return float(r["controller_state"]["steering_signal_hz"])
def dist(rows):
    c=Counter(r["selected_action"] for r in rows); n=len(rows)
    return {"LEFT":c["LEFT"]/n,"RIGHT":c["RIGHT"]/n,"NONE":c[None]/n}
def tv(a,b): return .5*sum(abs(a[k]-b[k]) for k in a)
def entropy(d): return float(-sum(x*math.log2(x) for x in d.values() if x))

def action_test(a,b,n=9999,seed=808):
    by=defaultdict(lambda:[[],[]])
    for r in a: by[r["seed"]][0].append(r)
    for r in b: by[r["seed"]][1].append(r)
    obs=tv(dist(a),dist(b)); rng=np.random.default_rng(seed); null=[]
    for _ in range(n):
        aa=[];bb=[]
        for s in sorted(by):
            x,y=by[s]
            if rng.integers(2): x,y=y,x
            aa+=x;bb+=y
        null.append(tv(dist(aa),dist(bb)))
    return {"metric":"total variation","effect_size":obs,"p_value":(1+sum(x>=obs-1e-15 for x in null))/(n+1),"permutations":n}

def continuous_test(a,b,n=9999,seed=809):
    seeds=sorted({r["seed"] for r in a}&{r["seed"] for r in b})
    values=np.array([np.mean([signal(r) for r in a if r["seed"]==s])-np.mean([signal(r) for r in b if r["seed"]==s]) for s in seeds])
    obs=float(values.mean()); rng=np.random.default_rng(seed)
    null=[float((values*rng.choice([-1,1],len(values))).mean()) for _ in range(n)]
    return {"metric":"paired seed mean steering difference Hz","effect_size":obs,"absolute_effect_size":abs(obs),
            "p_value":(1+sum(abs(x)>=abs(obs)-1e-15 for x in null))/(n+1),"permutations":n,"seed_differences":values.tolist()}

def summarize(rows):
    x=np.array([signal(r) for r in rows]); t=np.array([r["controller_state"]["threshold_hz"] for r in rows]); d=dist(rows); lag=[]
    for seed in sorted({r["seed"] for r in rows}):
        y=np.array([signal(r) for r in rows if r["seed"]==seed])
        if np.std(y[:-1]) and np.std(y[1:]): lag.append(np.corrcoef(y[:-1],y[1:])[0,1])
    return {"n":len(rows),"action_distribution":d,"action_rate":1-d["NONE"],"action_entropy_bits":entropy(d),
      "steering_mean_hz":float(x.mean()),"steering_abs_mean_hz":float(abs(x).mean()),"steering_sd_hz":float(x.std(ddof=1)),
      "steering_variance_hz2":float(x.var(ddof=1)),
      "steering_percentiles_hz":dict(zip(("p05","p25","p50","p75","p95"),map(float,np.percentile(x,[5,25,50,75,95])))),
      "fraction_positive":float(np.mean(x>0)),"fraction_negative":float(np.mean(x<0)),
      "fraction_above_left_threshold":float(np.mean(x>t)),"fraction_below_right_threshold":float(np.mean(x<-t)),
      "fraction_inside_deadband":float(np.mean(abs(x)<=t)),"threshold_crossing_rate":float(np.mean(abs(x)>t)),
      "lag1_autocorrelation_mean":float(np.mean(lag)) if lag else None}

def calibration_summary(cal):
    out=cal["selected_controller_outputs"]; result={}
    for condition in ("left_biased","right_biased","neutral","no_vision"):
        rows=[r for r in out if r["condition"]==condition]; x=np.array([r["steering_signal_hz"] for r in rows]); c=Counter(r["action"] for r in rows)
        result[condition]={"mean_steering_hz":float(x.mean()),"sd_steering_hz":float(x.std(ddof=1)),
          "LEFT_RIGHT_ratio":float((c["LEFT"]+.5)/(c["RIGHT"]+.5)),"LEFT_fraction":c["LEFT"]/len(rows),
          "RIGHT_fraction":c["RIGHT"]/len(rows),"NONE_fraction":c[None]/len(rows),
          "percentiles_hz":dict(zip(("p05","p25","p50","p75","p95"),map(float,np.percentile(x,[5,25,50,75,95]))))}
    return result

def main():
    td=json.loads((RESULTS/"trials.json").read_text()); cd=json.loads((RESULTS/"controls.json").read_text())
    cal=json.loads((RESULTS/"calibration.json").read_text()); thresholds=json.loads((RESULTS/"thresholds.json").read_text())
    rows=td["records"]; groups={c:[r for r in rows if r["condition"]==c] for c in td["metadata"]["conditions"]}
    summaries={k:summarize(v) for k,v in groups.items()}; live=groups["live_vision"]; frozen=groups["frozen_frame"]; mirror=groups["mirrored_vision"]
    comparisons={}
    pairs=(("live_vs_no_vision",groups["no_vision"]),("live_vs_frozen",frozen),("live_vs_shuffled",groups["shuffled_vision"]),("normal_vs_mirrored",mirror))
    for i,(name,other) in enumerate(pairs):
        comparisons[name]={"actions":action_test(live,other,seed=810+i),"continuous":continuous_test(live,other,seed=820+i)}
    lm={(r["seed"],r["decision_index"]):r for r in live}; fm={(r["seed"],r["decision_index"]):r for r in frozen}
    matched=[(lm[k],fm[k]) for k in lm if k[1]>0]
    feedback={"frame_hash_divergence_fraction":float(np.mean([a["post_action_frame_sha256"]!=b["post_action_frame_sha256"] for a,b in matched])),
      "steering_absolute_divergence_mean_hz":float(np.mean([abs(signal(a)-signal(b)) for a,b in matched])),
      "action_disagreement_fraction":float(np.mean([a["selected_action"]!=b["selected_action"] for a,b in matched])),
      "action_distribution_TV":comparisons["live_vs_frozen"]["actions"]["effect_size"]}
    swap=cd["mapping_swap"]; normal10=[r for r in live if r["seed"]<431]
    side_swap={"normal_distribution":dist(normal10),"swapped_distribution":dist(swap),"TV":tv(dist(normal10),dist(swap))}
    dynamic=cd["dynamic_condition"]; title_live=[r for r in dynamic if r["condition"]=="title_live"]; title_no=[r for r in dynamic if r["condition"]=="title_no_vision"]
    dynamic_stats={"condition":"title animation","live":summarize(title_live),"no_vision":summarize(title_no),
      "comparison":{"actions":action_test(title_live,title_no,seed=830),"continuous":continuous_test(title_live,title_no,seed=831)}}
    enc=np.array([r["encoder_lateral"]["left_minus_right"] for r in live]); dna=np.array([signal(r) for r in live])
    encoder_diag={"left_minus_right_mean":float(enc.mean()),"sd":float(enc.std(ddof=1)),
      "correlation_with_DNa02_steering":float(np.corrcoef(enc,dna)[0,1]) if np.std(enc) and np.std(dna) else None}
    perf={k:float(np.mean([r["timing_seconds"][k] for r in rows])) for k in rows[0]["timing_seconds"]}
    calibration=calibration_summary(cal); lateral=thresholds["validation_lateral_test"]
    lateral_ok=thresholds["validation_opposite_signed_means"] and lateral["p_value"]<.05
    natural=comparisons["live_vs_no_vision"]["actions"]["p_value"]<.05 or comparisons["live_vs_no_vision"]["continuous"]["p_value"]<.05
    mirror_ok=comparisons["normal_vs_mirrored"]["actions"]["p_value"]<.05 or comparisons["normal_vs_mirrored"]["continuous"]["p_value"]<.05
    feedback_ok=feedback["action_distribution_TV"]>0.005 and feedback["action_disagreement_fraction"]>0.021794871794871794
    if lateral_ok and natural and mirror_ok and feedback_ok: verdict,primary,answer="PASS","E","YES"
    elif lateral_ok and natural: verdict,primary,answer="WEAK",("D" if summaries["live_vision"]["action_rate"]>.05 else "C"),"PARTIALLY"
    elif lateral_ok: verdict,primary,answer="WEAK","B","NO"
    else: verdict,primary,answer="FAIL","A","NO"
    source=(ROOT/"flymon/steering.py").read_text().lower(); run=(ROOT/"run_baseline_steering_experiment.py").read_text().lower()
    leakage={"pixels_directly_into_controller":".framebuffer" not in source and "encoder." not in source,"RAM":"read_memory" not in source and "memory[" not in source,
      "class_labels":"class_label" not in source and "screen_class" not in source,"trained_policy":all(x not in source for x in ("fit(","predict(","sklearn","torch","reward")),
      "scripted_route":all(x not in run for x in ("up up","left left","desired_route","route ="))}
    e7=json.loads((ROOT/"results/experiment-07-closed-loop/statistics.json").read_text())
    e7cmp={"experiment_7_action_rate":1-e7["summaries"]["live_vision"]["distribution"]["NONE"],"experiment_8_action_rate":summaries["live_vision"]["action_rate"],
      "experiment_7_live_no_TV":e7["visual_dependence"]["live_vs_no_vision"]["observed"],"experiment_8_live_no_TV":comparisons["live_vs_no_vision"]["actions"]["effect_size"],
      "experiment_7_lateral_p":.0083,"experiment_8_lateral_p":lateral["p_value"],"experiment_7_live_frozen_TV":.005,"experiment_8_live_frozen_TV":feedback["action_distribution_TV"]}
    stats={"verdict":verdict,"primary_result":primary,"answer":answer,"calibration":calibration,"thresholds":thresholds,"natural_summaries":summaries,
      "comparisons":comparisons,"closed_loop_feedback":feedback,"side_swap":side_swap,"dynamic_condition":dynamic_stats,"encoder_lateral_diagnostic":encoder_diag,
      "performance_seconds":perf,"experiment_7_comparison":e7cmp,"leakage_checks":leakage,
      "criteria":{"controlled_lateral":lateral_ok,"natural_visual_dependence":natural,"mirror_effect":mirror_ok,"feedback_improved":feedback_ok}}
    (RESULTS/"statistics.json").write_text(json.dumps(stats,indent=2)+"\n"); pass
    L=calibration["left_biased"];R=calibration["right_biased"];B=summaries["live_vision"];N=summaries["no_vision"];F=summaries["frozen_frame"];S=summaries["shuffled_vision"];M=summaries["mirrored_vision"]
    lvn=comparisons["live_vs_no_vision"];lvf=comparisons["live_vs_frozen"];nvm=comparisons["normal_vs_mirrored"]
    final=f"""Experiment 8 verdict:
{verdict}

Calibration seeds:
401-420
Primary test seeds:
421-440

Baseline method:
10 no-vision windows (2.0 simulated s), per-trial frozen DNa02 L/R means and L-R variance
Threshold method:
inner selection among 1.0/1.5/2.0 x max(per-trial baseline SD, calibration pooled SD floor)
Selected threshold:
{thresholds['selected_multiplier']} SD; pooled floor {thresholds['pooled_baseline_sd_floor_hz']:.4f} Hz

CONTROLLED LATERAL ASSAY

Left-biased stimulus:
mean steering signal: {L['mean_steering_hz']:.4f} Hz
LEFT/RIGHT ratio: {L['LEFT_RIGHT_ratio']:.4f}

Right-biased stimulus:
mean steering signal: {R['mean_steering_hz']:.4f} Hz
LEFT/RIGHT ratio: {R['LEFT_RIGHT_ratio']:.4f}

lateral assay p-value: {lateral['p_value']:.6f}

NATURAL BEDROOM - LIVE

LEFT %: {100*B['action_distribution']['LEFT']:.3f}
RIGHT %: {100*B['action_distribution']['RIGHT']:.3f}
NO ACTION %: {100*B['action_distribution']['NONE']:.3f}
mean steering signal: {B['steering_mean_hz']:.4f} Hz
steering SD: {B['steering_sd_hz']:.4f} Hz
threshold-crossing rate: {B['threshold_crossing_rate']:.4f}

NO VISION

LEFT %: {100*N['action_distribution']['LEFT']:.3f}
RIGHT %: {100*N['action_distribution']['RIGHT']:.3f}
NO ACTION %: {100*N['action_distribution']['NONE']:.3f}
mean steering signal: {N['steering_mean_hz']:.4f} Hz

FROZEN FRAME
action distribution: {json.dumps(F['action_distribution'],sort_keys=True)}
mean steering signal: {F['steering_mean_hz']:.4f} Hz

SHUFFLED VISION
action distribution: {json.dumps(S['action_distribution'],sort_keys=True)}
mean steering signal: {S['steering_mean_hz']:.4f} Hz

MIRRORED VISION
action distribution: {json.dumps(M['action_distribution'],sort_keys=True)}
mean steering signal: {M['steering_mean_hz']:.4f} Hz

Live vs no-vision:
effect size: action TV {lvn['actions']['effect_size']:.6f}; steering difference {lvn['continuous']['effect_size']:.4f} Hz
p-value: actions {lvn['actions']['p_value']:.4f}; steering {lvn['continuous']['p_value']:.4f}

Live vs frozen:
effect size: action TV {lvf['actions']['effect_size']:.6f}; steering difference {lvf['continuous']['effect_size']:.4f} Hz
p-value: actions {lvf['actions']['p_value']:.4f}; steering {lvf['continuous']['p_value']:.4f}

Normal vs mirrored:
effect size: action TV {nvm['actions']['effect_size']:.6f}; steering difference {nvm['continuous']['effect_size']:.4f} Hz
p-value: actions {nvm['actions']['p_value']:.4f}; steering {nvm['continuous']['p_value']:.4f}

Experiment 7 action rate: {e7cmp['experiment_7_action_rate']:.6f}
Experiment 8 action rate: {e7cmp['experiment_8_action_rate']:.6f}

Experiment 7 live-vs-no-vision effect: TV {e7cmp['experiment_7_live_no_TV']:.6f}
Experiment 8 live-vs-no-vision effect: TV {e7cmp['experiment_8_live_no_TV']:.6f}

Dynamic Pokemon condition:
condition: title animation
action rate: {dynamic_stats['live']['action_rate']:.6f}
vision effect: action TV {dynamic_stats['comparison']['actions']['effect_size']:.6f}, p={dynamic_stats['comparison']['actions']['p_value']:.4f}; steering p={dynamic_stats['comparison']['continuous']['p_value']:.4f}

Primary result:
{primary}

Controller leakage checks:
pixels directly into controller: {'PASS' if leakage['pixels_directly_into_controller'] else 'FAIL'}
RAM: {'PASS' if leakage['RAM'] else 'FAIL'}
class labels: {'PASS' if leakage['class_labels'] else 'FAIL'}
trained policy: {'PASS' if leakage['trained_policy'] else 'FAIL'}
scripted route: {'PASS' if leakage['scripted_route'] else 'FAIL'}

Did baseline-relative DNa02 decoding convert the controlled visual steering effect into significant natural Pokemon visual-motor behavior?
{answer}
"""
    interpretations={"A":"Baseline-relative decoding did not improve DNa02 visual sensitivity.","B":"Controlled DNa02 steering remained valid, but natural Pokemon frames supplied too little useful lateral drive.","C":"Natural Pokemon frames modulated DNa02, but thresholding still prevented robust behavior.","D":"Natural vision significantly modulated DNa02 actions, while feedback remained weak.","E":"Natural vision robustly modulated steering and materially changed feedback."}
    (RESULTS/"analysis.md").write_text("# Experiment 8 analysis\n\n```text\n"+final+"```\n\n## Interpretation\n\n"+interpretations[primary]+"\n\nThis is an engineered BCI and does not establish natural Pokemon motor semantics or semantic understanding.\n")
    print(final)

if __name__=="__main__": main()
