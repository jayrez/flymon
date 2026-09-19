"""Experiment 9 preregistered locomotion analysis."""
from collections import Counter,defaultdict
import json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run_visual_experiment import ROOT
R=ROOT/"results/experiment-09-locomotion";C=ROOT/"captures/experiment-09"

def delta(r,p):return float(r["locomotion_state"]["forward_delta_hz" if p=="DNg100" else "backward_delta_hz"])
def dist(rows):
 c=Counter(r["selected_action"] for r in rows);n=len(rows)
 return {"UP":c["UP"]/n,"DOWN":c["DOWN"]/n,"NONE":c[None]/n}
def tv(a,b):return .5*sum(abs(a[k]-b[k]) for k in a)
def action_test(a,b,n=9999,seed=910):
 by=defaultdict(lambda:[[],[]])
 for x in a:by[x["seed"]][0].append(x)
 for x in b:by[x["seed"]][1].append(x)
 obs=tv(dist(a),dist(b));rng=np.random.default_rng(seed);null=[]
 for _ in range(n):
  aa=[];bb=[]
  for s in sorted(by):
   x,y=by[s]
   if rng.integers(2):x,y=y,x
   aa+=x;bb+=y
  null.append(tv(dist(aa),dist(bb)))
 return {"effect_TV":obs,"p_value":(1+sum(x>=obs-1e-15 for x in null))/(n+1),"permutations":n}
def continuous_test(a,b,p,n=9999,seed=920):
 seeds=sorted({x["seed"] for x in a}&{x["seed"] for x in b});v=np.array([np.mean([delta(x,p) for x in a if x["seed"]==s])-np.mean([delta(x,p) for x in b if x["seed"]==s]) for s in seeds])
 obs=float(v.mean());rng=np.random.default_rng(seed);null=[float((v*rng.choice([-1,1],len(v))).mean()) for _ in range(n)]
 return {"effect_hz":obs,"p_value":(1+sum(abs(x)>=abs(obs)-1e-15 for x in null))/(n+1),"seed_effects":v.tolist()}
def summary(rows):
 d=dist(rows);out={"distribution":d,"action_rate":1-d["NONE"],"actions_per_trial":float(np.mean([sum(x["selected_action"] is not None for x in rows if x["seed"]==s) for s in sorted({x["seed"] for x in rows})])),
 "action_entropy_bits":float(-sum(x*math.log2(x) for x in d.values() if x))}
 for p,key in (("DNg100","forward"),("MDN","backward")):
  x=np.array([delta(r,p) for r in rows]);cross=np.array([r["locomotion_state"][key+"_crossed"] for r in rows])
  out[p]={"mean_delta_hz":float(x.mean()),"sd_hz":float(x.std(ddof=1)),"percentiles_hz":dict(zip(("p05","p25","p50","p75","p95"),map(float,np.percentile(x,[5,25,50,75,95])))),
          "threshold_crossing_rate":float(cross.mean()),"seed_mean_variance":float(np.var([np.mean([delta(r,p) for r in rows if r["seed"]==s]) for s in sorted({r["seed"] for r in rows})],ddof=1))}
 return out
def calibration_stats(cal,th):
 result={}
 for condition in cal["metadata"]["conditions"]:
  rows=[r for r in cal["records"] if r["condition"]==condition];result[condition]={}
  for p,key,cfg in (("DNg100","forward",th["forward"]),("MDN","backward",th["backward"])):
   vals=np.array([r[key+"_delta_hz"] for r in rows]);cross=[]
   for r in rows:
    b=cal["baselines"][f"{r['seed']}:{condition}"];sd=max(b[("dng100" if p=="DNg100" else "mdn")+"_variance_hz2"]**.5,cfg["sd_floor_hz"]);cross.append(r[key+"_delta_hz"]>cfg["selected_multiplier"]*sd)
   result[condition][p]={"raw_rate_mean_hz":float(np.mean([r["rates_hz"][p] for r in rows])),"delta_mean_hz":float(vals.mean()),"delta_sd_hz":float(vals.std(ddof=1)),
    "percentiles_hz":dict(zip(("p05","p25","p50","p75","p95"),map(float,np.percentile(vals,[5,25,50,75,95])))),"threshold_crossing_rate":float(np.mean(cross)),
    "seed_mean_variance":float(np.var([np.mean([r[key+"_delta_hz"] for r in rows if r["seed"]==s]) for s in sorted({r["seed"] for r in rows})],ddof=1))}
 return result
def plot(path,title,series,actions=None):
 im=Image.new("RGB",(1050,450),"white");d=ImageDraw.Draw(im);d.text((15,10),title,fill="black");vals=[v for a,c in series for v in a];scale=max(1,max(abs(float(v)) for v in vals));d.line((35,225,1015,225),fill="#aaa")
 for a,color in series:
  pts=[(35+i*980/max(1,len(a)-1),225-180*float(v)/scale) for i,v in enumerate(a)]
  if len(pts)>1:d.line(pts,fill=color,width=3)
 if actions:
  for i,a in enumerate(actions):
   x=35+i*980/max(1,len(actions)-1)
   if a=="UP":d.polygon([(x,420),(x-5,430),(x+5,430)],fill="#4daf4a")
   if a=="DOWN":d.polygon([(x,440),(x-5,430),(x+5,430)],fill="#984ea3")
 im.save(path)
def figures(cal,live,no,frozen,integrated):
 C.mkdir(parents=True,exist_ok=True)
 for p,key,color in (("DNg100","forward_delta_hz","#4daf4a"),("MDN","backward_delta_hz","#984ea3")):
  series=[]
  for cond,c in (("no_vision","#777"),("bedroom","#377eb8"),("looming","#e41a1c"),("vertical_motion","#ff7f00")):
   vals=[r[key] for r in cal["records"] if r["condition"]==cond];series.append((np.histogram(vals,bins=np.arange(-16,21,1))[0],c))
  plot(C/(p+"-distributions.png"),p+" controlled delta distributions",series)
 sample=[r for r in live if r["seed"]==521];plot(C/"locomotion-action-timeline.png","DNg100/MDN delta and actions",[([delta(r,"DNg100") for r in sample],"#4daf4a"),([delta(r,"MDN") for r in sample],"#984ea3")],[r["selected_action"] for r in sample])
 plot(C/"threshold-plots.png","Seed 521 deltas",[([delta(r,"DNg100") for r in sample],"#4daf4a"),([delta(r,"MDN") for r in sample],"#984ea3")])
 lm={(r["seed"],r["decision_index"]):r for r in live};nm={(r["seed"],r["decision_index"]):r for r in no};fm={(r["seed"],r["decision_index"]):r for r in frozen}
 curve=[np.mean([abs(delta(lm[(s,i)],"DNg100")-delta(nm[(s,i)],"DNg100"))+abs(delta(lm[(s,i)],"MDN")-delta(nm[(s,i)],"MDN")) for s in range(521,541)]) for i in range(60)]
 plot(C/"live-vs-control-comparison.png","Live vs no-vision combined delta divergence",[(curve,"#377eb8")])
 sample=[r for r in integrated if r["seed"]==521];plot(C/"exploratory-four-direction-action-timeline.png","Exploratory four-direction signals",[([r["steering_state"]["steering_signal_hz"] for r in sample],"#377eb8"),([delta(r,"DNg100") for r in sample],"#4daf4a"),([delta(r,"MDN") for r in sample],"#984ea3")],[r["selected_action"] for r in sample])

def main():
 td=json.loads((R/"trials.json").read_text());cd=json.loads((R/"controls.json").read_text());ic=json.loads((R/"integrated-controller.json").read_text());cal=json.loads((R/"calibration.json").read_text());th=json.loads((R/"thresholds.json").read_text())
 rows=td["records"];groups={c:[r for r in rows if r["condition"]==c] for c in td["metadata"]["conditions"]};summaries={k:summary(v) for k,v in groups.items()};live=groups["live_vision"];no=groups["no_vision"];frozen=groups["frozen_frame"];shuffled=groups["shuffled_vision"]
 comparisons={}
 for i,(name,other) in enumerate((("live_vs_no",no),("live_vs_frozen",frozen),("live_vs_shuffled",shuffled))):
  comparisons[name]={"actions":action_test(live,other,seed=930+i),"DNg100":continuous_test(live,other,"DNg100",seed=940+i),"MDN":continuous_test(live,other,"MDN",seed=950+i)}
 abl=cd["ablations"];ablation={}
 for p,action in (("DNg100","UP"),("MDN","DOWN")):
  sub=[r for r in abl if r["ablation"]==p];base=[r for r in live if r["seed"]<531]
  ablation[p]={"baseline_action_fraction":dist(base)[action],"ablated_action_fraction":dist(sub)[action],"eliminated":dist(sub)[action]==0,"distribution":dist(sub)}
 dynamic=cd["dynamic"];tl=[r for r in dynamic if r["condition"]=="title_live"];tn=[r for r in dynamic if r["condition"]=="title_no_vision"]
 dyn={"live":summary(tl),"no_vision":summary(tn),"actions":action_test(tl,tn,seed=960),"DNg100":continuous_test(tl,tn,"DNg100",seed=961),"MDN":continuous_test(tl,tn,"MDN",seed=962)}
 integ=ic["records"];cnt=Counter(r["selected_action"] for r in integ);integ_dist={k:cnt[None if k=="NONE" else k]/len(integ) for k in ("LEFT","RIGHT","UP","DOWN","NONE")}
 calstats=calibration_stats(cal,th)
 action_sig=comparisons["live_vs_no"]["actions"]["p_value"]<.05
 useful={}
 for p,action in (("DNg100","UP"),("MDN","DOWN")):
  test=comparisons["live_vs_no"][p];useful[p]=test["effect_hz"]>0 and test["p_value"]<.05 and action_sig and ablation[p]["eliminated"] and summaries["live_vision"][p]["threshold_crossing_rate"]>summaries["no_vision"][p]["threshold_crossing_rate"]
 both=all(useful.values());one=any(useful.values());credible=all(integ_dist[x]>0 for x in ("LEFT","RIGHT","UP","DOWN")) and integ_dist["NONE"]<.95
 if one:verdict="PASS";primary="E" if both and credible else "D" if both else "C";answer="YES" if both and credible else "PARTIALLY"
 else:
  controlled=any(th["validation"][p]["effect_hz"]>0 and th["validation"][p]["p_value"]<.05 for p in ("DNg100","MDN"))
  verdict="WEAK" if controlled else "FAIL";primary="B" if controlled else "A";answer="NO"
 src=(ROOT/"flymon/locomotion.py").read_text().lower();run=(ROOT/"run_locomotion_experiment.py").read_text().lower()
 leakage={"RAM":"read_memory" not in src and "memory[" not in src,"framebuffer_directly_into_controller":".framebuffer" not in src,"class_labels":"class_label" not in src and "screen_class" not in src,
 "player_position":"player_position" not in src and "sprite" not in src,"scripted_route":all(x not in run for x in ("up up","desired_route","route =")),"trained_policy":all(x not in src for x in ("fit(","predict(","sklearn","torch")),"reward":"reward" not in src}
 e7=json.loads((ROOT/"results/experiment-07-closed-loop/statistics.json").read_text());e7cmp={"DNg100_action_rate":e7["summaries"]["live_vision"]["distribution"]["UP"],"MDN_action_rate":e7["summaries"]["live_vision"]["distribution"]["DOWN"],
 "DNg100_live_no_effect":e7["summaries"]["live_vision"]["distribution"]["UP"]-e7["summaries"]["no_vision"]["distribution"]["UP"],"MDN_live_no_effect":e7["summaries"]["live_vision"]["distribution"]["DOWN"]-e7["summaries"]["no_vision"]["distribution"]["DOWN"]}
 perf={k:float(np.mean([r["timing_seconds"][k] for r in rows])) for k in rows[0]["timing_seconds"]}
 stats={"verdict":verdict,"primary_result":primary,"answer":answer,"useful":useful,"calibration":calstats,"thresholds":th,"summaries":summaries,"comparisons":comparisons,"ablations":ablation,
 "dynamic_condition":dyn,"integrated_distribution":integ_dist,"experiment7_comparison":e7cmp,"performance_seconds":perf,"leakage_checks":leakage}
 (R/"statistics.json").write_text(json.dumps(stats,indent=2)+"\n");figures(cal,live,no,frozen,integ)
 L=summaries["live_vision"];N=summaries["no_vision"];F=summaries["frozen_frame"];S=summaries["shuffled_vision"];dtest=comparisons["live_vs_no"]
 pop=json.loads((R/"motor-populations.json").read_text())["populations"]
 final=f"""Experiment 9 verdict:
{verdict}

Calibration seeds:
501-520
Primary test seeds:
521-540

DNg100 population:
cells: {len(pop['DNg100'])}
baseline: {np.mean([r['population_baseline']['dng100_mean_hz'] for r in live]):.4f} Hz
threshold: {th['forward']['selected_multiplier']} SD, floor {th['forward']['sd_floor_hz']:.4f} Hz
natural vision effect: {dtest['DNg100']['effect_hz']:.4f} Hz
p-value: {dtest['DNg100']['p_value']:.4f}

MDN population:
cells: {len(pop['MDN'])}
baseline: {np.mean([r['population_baseline']['mdn_mean_hz'] for r in live]):.4f} Hz
threshold: {th['backward']['selected_multiplier']} SD, floor {th['backward']['sd_floor_hz']:.4f} Hz
natural vision effect: {dtest['MDN']['effect_hz']:.4f} Hz
p-value: {dtest['MDN']['p_value']:.4f}

LOCOMOTION-ONLY LIVE

UP %: {100*L['distribution']['UP']:.3f}
DOWN %: {100*L['distribution']['DOWN']:.3f}
NO ACTION %: {100*L['distribution']['NONE']:.3f}
actions/trial: {L['actions_per_trial']:.3f}

NO VISION

UP %: {100*N['distribution']['UP']:.3f}
DOWN %: {100*N['distribution']['DOWN']:.3f}
NO ACTION %: {100*N['distribution']['NONE']:.3f}

FROZEN

UP %: {100*F['distribution']['UP']:.3f}
DOWN %: {100*F['distribution']['DOWN']:.3f}
NO ACTION %: {100*F['distribution']['NONE']:.3f}

SHUFFLED

UP %: {100*S['distribution']['UP']:.3f}
DOWN %: {100*S['distribution']['DOWN']:.3f}
NO ACTION %: {100*S['distribution']['NONE']:.3f}

DNg100 live-vs-no-vision:
effect: {dtest['DNg100']['effect_hz']:.4f} Hz
p-value: {dtest['DNg100']['p_value']:.4f}

MDN live-vs-no-vision:
effect: {dtest['MDN']['effect_hz']:.4f} Hz
p-value: {dtest['MDN']['p_value']:.4f}

DNg100 ablation:
result: UP {100*ablation['DNg100']['baseline_action_fraction']:.3f}% -> {100*ablation['DNg100']['ablated_action_fraction']:.3f}%

MDN ablation:
result: DOWN {100*ablation['MDN']['baseline_action_fraction']:.3f}% -> {100*ablation['MDN']['ablated_action_fraction']:.3f}%

Experiment 7 DNg100 action rate: {100*e7cmp['DNg100_action_rate']:.3f}%
Experiment 9 DNg100 action rate: {100*L['distribution']['UP']:.3f}%

Experiment 7 MDN action rate: {100*e7cmp['MDN_action_rate']:.3f}%
Experiment 9 MDN action rate: {100*L['distribution']['DOWN']:.3f}%

Exploratory four-direction controller:
LEFT %: {100*integ_dist['LEFT']:.3f}
RIGHT %: {100*integ_dist['RIGHT']:.3f}
UP %: {100*integ_dist['UP']:.3f}
DOWN %: {100*integ_dist['DOWN']:.3f}
NONE %: {100*integ_dist['NONE']:.3f}

Primary result:
{primary}

Leakage checks:
RAM: {'PASS' if leakage['RAM'] else 'FAIL'}
framebuffer directly into controller: {'PASS' if leakage['framebuffer_directly_into_controller'] else 'FAIL'}
class labels: {'PASS' if leakage['class_labels'] else 'FAIL'}
player position: {'PASS' if leakage['player_position'] else 'FAIL'}
scripted route: {'PASS' if leakage['scripted_route'] else 'FAIL'}
trained policy: {'PASS' if leakage['trained_policy'] else 'FAIL'}
reward: {'PASS' if leakage['reward'] else 'FAIL'}

Is Flymon ready for unrestricted four-direction free exploration in Experiment 10?
{answer}
"""
 interp={"A":"Neither DNg100 nor MDN provided useful visually modulated locomotion.","B":"Controlled stimuli modulated locomotor DNs, but natural Pokemon vision did not.","C":"One locomotion channel generalized to natural Pokemon vision.","D":"Both UP and DOWN channels were significantly visually modulated.","E":"The exploratory fixed BCI credibly combined all four directions."}[primary]
 (R/"analysis.md").write_text("# Experiment 9 analysis\n\n```text\n"+final+"```\n\n## Interpretation\n\n"+interp+"\n\nThe integrated run is exploratory and did not determine the verdict. No navigation outcome was scored.\n")
 print(final)
if __name__=="__main__":main()
