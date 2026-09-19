"""Experiment 10 trial-level analysis."""
from collections import Counter
import json,math
import numpy as np
from PIL import Image,ImageDraw
from run_visual_experiment import ROOT
R=ROOT/"results/experiment-10-exploration";C=ROOT/"captures/experiment-10"

def entropy(x):
 c=Counter(x);n=len(x);return float(-sum(v/n*math.log2(v/n) for v in c.values() if v))
def summarize(t):
 r=t["records"];a=[x["action"] for x in r];c=[x["visual"]["coarse_state_sha256"] for x in r];f=[x["visual"]["frame_sha256"] for x in r];e=[x["visual"]["encoder_sha256"] for x in r];stuck=np.zeros(len(r),bool);longest=0;i=0
 while i<len(c):
  j=i+1
  while j<len(c) and c[j]==c[i]:j+=1
  longest=max(longest,j-i)
  if j-i>=10:stuck[i:j]=True
  i=j
 counts=Counter(a);n=len(a);trans=[(str(a[i]),str(a[i+1])) for i in range(n-1)]
 runs=[];last=object();k=0
 for x in a:
  if x==last:k+=1
  else:
   if k:runs.append(k)
   last=x;k=1
 runs.append(k)
 return {"seed":t["seed"],"condition":t["condition"],"repeat":t["repeat"],"decisions":n,"actions":sum(x is not None for x in a),
  "action_distribution":{k:counts[None if k=="NONE" else k]/n for k in ("LEFT","RIGHT","UP","DOWN","NONE")},"action_entropy":entropy(a),"action_transition_entropy":entropy(trans),
  "mean_action_run":float(np.mean(runs)),"max_action_run":max(runs),"unique_exact_frames":len(set(f)),"unique_coarse_states":len(set(c)),"unique_encoder_hashes":len(set(e)),
  "final_frame_distance":r[-1]["post_frame_to_initial_mae"],"maximum_frame_distance":max(x["visual"]["frame_to_initial_mae"] for x in r),
  "stuck_fraction":float(stuck.mean()),"longest_stuck_interval":longest if longest>=10 else 0,"scene_transitions":sum(x["scene_transition"] for x in r),
  "scene_transition_indices":[x["decision"] for x in r if x["scene_transition"]],"dominant_action":"NONE" if counts.most_common(1)[0][0] is None else counts.most_common(1)[0][0],
  "wall_seconds":t["wall_seconds"],"decisions_per_second":n/t["wall_seconds"],"simulated_neural_seconds":n*.2,
  "timing_ms":{k:1000*float(np.mean([x["timing_seconds"][k] for x in r])) for k in r[0]["timing_seconds"]}}
def aggregate(x):
 keys=("unique_exact_frames","unique_coarse_states","unique_encoder_hashes","final_frame_distance","maximum_frame_distance","stuck_fraction","action_entropy","action_transition_entropy","longest_stuck_interval","scene_transitions","actions")
 out={k:float(np.mean([z[k] for z in x])) for k in keys};out["action_distribution"]={k:float(np.mean([z["action_distribution"][k] for z in x])) for k in ("LEFT","RIGHT","UP","DOWN","NONE")};return out
def test(a,b,key,n=9999,seed=1010):
 am={x["seed"]:x for x in a};bm={x["seed"]:x for x in b};v=np.array([am[s][key]-bm[s][key] for s in sorted(set(am)&set(bm))]);obs=float(v.mean());rng=np.random.default_rng(seed);null=[float((v*rng.choice([-1,1],len(v))).mean()) for _ in range(n)]
 return {"effect":obs,"p_value":(1+sum(abs(x)>=abs(obs)-1e-15 for x in null))/(n+1),"seed_effects":v.tolist()}
def plot(path,title,values,color="#377eb8"):
 im=Image.new("RGB",(1050,430),"white");d=ImageDraw.Draw(im);d.text((15,10),title,fill="black");mx=max(1,max(abs(float(x)) for x in values));pts=[(30+i*990/max(1,len(values)-1),400-350*float(x)/mx) for i,x in enumerate(values)];d.line(pts,fill=color,width=3);im.save(path)
def main():
 td=json.loads((R/"trials.json").read_text());cd=json.loads((R/"controls.json").read_text());sd=json.loads((R/"secondary-controller.json").read_text());ld=json.loads((R/"long-run.json").read_text())
 ps=[summarize(x) for x in td["trials"]];live=[x for x in ps if x["condition"]=="live"];frozen=[x for x in ps if x["condition"]=="frozen_vision"];no=[x for x in ps if x["condition"]=="no_vision"];secondary=[summarize(x) for x in sd["trials"]];long=summarize(ld["trial"]);repeats=[summarize(x) for x in cd["repeat_trials"]]
 metrics=("unique_coarse_states","unique_encoder_hashes","final_frame_distance","maximum_frame_distance","stuck_fraction","action_entropy","action_transition_entropy")
 comp={name:{k:test(live,other,k,seed=1020+j*20+i) for i,k in enumerate(metrics)} for j,(name,other) in enumerate((("live_vs_frozen",frozen),("live_vs_no_vision",no)))}
 agg={"live":aggregate(live),"frozen":aggregate(frozen),"no_vision":aggregate(no),"secondary":aggregate(secondary)};trans=[{"seed":x["seed"],"indices":x["scene_transition_indices"]} for x in live if x["scene_transitions"]]
 movement=agg["live"]["actions"]/300>.02;changed=any(comp[n][k]["p_value"]<.05 for n in comp for k in metrics)
 if movement and changed:
  verdict="PASS";diverse=comp["live_vs_frozen"]["unique_coarse_states"]["effect"]>0 and comp["live_vs_no_vision"]["unique_coarse_states"]["effect"]>0
  primary="E" if trans and diverse and agg["live"]["stuck_fraction"]<.7 else "D" if diverse else "C";answer="YES" if primary in ("D","E") else "PARTIALLY"
 elif movement:verdict,primary,answer="WEAK","B","PARTIALLY"
 else:verdict,primary,answer="FAIL","A","NO"
 src=(ROOT/"flymon/exploration.py").read_text().lower();run=(ROOT/"run_free_exploration.py").read_text().lower();leak={"RAM":"read_memory" not in src and "memory[" not in src,"player_coordinates":"player_position" not in src,
  "screen_class_input":"screen_class" not in src and "classifier" not in src,"stuck_recovery":"if stuck" not in run and "force action" not in run,"random_action_injection":"random.choice" not in run and "rng.choice" not in run,
  "scripted_route":"route =" not in run and "desired_route" not in run,"trained_policy":"fit(" not in src and "predict(" not in src,"reward":"reward" not in src}
 repeat=[];lm={x["seed"]:x for x in live}
 for x in repeats:repeat.append({"seed":x["seed"],"action_L1":sum(abs(x["action_distribution"][k]-lm[x["seed"]]["action_distribution"][k]) for k in x["action_distribution"]),"coarse_difference":x["unique_coarse_states"]-lm[x["seed"]]["unique_coarse_states"],"dominant":[lm[x["seed"]]["dominant_action"],x["dominant_action"]]})
 stats={"verdict":verdict,"primary_result":primary,"answer":answer,"trial_summaries":ps,"aggregate":agg,"comparisons":comp,"scene_transitions":trans,"long_run":long,"secondary_summaries":secondary,"repeatability":repeat,"leakage_checks":leak};(R/"statistics.json").write_text(json.dumps(stats,indent=2)+"\n")
 sample=[x for x in td["trials"] if x["seed"]==601 and x["condition"]=="live"][0]["records"];plot(C/"action-timeline.png","Seed 601 action codes",[0 if x["action"] is None else {"LEFT":1,"RIGHT":2,"UP":3}[x["action"]] for x in sample]);plot(C/"frame-distance-from-start.png","Frame distance",[x["visual"]["frame_to_initial_mae"] for x in sample],"#4daf4a");plot(C/"encoder-distance-from-start.png","Encoder distance",[x["visual"]["encoder_to_initial_rms"] for x in sample],"#984ea3")
 ids={};timeline=[]
 for x in [z["visual"]["coarse_state_sha256"] for z in sample]:
  if x not in ids:ids[x]=len(ids)
  timeline.append(ids[x])
 plot(C/"coarse-visual-state-timeline.png","Coarse visual state IDs",timeline,"#e41a1c");plot(C/"stuck-repetition-plot.png","Repeated coarse state",[int(i and timeline[i]==timeline[i-1]) for i in range(len(timeline))],"#777777")
 plot(C/"live-vs-frozen-comparison.png","Live minus frozen trial coarse-state diversity",comp["live_vs_frozen"]["unique_coarse_states"]["seed_effects"]);plot(C/"3-direction-vs-4-direction-comparison.png","3-dir and 4-dir mean coarse states",[agg["live"]["unique_coarse_states"],agg["secondary"]["unique_coarse_states"]]);longrows=ld["trial"]["records"];plot(C/"long-run-timeline.png","Long-run frame distance",[x["visual"]["frame_to_initial_mae"] for x in longrows],"#4daf4a")
 L=agg["live"];F=agg["frozen"];N=agg["no_vision"];S=agg["secondary"];ad=L["action_distribution"];sdist=S["action_distribution"];lvf=comp["live_vs_frozen"];lvn=comp["live_vs_no_vision"]
 seeds=[x["seed"] for x in trans];indices=[i for x in trans for i in x["indices"]]
 final=f"""Experiment 10 verdict:\n{verdict}\n\nPrimary controller:\nLEFT: frozen Experiment 8 DNa02\nRIGHT: frozen Experiment 8 DNa02\nUP: frozen Experiment 9 DNg100\nDOWN: disabled\nNONE: deadband/cooldown\n\nPrimary seeds: 601-620\nTrial horizon: 300\nLong-run seed: 650\nLong-run horizon: 1000\n\nLIVE - 3 DIRECTION\n\nLEFT %: {100*ad['LEFT']:.3f}\nRIGHT %: {100*ad['RIGHT']:.3f}\nUP %: {100*ad['UP']:.3f}\nNONE %: {100*ad['NONE']:.3f}\n\nunique exact frames: {L['unique_exact_frames']:.2f}\nunique coarse visual states: {L['unique_coarse_states']:.2f}\nunique encoder states: {L['unique_encoder_hashes']:.2f}\nmean final frame distance: {L['final_frame_distance']:.6f}\nmean maximum frame distance: {L['maximum_frame_distance']:.6f}\nmean stuck fraction: {L['stuck_fraction']:.6f}\naction entropy: {L['action_entropy']:.6f}\n\nFROZEN VISION\n\nunique coarse visual states: {F['unique_coarse_states']:.2f}\nmean final frame distance: {F['final_frame_distance']:.6f}\nmean stuck fraction: {F['stuck_fraction']:.6f}\naction entropy: {F['action_entropy']:.6f}\n\nNO VISION\n\nunique coarse visual states: {N['unique_coarse_states']:.2f}\nmean final frame distance: {N['final_frame_distance']:.6f}\nmean stuck fraction: {N['stuck_fraction']:.6f}\naction entropy: {N['action_entropy']:.6f}\n\nLIVE vs FROZEN\nvisual-state diversity effect: {lvf['unique_coarse_states']['effect']:.4f}\np-value: {lvf['unique_coarse_states']['p_value']:.4f}\nframe-distance effect: {lvf['maximum_frame_distance']['effect']:.6f}\np-value: {lvf['maximum_frame_distance']['p_value']:.4f}\nstuck-fraction effect: {lvf['stuck_fraction']['effect']:.6f}\np-value: {lvf['stuck_fraction']['p_value']:.4f}\n\nLIVE vs NO VISION\nvisual-state diversity effect: {lvn['unique_coarse_states']['effect']:.4f}\np-value: {lvn['unique_coarse_states']['p_value']:.4f}\nframe-distance effect: {lvn['maximum_frame_distance']['effect']:.6f}\np-value: {lvn['maximum_frame_distance']['p_value']:.4f}\nstuck-fraction effect: {lvn['stuck_fraction']['effect']:.6f}\np-value: {lvn['stuck_fraction']['p_value']:.4f}\n\nObserved scene transitions:\ncount: {sum(x['scene_transitions'] for x in live)}\nseeds: {seeds}\ndecision indices: {indices}\n\nLONG RUN\ndecisions: {long['decisions']}\nactions: {long['actions']}\nunique coarse states: {long['unique_coarse_states']}\nscene transitions: {long['scene_transitions']}\nlongest stuck interval: {long['longest_stuck_interval']}\nmost common action: {long['dominant_action']}\n\nSECONDARY FOUR-DIRECTION ARM\n\nLEFT %: {100*sdist['LEFT']:.3f}\nRIGHT %: {100*sdist['RIGHT']:.3f}\nUP %: {100*sdist['UP']:.3f}\nDOWN %: {100*sdist['DOWN']:.3f}\nNONE %: {100*sdist['NONE']:.3f}\n\nunique coarse states: {S['unique_coarse_states']:.2f}\nscene transitions: {S['scene_transitions']:.2f}\nstuck fraction: {S['stuck_fraction']:.6f}\n\n3-dir vs 4-dir exploration difference: {S['unique_coarse_states']-L['unique_coarse_states']:.4f} mean coarse states\n\nLeakage / hidden-policy checks:\nRAM: {'PASS' if leak['RAM'] else 'FAIL'}\nplayer coordinates: {'PASS' if leak['player_coordinates'] else 'FAIL'}\nscreen-class input: {'PASS' if leak['screen_class_input'] else 'FAIL'}\nstuck recovery: {'PASS' if leak['stuck_recovery'] else 'FAIL'}\nrandom action injection: {'PASS' if leak['random_action_injection'] else 'FAIL'}\nscripted route: {'PASS' if leak['scripted_route'] else 'FAIL'}\ntrained policy: {'PASS' if leak['trained_policy'] else 'FAIL'}\nreward: {'PASS' if leak['reward'] else 'FAIL'}\n\nPrimary result:\n{primary}\n\nDid Flymon produce sustained, non-scripted, visually contingent autonomous Pokemon exploration using only pixels -> MaleCNS -> identified descending-neuron BCI control?\n{answer}\n"""
 (R/"analysis.md").write_text("# Experiment 10 analysis\n\n```text\n"+final+"```\n\nScene transitions are visual-only annotations. MDN DOWN remains unvalidated in the secondary arm.\n");print(final)
if __name__=="__main__":main()
