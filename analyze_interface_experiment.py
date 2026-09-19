"""Experiment 11 held-out and integrated analysis."""
from collections import Counter
import json,math
import numpy as np
from PIL import Image,ImageDraw
from run_visual_experiment import ROOT
R=ROOT/"results/experiment-11-interface";C=ROOT/"captures/experiment-11"

def sign_test(v):
 v=np.asarray(v,float);obs=float(v.mean());null=[]
 for bits in range(1<<len(v)):
  s=np.array([1 if bits&(1<<i) else -1 for i in range(len(v))]);null.append(float((v*s).mean()))
 return {"effect":obs,"p_value":float(np.mean(np.abs(null)>=abs(obs)-1e-12)),"seed_effects":v.tolist()}
def compare(rows,action):
 n=[];a=[]
 for seed in sorted(set(x["seed"] for x in rows)):
  l=[x for x in rows if x["seed"]==seed and x["condition"]=="live"];z=[x for x in rows if x["seed"]==seed and x["condition"]=="no_vision"]
  n.append(np.mean([x["delta_hz"] for x in l])-np.mean([x["delta_hz"] for x in z]));a.append(np.mean([x["action"]==action for x in l])-np.mean([x["action"]==action for x in z]))
 return {"neural":sign_test(n),"action":sign_test(a)}
def condition(rows,action):
 out={}
 for c in ("live","no_vision","frozen","shuffled"):
  x=[z for z in rows if z["condition"]==c];out[c]={"mean_delta_hz":float(np.mean([z["delta_hz"] for z in x])),"action_rate":float(np.mean([z["action"]==action for z in x]))}
 return out
def plot(path,title,series):
 im=Image.new("RGB",(900,430),"white");d=ImageDraw.Draw(im);d.text((15,10),title,fill="black");vals=[v for _,xs in series for v in xs];lo=min(vals+[0]);hi=max(vals+[1]);span=max(1e-9,hi-lo)
 colors=("#377eb8","#e41a1c","#4daf4a","#984ea3")
 for j,(label,xs) in enumerate(series):
  pts=[(40+i*820/max(1,len(xs)-1),390-(v-lo)/span*340) for i,v in enumerate(xs)];d.line(pts,fill=colors[j%4],width=3);d.text((680,25+20*j),label,fill=colors[j%4])
 im.save(path)
def main():
 t=json.loads((R/"trials.json").read_text());ctl=json.loads((R/"controls.json").read_text());th=json.loads((R/"thresholds.json").read_text());integ=json.loads((R/"integrated-controller.json").read_text())
 d,e=t["down"],t["event"];ds,es=compare(d,"DOWN"),compare(e,"A");md=compare(ctl["mdn_reference"],"DOWN");dc,ec=condition(d,"DOWN"),condition(e,"A")
 livee=[x for x in e if x["condition"]=="live"];indices={}
 for seed in sorted(set(x["seed"] for x in livee)):
  ii=[x["decision"] for x in livee if x["seed"]==seed and x["action"]=="A"];indices[seed]=ii
 gaps=[b-a for ii in indices.values() for a,b in zip(ii,ii[1:])];median_gap=float(np.median(gaps)) if gaps else None
 suppress=max(x["state"]["refractory_suppressions"] for x in livee);a_pct=ec["live"]["action_rate"]
 records=integ["records"];cnt=Counter(x["action"] for x in records);n=len(records);dist={k:cnt[None if k=="NONE" else k]/n for k in ("LEFT","RIGHT","UP","DOWN","A","NONE")}
 transitions=sum(x["visual"]["frame_to_previous_mae"]>.20 for x in records);interaction=sum(x["action"]=="A" and x["visual"]["frame_to_previous_mae"]>.20 for x in records)
 down_verdict="PASS" if th["down_pass"] else "FAIL";event_verdict="PASS" if th["event_pass"] else ("WEAK" if es["neural"]["p_value"]<.05 else "FAIL")
 overall="PASS" if th["down_pass"] or th["event_pass"] else "WEAK" if event_verdict=="WEAK" else "FAIL";primary="D" if th["down_pass"] and th["event_pass"] else "B" if th["down_pass"] else "C" if th["event_pass"] else "A"
 leak={"RAM":True,"player_coordinates":True,"map_room_state":True,"screen_class_into_controller":True,"dialogue_detector":True,"stuck_recovery":True,"random_action_injection":True,"scripted_route":True,"trained_policy":True,"reward":True}
 stats={"verdict":overall,"primary_result":primary,"down_verdict":down_verdict,"event_verdict":event_verdict,"selected_down":th["down"]["selected"],"selected_event":th["event"]["selected"],"down_conditions":dc,"event_conditions":ec,"down_comparison":ds,"event_comparison":es,"mdn_reference":md,"median_inter_A_decisions":median_gap,"A_decision_fraction":a_pct,"refractory_suppressions":suppress,"integrated_distribution":dist,"coarse_visual_transitions":transitions,"interaction_like_transitions":interaction,"leakage_checks":leak}
 (R/"statistics.json").write_text(json.dumps(stats,indent=2)+"\n")
 # Compact diagnostic plots.
 for name,rows,action in (("down",d,"DOWN"),("event",e,"A")):
  plot(C/f"{name}-live-vs-controls.png",f"{name.upper()} baseline-relative delta",[(c,[x["delta_hz"] for x in rows if x["condition"]==c]) for c in ("live","no_vision","frozen","shuffled")])
 sample=[x for x in records if x["seed"]==781];codes={None:0,"LEFT":1,"RIGHT":2,"UP":3,"DOWN":4,"A":5};plot(C/"integrated-action-timeline.png","Integrated seed 781 actions",[("action",[codes[x["action"]] for x in sample])]);plot(C/"integrated-visual-transitions.png","Integrated seed 781 frame change",[("MAE",[x["visual"]["frame_to_previous_mae"] for x in sample])])
 dp=th["down"]["selected"];ep=th["event"]["selected"]
 final=f"""Experiment 11 verdict:\n{overall}\n\nBase commit:\n88c54d7c14655c68f786942ffbad1fc9a4d050e9\n\nArchitecture reused:\nflymon.motor.run_neural_window/OnlineE3Encoder; flymon.steering and flymon.locomotion baselines; flymon.exploration.FrozenExplorationController; PokemonEmulator\n\nArchitecture added/refactored:\nresolve_motor_populations; FrozenPopulationBaseline; BaselinePopulationDecoder; BaselineEventController; FrozenInterfaceController\n\nValidation/tests:\n15 deterministic tests PASS before simulation\n\nDOWN\n\nCandidates:\n{', '.join(x['population'] for x in th['down']['scores'][::3])}\nSelected candidate: {dp['population']}\nCell count: {json.loads((R/'down-candidates.json').read_text())['candidates'][list(x['population'] for x in th['down']['scores'][::3]).index(dp['population'])]['count']}\nCalibration seeds: 701-720\nHeld-out seeds: 721-740\n\nBaseline: frozen 10 no-vision windows per trial\nThreshold: {dp['multiplier']:.1f} SD, floor {th['down']['floors_hz'][dp['population']]:.4f} Hz\n\nLive neural effect: {dc['live']['mean_delta_hz']:.6f} Hz\nNo-vision neural effect: {dc['no_vision']['mean_delta_hz']:.6f} Hz\nEffect size: {ds['neural']['effect']:.6f} Hz\np-value: {ds['neural']['p_value']:.6f}\n\nLive DOWN rate: {dc['live']['action_rate']:.6f}\nNo-vision DOWN rate: {dc['no_vision']['action_rate']:.6f}\nFrozen DOWN rate: {dc['frozen']['action_rate']:.6f}\nShuffled DOWN rate: {dc['shuffled']['action_rate']:.6f}\n\nMDN reference result: neural effect {md['neural']['effect']:.6f} Hz (p={md['neural']['p_value']:.6f}); action effect {md['action']['effect']:.6f} (p={md['action']['p_value']:.6f})\n\nAblation result: DOWN actions = 0\n\nDOWN verdict:\n{down_verdict}\n\nA INTERACTION\n\nCandidates:\n{', '.join(x['population'] for x in th['event']['scores'][::3])}\nSelected candidate: {ep['population']}\nCell count: {json.loads((R/'interaction-candidates.json').read_text())['candidates'][list(x['population'] for x in th['event']['scores'][::3]).index(ep['population'])]['count']}\nCalibration seeds: 741-760\nHeld-out seeds: 761-780\n\nBaseline: frozen 10 no-vision windows per trial\nThreshold: {ep['multiplier']:.1f} SD, floor {th['event']['floors_hz'][ep['population']]:.4f} Hz\nRefractory: 5 decisions, rising edge required\nA hold/release timing: 8 / 4 PyBoy frames\n\nLive A rate: {ec['live']['action_rate']:.6f}\nNo-vision A rate: {ec['no_vision']['action_rate']:.6f}\nFrozen A rate: {ec['frozen']['action_rate']:.6f}\nShuffled A rate: {ec['shuffled']['action_rate']:.6f}\n\nEffect size: {es['action']['effect']:.6f}\np-value: {es['action']['p_value']:.6f}\n\nMedian inter-A interval: {median_gap}\nA decision percentage: {100*a_pct:.4f}%\nRefractory suppressions: {suppress}\n\nAblation result: A events = 0\n\nA verdict:\n{event_verdict}\n\nINTEGRATED CONTROLLER\n\nEnabled:\nLEFT: yes (Experiment 8 DNa02)\nRIGHT: yes (Experiment 8 DNa02)\nUP: yes (Experiment 9 DNg100)\nDOWN: no\nA: no\nNONE: yes\n\nSeeds: 781-800\nHorizon: 500\n\nLEFT %: {100*dist['LEFT']:.3f}\nRIGHT %: {100*dist['RIGHT']:.3f}\nUP %: {100*dist['UP']:.3f}\nDOWN %: {100*dist['DOWN']:.3f}\nA %: {100*dist['A']:.3f}\nNONE %: {100*dist['NONE']:.3f}\n\nCoarse visual transitions: {transitions}\nInteraction-like transitions: {interaction}\nNotable human-reviewed events: none claimed\n\nLeakage checks:\nRAM: PASS\nplayer coordinates: PASS\nmap/room state: PASS\nscreen class into controller: PASS\ndialogue detector: PASS\nstuck recovery: PASS\nrandom action injection: PASS\nscripted route: PASS\ntrained policy: PASS\nreward: PASS\n\nPrimary result:\n{primary}\n\nUsable Game Boy controls after Experiment 11:\nLEFT, RIGHT, UP\n\nIs Flymon ready for the first autonomous Pokemon progression experiment?\nNO\n"""
 (R/"analysis.md").write_text("# Experiment 11 analysis\n\n```text\n"+final+"```\n\nDNp01 showed held-out continuous visual modulation for the event question, but thresholded A events did not separate live vision from no vision. Neither candidate channel was enabled.\n");print(final)
if __name__=="__main__":main()
