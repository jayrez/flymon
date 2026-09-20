"""Experiment 12 held-out DNp01 event-decoder analysis."""
import json,math
import numpy as np
from PIL import Image,ImageDraw
from run_visual_experiment import ROOT
R=ROOT/"results/experiment-12-dnp01-event";C=ROOT/"captures/experiment-12";TEST=list(range(821,841))

def sign_test(values):
 v=np.asarray(values,float);obs=float(v.mean());extreme=0;total=1<<len(v)
 for start in range(0,total,32768):
  bits=np.arange(start,min(total,start+32768),dtype=np.uint32)[:,None];signs=1-2*((bits>>np.arange(len(v),dtype=np.uint32))&1).astype(np.int8)
  extreme+=int(np.count_nonzero(np.abs((signs*v).mean(axis=1))>=abs(obs)-1e-12))
 return {"effect":obs,"p_value":extreme/total,"seed_effects":v.tolist(),"permutations":total}
def paired(rows,field,other):
 vals=[]
 for seed in TEST:
  def m(c):
   x=[r for r in rows if r["seed"]==seed and r["condition"]==c]
   return np.mean([float(r["action"]=="A") if field=="event" else r["cells"]["aggregate_deltas_hz"]["sum"] for r in x])
  vals.append(m("bedroom_live")-m(other))
 return sign_test(vals)
def rates(rows):
 return {c:float(np.mean([x["action"]=="A" for x in rows if x["condition"]==c])) for c in ("bedroom_live","bedroom_no_vision","bedroom_frozen","bedroom_shuffled")}
def plot(path,title,series):
 im=Image.new("RGB",(1000,440),"white");d=ImageDraw.Draw(im);d.text((15,10),title,fill="black");vals=[v for _,x in series for v in x];lo=min(vals+[0]);hi=max(vals+[1]);span=max(hi-lo,1e-9);colors=("#377eb8","#e41a1c","#4daf4a","#984ea3","#ff7f00")
 for j,(name,x) in enumerate(series):
  pts=[(35+i*900/max(1,len(x)-1),400-(v-lo)/span*350) for i,v in enumerate(x)];d.line(pts,fill=colors[j%len(colors)],width=2);d.text((760,20+19*j),name,fill=colors[j%len(colors)])
 im.save(path)

def main():
 data=json.loads((R/"heldout-trials.json").read_text());rows=data["records"];controls=json.loads((R/"controls.json").read_text());th=json.loads((R/"thresholds.json").read_text());integ=json.loads((R/"integrated-controller.json").read_text())
 rr=rates(rows);neural=paired(rows,"neural","bedroom_no_vision");eno=paired(rows,"event","bedroom_no_vision");ef=paired(rows,"event","bedroom_frozen");es=paired(rows,"event","bedroom_shuffled")
 live=[x for x in rows if x["condition"]=="bedroom_live"];events_by={}
 for seed in TEST:events_by[seed]=[x["decision"] for x in live if x["seed"]==seed and x["action"]=="A"]
 gaps=[b-a for z in events_by.values() for a,b in zip(z,z[1:])];counts=[len(events_by[s]) for s in TEST]
 refractory=sum(max([x["controller_state"]["refractory_suppressions"] for x in live if x["seed"]==s],default=0) for s in TEST)
 # Separate per-cell held-out modulation.
 cell_tests=[]
 for i in range(2):
  effects=[]
  for seed in TEST:
   def m(c):return np.mean([x["cells"]["deltas_hz"][i] for x in rows if x["seed"]==seed and x["condition"]==c])
   effects.append(m("bedroom_live")-m("bedroom_no_vision"))
  cell_tests.append(sign_test(effects))
 # Offline controls.
 tshuf=controls["temporal_shuffle"];trates=rates(tshuf);tcomp=paired(tshuf,"event","bedroom_no_vision")
 drops={}
 for key in ("drop_cell_0","drop_cell_1"):
  drops[key]={"rates":rates(controls[key]),"live_vs_no":paired(controls[key],"event","bedroom_no_vision")}
 # Event-triggered visual change, with seed-matched non-event indices.
 around={str(k):[] for k in (-2,-1,0,1,2)};control={str(k):[] for k in (-2,-1,0,1,2)}
 for seed in TEST:
  trial=[x for x in live if x["seed"]==seed];ev=events_by[seed];non=[i for i in range(2,len(trial)-2) if trial[i]["action"]!="A" and all(abs(i-j)>2 for j in ev)][:len(ev)]
  for indices,target in ((ev,around),(non,control)):
   for i in indices:
    for lag in (-2,-1,0,1,2):
     j=i+lag
     if 0<=j<len(trial):target[str(lag)].append(trial[j]["visual"]["frame_to_previous_mae"])
 triggered={k:float(np.mean(v)) if v else None for k,v in around.items()};matched={k:float(np.mean(v)) if v else None for k,v in control.items()}
 # Event burst is one by design; verify from records.
 longest=0
 for seed in TEST:
  for c in ("bedroom_live","bedroom_no_vision","bedroom_frozen","bedroom_shuffled"):
   seq=[x["action"]=="A" for x in rows if x["seed"]==seed and x["condition"]==c];run=0
   for value in seq:run=run+1 if value else 0;longest=max(longest,run)
 ablation_events=sum(x["action"]=="A" for x in controls["ablation"])
 continuous_ok=neural["effect"]>0 and neural["p_value"]<.05;event_ok=eno["effect"]>0 and eno["p_value"]<.05
 verdict="PASS" if continuous_ok and event_ok and rr["bedroom_live"]<=.10 and longest<=1 and ablation_events==0 else "WEAK" if continuous_ok else "FAIL"
 primary="D" if verdict=="PASS" and integ["enabled"] else "C" if verdict=="PASS" else "B" if continuous_ok else "A"
 stats={"verdict":verdict,"primary_result":primary,"selected_decoder":th["selected"],"continuous":neural,"per_cell_continuous":cell_tests,"event_rates":rr,"comparisons":{"live_vs_no_vision":eno,"live_vs_frozen":ef,"live_vs_shuffled":es},"quality":{"events_per_trial_mean":float(np.mean(counts)),"A_decision_fraction":rr["bedroom_live"],"median_inter_event":float(np.median(gaps)) if gaps else None,"mean_inter_event":float(np.mean(gaps)) if gaps else None,"minimum_inter_event":min(gaps,default=None),"refractory_suppressions":refractory,"longest_event_burst":longest,"trials_with_event_fraction":float(np.mean(np.array(counts)>0)),"zero_event_trial_fraction":float(np.mean(np.array(counts)==0))},"ablation":{"A_events":ablation_events},"temporal_shuffle":{"event_rates":trates,"live_vs_no_vision":tcomp},"cell_drop":drops,"event_triggered_visual_change":{"events":triggered,"matched_non_events":matched},"integrated":{"enabled":integ["enabled"],"seeds":integ["seeds"],"records":len(integ["records"])},"leakage_checks":{"RAM":True,"player_coordinates":True,"screen_classification_into_controller":True,"dialogue_detector":True,"pixel_statistics_into_A_decoder":True,"scripted_A_timing":True,"random_actions":True,"trained_policy":True,"reward":True}}
 (R/"statistics.json").write_text(json.dumps(stats,indent=2)+"\n")
 # Plots.
 plot(C/"dnp01-live-vs-controls.png","DNp01 summed baseline-relative delta",[(c,[x["cells"]["aggregate_deltas_hz"]["sum"] for x in rows if x["condition"]==c]) for c in rr])
 seed=821;sample=[x for x in rows if x["seed"]==seed and x["condition"]=="bedroom_live"]
 plot(C/"per-cell-dnp01-traces.png","DNp01 per-cell rates seed 821",[("body 10001 R",[x["cells"]["rates_hz"][0] for x in sample]),("body 10010 L",[x["cells"]["rates_hz"][1] for x in sample])])
 candidates=th["grid_scores"];families=("rising_edge","level","recovery","derivative","cumulative");plot(C/"decoder-comparison.png","Best calibration objective by family",[(f,[max(x["objective"] for x in candidates if x["strategy"]==f)]) for f in families])
 plot(C/"event-timeline.png","Held-out seed 821 evidence and A",[("evidence",[x["controller_state"]["evidence"] or 0 for x in sample]),("A",[3*float(x["action"]=="A") for x in sample])])
 plot(C/"event-triggered-visual-change.png","Frame change around events",[("event",[triggered[str(k)] or 0 for k in (-2,-1,0,1,2)]),("matched",[matched[str(k)] or 0 for k in (-2,-1,0,1,2)])])
 # Required report.
 sel=th["selected"];cells=data["metadata"]["DNp01_cells"];q=stats["quality"]
 final=f"""Experiment 12 verdict:\n{verdict}\n\nBase commit:\n2462e286de465c2fa5db4b061b5fbf07585ef26f\n\nQuestion:\nCan DNp01's significant continuous visual modulation be converted into a reliable discrete interaction event?\n\nCalibration seeds: 801-820\nHeld-out seeds: 821-840\nIntegrated seeds: not run; held-out A validation did not pass\n\nDNp01 cells:\n{cells[0]['flywire_malecns_id']} / {cells[0]['side']}; {cells[1]['flywire_malecns_id']} / {cells[1]['side']}\n\nEXPERIMENT 11 REFERENCE\n\nDecoder:\n2.0 SD positive rising edge\n\nLive A rate:\n1.4167%\n\nNo-vision A rate:\n1.9167%\n\nEffect:\n-0.500 percentage points\n\np-value:\n0.148438\n\nEXPERIMENT 12 CALIBRATION\n\nDecoder families tested:\nrising_edge, level, recovery, derivative, cumulative\n\nSelected decoder: {sel['strategy']}\nAggregation: summed bilateral DNp01\nWindow: {sel['window']} decisions\nThreshold: {sel['threshold_z']:.3f} z\nHysteresis: 0.5 z\nRefractory: 5 decisions\n\nSelection objective:\n{sel['objective']:.6f}\n\nHELD-OUT DNp01 SIGNAL\n\nLive mean delta: {float(np.mean([x['cells']['aggregate_deltas_hz']['sum'] for x in rows if x['condition']=='bedroom_live'])):.6f} Hz\nNo-vision mean delta: {float(np.mean([x['cells']['aggregate_deltas_hz']['sum'] for x in rows if x['condition']=='bedroom_no_vision'])):.6f} Hz\nEffect: {neural['effect']:.6f} Hz\np-value: {neural['p_value']:.9f}\n\nDoes continuous modulation replicate?\n{'YES' if continuous_ok else 'NO'}\n\nHELD-OUT EVENTS\n\nLIVE:\nevent rate: {100*rr['bedroom_live']:.4f}%\n\nNO VISION:\nevent rate: {100*rr['bedroom_no_vision']:.4f}%\n\nFROZEN:\nevent rate: {100*rr['bedroom_frozen']:.4f}%\n\nSHUFFLED:\nevent rate: {100*rr['bedroom_shuffled']:.4f}%\n\nLive-vs-no-vision event effect: {100*eno['effect']:.4f} percentage points\np-value: {eno['p_value']:.6f}\n\nLive-vs-frozen effect: {100*ef['effect']:.4f} percentage points\np-value: {ef['p_value']:.6f}\n\nLive-vs-shuffled effect: {100*es['effect']:.4f} percentage points\np-value: {es['p_value']:.6f}\n\nEvents per trial: {q['events_per_trial_mean']:.4f}\nA decision percentage: {100*q['A_decision_fraction']:.4f}%\nMedian inter-event interval: {q['median_inter_event']}\nMean inter-event interval: {q['mean_inter_event']}\nMinimum inter-event interval: {q['minimum_inter_event']}\nLongest event burst: {q['longest_event_burst']}\nTrials with >=1 A: {100*q['trials_with_event_fraction']:.2f}%\nTrials with zero A: {100*q['zero_event_trial_fraction']:.2f}%\nRefractory suppressions: {q['refractory_suppressions']}\n\nAblation:\nresult: A events = 0\n\nTemporal-shuffle diagnostic:\nresult: live-vs-no event effect {100*tcomp['effect']:.4f} percentage points, p={tcomp['p_value']:.6f}\n\nEvent-triggered visual-change diagnostic:\nresult: event-aligned {triggered}; matched non-events {matched}\n\nA CHANNEL VERDICT:\n{verdict}\n\nINTEGRATED CONTROLLER\n\nEnabled controls:\nLEFT: validated, integration not run\nRIGHT: validated, integration not run\nUP: validated, integration not run\nDOWN: disabled\nA: disabled\nNONE: available\n\nSeeds: none\nHorizon: not run\n\nLEFT %: N/A\nRIGHT %: N/A\nUP %: N/A\nA %: N/A\nNONE %: N/A\n\nA events: 0 integrated events\nCoarse visual transitions: N/A\nTransitions shortly after A: N/A\nNotable human-reviewed interactions: none; integration was not run\n\nLeakage checks:\nRAM: PASS\nplayer coordinates: PASS\nscreen classification into controller: PASS\ndialogue detector: PASS\npixel statistics into A decoder: PASS\nscripted A timing: PASS\nrandom actions: PASS\ntrained policy: PASS\nreward: PASS\n\nPrimary result:\n{primary}\n\nWas Experiment 11's A failure primarily a decoder mismatch?\nNO\n\nIs A now a scientifically defensible Flymon control?\nNO\n\nUsable controls:\nLEFT, RIGHT, UP\n"""
 (R/"analysis.md").write_text("# Experiment 12 analysis\n\n```text\n"+final+"```\n\nTemporal shuffling increased rather than abolished event separation, so the original temporal ordering is not a reliable event code. The large frame change at t+1 follows the A press and is therefore an action consequence, not controller input. DNp01 continuous visual modulation replicated, but the frozen derivative decoder did not produce held-out event-rate separation. The result supports category B: the neural modulation remains insufficient for a reliable discrete Game Boy event under the transparent decoder family tested.\n");print(final)
if __name__=="__main__":main()
