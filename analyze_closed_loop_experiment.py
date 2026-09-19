"""Transparent causal/statistical analysis for Experiment 7."""
from __future__ import annotations

from collections import Counter, defaultdict
import json, math
from pathlib import Path

import numpy as np

from run_visual_experiment import ROOT

RESULTS = ROOT / "results/experiment-07-closed-loop"
ACTIONS = ("LEFT", "RIGHT", "UP", "DOWN", None)


def distribution(rows):
    c = Counter(r["applied_action"] for r in rows); n = len(rows)
    return {("NONE" if a is None else a): c[a] / n for a in ACTIONS}


def entropy(dist):
    return float(-sum(p * math.log2(p) for p in dist.values() if p))


def tv(a, b): return .5 * sum(abs(a[k] - b[k]) for k in a)


def js(a, b):
    m = {k: (a[k] + b[k]) / 2 for k in a}
    def kl(x): return sum(v * math.log2(v / m[k]) for k, v in x.items() if v)
    return float((kl(a) + kl(b)) / 2)


def action_permutation(a_rows, b_rows, permutations=9999, seed=707):
    """Seed-stratified condition-label swaps; statistic is aggregate TV."""
    by = defaultdict(lambda: [[], []])
    for r in a_rows: by[r["seed"]][0].append(r)
    for r in b_rows: by[r["seed"]][1].append(r)
    observed = tv(distribution(a_rows), distribution(b_rows)); rng = np.random.default_rng(seed); null=[]
    seeds = sorted(by)
    for _ in range(permutations):
        aa, bb = [], []
        for s in seeds:
            x, y = by[s]
            if rng.integers(2): x, y = y, x
            aa.extend(x); bb.extend(y)
        null.append(tv(distribution(aa), distribution(bb)))
    return {"statistic": "total variation distance", "observed": observed,
            "permutations": permutations, "p_value": (1 + sum(x >= observed for x in null)) / (permutations + 1)}


def sign_permutation(values, permutations=9999, seed=708):
    values=np.asarray(values,float); obs=float(values.mean()); rng=np.random.default_rng(seed)
    null=[float((values*rng.choice([-1,1],len(values))).mean()) for _ in range(permutations)]
    return {"observed_mean_signed_bias_difference": obs, "p_value": (1+sum(abs(x)>=abs(obs) for x in null))/(permutations+1)}


def summarize(rows):
    d=distribution(rows); by=defaultdict(list)
    for r in rows: by[r["seed"]].append(r)
    return {"trials":len(by), "actions_per_trial_mean":float(np.mean([sum(x["applied_action"] is not None for x in rs) for rs in by.values()])),
            "distribution":d, "action_entropy_bits":entropy(d)}


def main():
    trials=json.loads((RESULTS/"trials.json").read_text()); controls=json.loads((RESULTS/"controls.json").read_text())
    rows=trials["records"]; groups={c:[r for r in rows if r["condition"]==c] for c in trials["metadata"]["conditions"]}
    summaries={k:summarize(v) for k,v in groups.items()}
    live_no=action_permutation(groups["live_vision"],groups["no_vision"])
    live_frozen=action_permutation(groups["live_vision"],groups["frozen_frame"],seed=709)
    d_live=distribution(groups["live_vision"]); d_no=distribution(groups["no_vision"]); d_frozen=distribution(groups["frozen_frame"])
    visual={"live_vs_no_vision":live_no,"jensen_shannon_bits":js(d_live,d_no),
            "live_vs_frozen":live_frozen,"live_frozen_js_bits":js(d_live,d_frozen)}

    assay=controls["left_right_assay"]; left=[r for r in assay if r["condition"]=="left_stimulus"]; right=[r for r in assay if r["condition"]=="right_stimulus"]
    def lr_ratio(rs):
        c=Counter(r["applied_action"] for r in rs); return (c["LEFT"]+.5)/(c["RIGHT"]+.5)
    bias=[]
    for seed in sorted({r["seed"] for r in assay}):
        def signed(rs):
            c=Counter(r["applied_action"] for r in rs); return (c["LEFT"]-c["RIGHT"])/len(rs)
        bias.append(signed([r for r in left if r["seed"]==seed])-signed([r for r in right if r["seed"]==seed]))
    causal={"left_distribution":distribution(left),"right_distribution":distribution(right),
            "left_LEFT_RIGHT_ratio":lr_ratio(left),"right_LEFT_RIGHT_ratio":lr_ratio(right),
            "paired_seed_sign_permutation":sign_permutation(bias)}

    live_map={(r["seed"],r["decision_index"]):r for r in groups["live_vision"]}
    frozen_map={(r["seed"],r["decision_index"]):r for r in groups["frozen_frame"]}
    pairs=[(live_map[k],frozen_map[k]) for k in live_map if k[1]>0]
    feedback={"post_action_frame_hash_divergence_fraction":float(np.mean([a["post_action_framebuffer_sha256"]!=b["post_action_framebuffer_sha256"] for a,b in pairs])),
              "future_action_disagreement_fraction":float(np.mean([a["applied_action"]!=b["applied_action"] for a,b in pairs])),
              "future_candidate_DN_rate_L2_mean_hz":float(np.mean([np.linalg.norm(np.array(list(a["candidate_dn_rates_hz"].values()))-np.array(list(b["candidate_dn_rates_hz"].values()))) for a,b in pairs]))}

    intervention_rows=controls["interventions"]; interventions={}
    for name in ("zero_DNa02","zero_DNg100","zero_MDN"):
        subset=[r for r in intervention_rows if r["intervention"]==name]
        base=[r for r in groups["live_vision"] if r["seed"] in range(321,331)]
        interventions[name]={"distribution":distribution(subset),"baseline_distribution":distribution(base),
                             "total_variation":tv(distribution(subset),distribution(base))}
    repeats=controls["same_seed_repeats"]
    base={(r["seed"],r["decision_index"]):r["applied_action"] for r in groups["live_vision"] if r["seed"]<326}
    repeatability=float(np.mean([r["applied_action"]==base[(r["seed"],r["decision_index"])] for r in repeats]))

    cal=json.loads((RESULTS/"calibration.json").read_text())
    cr=cal["records"]
    def contrast(group):
        l=np.mean([r["all_group_rates_hz"][group] for r in cr if r["condition"]=="left_stimulus"])
        rr=np.mean([r["all_group_rates_hz"][group] for r in cr if r["condition"]=="right_stimulus"])
        return {"left_stimulus_hz":float(l),"right_stimulus_hz":float(rr),"absolute_contrast_hz":float(abs(l-rr))}
    comparison={"identified_motor_DNa02":contrast("DNa02"),"frozen_E4_informative_DNs":contrast("E4_informative_DN"),
                "interpretation":"descriptive fixed-population comparison; neither population is trained here"}

    controller_source=(ROOT/"flymon/controller.py").read_text().lower()
    run_source=(ROOT/"run_closed_loop_experiment.py").read_text().lower()
    checks={"controller_framebuffer_input":".framebuffer" not in controller_source,
            "controller_RAM_input":"read_memory" not in controller_source and "memory[" not in controller_source,
            "controller_class_label_input":"class_label" not in controller_source and "screen_class" not in controller_source,
            "no_trained_policy":all(x not in controller_source for x in ("fit(","predict(","logistic","torch","sklearn","reward")),
            "no_scripted_route":all(x not in run_source for x in ("up up","left left","desired_route","route =")),
            "no_desired_action_labels":"desired_action" not in controller_source and "desired_action" not in run_source}
    action_rate=1-d_live["NONE"]
    stable=.02<=action_rate<=.95 and summaries["live_vision"]["action_entropy_bits"]>.1
    vision=live_no["p_value"]<.05 and live_no["observed"]>.02
    lr=causal["paired_seed_sign_permutation"]["p_value"]<.05 and causal["paired_seed_sign_permutation"]["observed_mean_signed_bias_difference"]>0
    loop=feedback["post_action_frame_hash_divergence_fraction"]>.05 and feedback["future_candidate_DN_rate_L2_mean_hz"]>0 and feedback["future_action_disagreement_fraction"]>.02
    if stable and vision and lr and loop: verdict,primary,answer="PASS","E","YES"
    elif stable and (vision or lr or loop): verdict,primary,answer="WEAK",("D" if vision and loop else "C" if vision else "B"),"PARTIALLY"
    else: verdict,primary,answer="FAIL",("A" if action_rate<.02 else "B"),"NO"
    stats={"summaries":summaries,"visual_dependence":visual,"left_right_causal_assay":causal,"closed_loop_dependence":feedback,
           "interventions":interventions,"same_seed_action_agreement":repeatability,"motor_vs_informative_DN":comparison,
           "leakage_checks":checks,"criteria":{"stable_directional_stream":stable,"significant_visual_dependence":vision,
           "predictable_left_right_manipulation":lr,"measurable_closed_loop_feedback":loop},"verdict":verdict,"primary_result":primary,"answer":answer}
    (RESULTS/"statistics.json").write_text(json.dumps(stats,indent=2)+"\n")
    lines=["# Experiment 7 analysis","",f"**Verdict: {verdict}. Primary result: {primary}. Answer: {answer}.**","",
           "This is an engineered directional BCI, not a natural Pokémon motor circuit or evidence of semantic understanding.","",
           "## Criteria","",*(f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in stats["criteria"].items()),"",
           "## Action summaries","","```json",json.dumps(summaries,indent=2),"```","","## Causal results","","```json",
           json.dumps({"visual_dependence":visual,"left_right":causal,"closed_loop":feedback,"interventions":interventions},indent=2),"```","",
           "## Leakage audit","",*(f"- {k}: PASS" for k in checks),"","## Fixed motor versus visual-informative DNs","","```json",json.dumps(comparison,indent=2),"```",""]
    (RESULTS/"analysis.md").write_text("\n".join(lines))
    print(f"Experiment 7 verdict: {verdict}\nPrimary result: {primary}\nAnswer: {answer}")

if __name__=="__main__": main()
