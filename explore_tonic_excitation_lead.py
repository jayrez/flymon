"""EXPLORATORY (not preregistered): characterise the A1 ablation observation.

A1 = frozen config with tonic inhibition removed (tonic excitation r0_exc = 1 retained).
It was never a calibrated candidate (the D1 grid tied r0_exc = r0_inh) and it fails Gate 1
by construction, so it cannot be selected. This only checks whether the held-out effect is
also present on calibration data and whether it depends on geometry and ON polarity.
"""
import json, numpy as np
import run_tonic_disinhibition_experiment as R
import run_motion_nonlinearity_experiment as E19
from flymon import tonic_disinhibition as td
from flymon import column_motion as cm

sel = json.loads((R.RESULTS / "selected-config.json").read_text())
held = json.loads((R.RESULTS / "heldout-results.json").read_text())
const = sel["constants"]
frozen = td.TonicConfig.from_json(sel["config"])
a1 = frozen.replace(r0_inh=0.0)
ctx = E19.load_context()
out = dict(label="EXPLORATORY - discovered in a held-out ablation; not a preregistered candidate",
           config=a1.to_json())
for split in ("calibration", "heldout"):
    bank = E19.stimulus_bank(split)
    lum = E19.luminance_cache(bank, ctx["records"], ctx["colindex"], ctx["uv"])
    sim = R.Sim(ctx, bank, lum, const["sigma"], const["M"], const["chemistry"])
    m = R.direction_metrics(sim, a1)
    s = R.summarise(m)
    onoff = {}
    for sub in R.SUBTYPES:
        on = [np.mean(sim.scalar(a1, sub, n)) for n, (_q, mm) in bank.items() if mm.get("polarity") == "ON"]
        off = [np.mean(sim.scalar(a1, sub, n)) for n, (_q, mm) in bank.items() if mm.get("polarity") == "OFF"]
        onoff[sub] = dict(on=float(np.mean(on)), off=float(np.mean(off)))
    out[split] = dict(summary=s, subtypes={k: dict(dsi=v["dsi_mean"], signResp=v["sign_consistency_responsive"])
                                          for k, v in m.items()}, on_off=onoff)
    print(f"[{split}] T4 {s['t4_mean']:+.4f} ({s['t4_positive']}/4) all-8 {s['mean_dsi']:+.4f} "
          f"{ {k: round(v['dsi_mean'],3) for k, v in m.items()} }", flush=True)
    print(f"   T4 ON>OFF: {[onoff[x]['on'] > onoff[x]['off'] for x in R.T4]}  "
          f"T5 OFF>ON: {[onoff[x]['off'] > onoff[x]['on'] for x in R.SUBTYPES[4:]]}", flush=True)
    if split == "heldout":
        g = R.gate_metrics(sim, a1, speeds=(0.75, 1.5, 3.0))
        out["heldout_gates"] = g
        speed_bank = {n: v for n, v in bank.items() if v[1].get("axis") == "speed"}
        obs = R.summarise(R.direction_metrics(sim, a1, subtypes=R.T4, bank=speed_bank))["t4_mean"]
        rng = np.random.default_rng(21099)
        ncol = len(ctx["colindex"]); nulls = []
        for i in range(50):
            perm = {t: rng.permutation(ncol) for t in cm.ALL_PARTNERS}
            ps = {s_: {t: mm[:, perm[t]] for t, mm in mats.items()} for s_, mats in ctx["proj"].items()}
            nulls.append(R.summarise(R.direction_metrics(sim, a1, subtypes=R.T4, bank=speed_bank, proj=ps))["t4_mean"])
        nulls = np.array(nulls)
        out["geometry_null_50"] = dict(observed=float(obs), null_mean=float(nulls.mean()),
                                       null_sd=float(nulls.std()), residual_fraction=float(nulls.mean() / obs))
        flat = R.summarise(R.direction_metrics(sim, td.temporal_flat(a1), subtypes=R.T4))
        out["temporal_flat_t4"] = flat["t4_mean"]
        print(f"   gate1 {g['gate1']} (release fraction {g['release_fraction']:.2f}); geometry residual "
              f"{out['geometry_null_50']['residual_fraction']:.3f}; temporal-flat T4 {flat['t4_mean']:+.4f}", flush=True)
(R.RESULTS / "exploratory-a1-tonic-excitation.json").write_text(json.dumps(out, indent=2) + "\n")
print("saved exploratory-a1-tonic-excitation.json")
