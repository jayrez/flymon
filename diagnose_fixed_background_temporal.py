"""EXPLORATORY (calibration split only; not preregistered): why does fixed-reference T4 DS
at the E21 parameters survive temporal flattening?

Decomposes calibration T4 mean DSI at E21 A1 parameters under R2 by input class and by
temporal flattening, and adds an instantaneous-membrane variant (c_mem -> tiny) to test
whether the membrane/conductance time course, rather than the Mi1/Mi9/Mi4 filter asymmetry,
supplies the temporal asymmetry. Never used for selection; held-out data are not touched.
"""
import json
from dataclasses import replace

import numpy as np

import run_fixed_background_experiment as R
from flymon import adapted_reference as ar
from flymon import tonic_disinhibition as td

ctx, const = R.contexts()
sim = R.make_sim(ctx, const, R.gray_bank("calibration", 0.5))
base = R.e21_params("fixed", ar.ADAPT_FRAMES)
_orig = td.TonicConfig


def fast_membrane(cfg_tonic):
    return replace(cfg_tonic, c_mem=1e-3)


rows = {}
variants = {
    "R2_E21_params": base,
    "R2_temporal_flat": ar.temporal_flat(base),
    "R2_excitation_only(Mi1)": base.replace(drop=("Mi9", "Mi4")),
    "R2_excitation_only_flat": ar.temporal_flat(base).replace(drop=("Mi9", "Mi4")),
    "R2_Mi1+Mi9": base.replace(drop=("Mi4",)),
    "R2_Mi1+Mi4": base.replace(drop=("Mi9",)),
    "R2_Mi1+Mi4_flat": ar.temporal_flat(base).replace(drop=("Mi9",)),
    "R2_no_filters(tau=0)": base.replace(tau_fast=0.0, tau_slow=0.0),
}
for name, c in variants.items():
    s, m = R.t4_dsi(sim, c)
    rows[name] = dict(t4_mean=s["t4_mean"], t4_positive=s["t4_positive"],
                      per_subtype={k: v["dsi_mean"] for k, v in m.items()})
    print(f"{name:<28} T4 {s['t4_mean']:+.4f} ({s['t4_positive']}/4)", flush=True)

# instantaneous membrane: patch the tonic config factory for this run only
ar_tonic = ar.AdaptedConfig.tonic
ar.AdaptedConfig.tonic = lambda self, r0_inh=0.0: fast_membrane(ar_tonic(self, r0_inh))
for name, c in (("R2_instant_membrane", base), ("R2_instant_membrane_flat", ar.temporal_flat(base)),
                ("R2_instant_membrane_no_filters", base.replace(tau_fast=0.0, tau_slow=0.0))):
    sim._sig.clear()
    s, m = R.t4_dsi(sim, c)
    rows[name] = dict(t4_mean=s["t4_mean"], t4_positive=s["t4_positive"],
                      per_subtype={k: v["dsi_mean"] for k, v in m.items()})
    print(f"{name:<28} T4 {s['t4_mean']:+.4f} ({s['t4_positive']}/4)", flush=True)
ar.AdaptedConfig.tonic = ar_tonic

(R.RESULTS / "exploratory-temporal-diagnostic.json").write_text(json.dumps(dict(
    label="EXPLORATORY - calibration split only; not preregistered; not used for selection",
    rows=rows), indent=2) + "\n")
