"""E26 diagnostic (no learning): firing regime of mushroom-body populations in stock FlyBrain.

Runs unmodified FlyBrain.step (no injection) from reset for 600 steps on the E26 seeds and
reports, per population, the fraction of steps each cell spikes (1.0 = the 50 Hz ceiling at
dt = 20 ms) and when saturation sets in. Also reports the whole-brain distribution and, for
reference, the E14 controller DNs. Writes results/experiment-26/mb-saturation-diagnostic.json.
"""
import json
from pathlib import Path

import numpy as np
from flybrain import FlyBrain

from run_visual_experiment import DATA

ROOT = Path(__file__).resolve().parent
SEEDS = tuple(range(2701, 2707))
POPS = ("KCg-d", "KCg-m", "KCab", "KCapbp-ap2", "MBON01", "MBON11", "PAM01", "PPL101", "APL", "DPM")
CONTROLLER_DN = ("DNa02", "DNg100", "MDN", "DNp01")

b = FlyBrain(data=DATA, device="cuda")
ct = b.cell_type.astype(str)
idx = {p: np.flatnonzero(ct == p) for p in POPS}
idx["controller_DN"] = np.flatnonzero(np.isin(ct, CONTROLLER_DN))
idx["all_KC"] = np.flatnonzero(np.char.startswith(ct, "KC"))
steps = 600
out = {"seeds": list(SEEDS), "steps": steps, "dt_s": float(b.dt), "max_rate_hz": 1.0 / float(b.dt), "populations": {}}
per = {p: [] for p in idx}; onset = {p: [] for p in idx}; whole = []
for s in SEEDS:
    b.reset(seed=s)
    counts = np.zeros(b.n, np.int32); frac_t = {p: [] for p in idx}
    for t in range(steps):
        f = b.step()
        counts[f] += 1
        m = np.zeros(b.n, bool); m[f] = True
        for p, ix in idx.items():
            frac_t[p].append(float(m[ix].mean()))
    for p, ix in idx.items():
        per[p].append(float((counts[ix] / steps).mean()))
        ft = np.array(frac_t[p]); onset[p].append(int(np.argmax(ft >= 0.9)) if (ft >= 0.9).any() else None)
    whole.append(counts / steps)
w = np.mean(whole, axis=0)
for p, ix in idx.items():
    out["populations"][p] = dict(n=int(len(ix)), spike_fraction_of_steps=float(np.mean(per[p])),
                                 rate_hz=float(np.mean(per[p]) / b.dt), first_step_90pct_active=onset[p])
out["whole_brain"] = dict(fraction_neurons_above_0p9_of_steps=float(np.mean(w >= 0.9)),
                          n_neurons_above_0p9=int(np.sum(w >= 0.9)),
                          fraction_silent=float(np.mean(w == 0)), median_rate_hz=float(np.median(w) / b.dt))
sat = w >= 0.9
out["saturated_by_superclass"] = {sc: int(n) for sc, n in zip(*np.unique(b.superclass[sat].astype(str), return_counts=True))}
Path(ROOT / "results/experiment-26/mb-saturation-diagnostic.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
