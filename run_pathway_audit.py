"""Experiment 15 Stage 2: label-free biological connectivity audit.

Ranks descending neurons and visual-projection cells by how strongly their
synaptic input traces back to the Experiment-5 mapped R1-R6 photoreceptors,
using only the released MaleCNS connectome. Produces the anatomically selected
DN subset used (frozen) by the downstream held-out classifier. No stimulus,
neural activity, or class label is read here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flymon.pathway import audit, input_contributor_count, load_graph
from flymon.retina import load_mapping

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "redfly-benchmark" / "data"
EXP5 = ROOT / "results" / "experiment-05-retinotopic"
RESULTS = ROOT / "results" / "experiment-15-biological-pathway"
ALPHA = 0.85
TOP_K = (10, 20, 50)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    W, meta = load_graph(DATA)
    ct = meta["cell_type"].astype(str)
    sc = meta["superclass"].astype(str)
    side = meta["side"].astype(str)
    ids = meta["ids"]

    records, diagnostics = load_mapping(EXP5 / "retina-mapping.json")
    source_r16 = np.array(sorted({r.brain_index for r in records}), int)
    t5 = np.flatnonzero(np.char.startswith(ct, "T5"))
    t4 = np.flatnonzero(np.char.startswith(ct, "T4"))
    vp = np.flatnonzero(sc == "visual_projection")
    dn = np.flatnonzero(sc == "descending_neuron")
    print(f"source R1-6 mapped: {len(source_r16)}  T5: {len(t5)}  VP: {len(vp)}  DN: {len(dn)}", flush=True)

    a = audit(W, source_r16, t5, alpha=ALPHA)

    source_mask = np.zeros(W.shape[0], bool); source_mask[source_r16] = True
    vp_mask = np.zeros(W.shape[0], bool); vp_mask[vp] = True
    t5_mask = np.zeros(W.shape[0], bool); t5_mask[t5] = True

    dn_direct_vis = input_contributor_count(W, source_mask, dn)
    dn_vp_inputs = input_contributor_count(W, vp_mask, dn)
    dn_t5_inputs = input_contributor_count(W, t5_mask, dn)
    vp_t5_inputs = input_contributor_count(W, t5_mask, vp)

    def neuron_row(i):
        return dict(brain_index=int(i), flywire_malecns_id=int(ids[i]), cell_type=str(ct[i]),
                    side=str(side[i]), influence=float(a.influence[i]),
                    influence_from_t5=float(a.influence_from_t5[i]),
                    hops_from_visual=int(a.hops_from_visual[i]),
                    hops_from_t5=int(a.hops_from_t5[i]))

    dn_pos = {int(x): k for k, x in enumerate(dn)}
    dn_order = dn[np.argsort(-a.influence[dn], kind="stable")]
    dn_ranked = []
    for rank, i in enumerate(dn_order):
        k = dn_pos[int(i)]
        row = neuron_row(int(i))
        row.update(rank=rank, direct_visual_contributors=int(dn_direct_vis[k]),
                   visual_projection_inputs=int(dn_vp_inputs[k]), t5_inputs=int(dn_t5_inputs[k]))
        dn_ranked.append(row)

    vp_order = vp[np.argsort(-a.influence[vp], kind="stable")]
    vp_pos = {int(x): k for k, x in enumerate(vp)}
    vp_ranked = []
    for rank, i in enumerate(vp_order):
        row = neuron_row(i); row.update(rank=rank, t5_inputs=int(vp_t5_inputs[vp_pos[int(i)]]))
        vp_ranked.append(row)

    # Anatomically selected subsets (frozen before any classification result).
    selected = {f"bio_dn_top{k}": [int(ids[i]) for i in dn_order[:k]] for k in TOP_K}
    selected_idx = {f"bio_dn_top{k}": dn_order[:k].tolist() for k in TOP_K}
    vp_selected = {f"bio_vp_top{k}": [int(ids[i]) for i in vp_order[:k]] for k in (50, 100, 250)}

    # Named controller / reference neurons.
    named = {}
    for name in ("DNa02", "DNg100", "DNg13", "DNp01"):
        idx = np.flatnonzero(ct == name)
        named[name] = [dict(neuron_row(int(i)),
                            rank_among_dn=int(np.flatnonzero(dn_order == i)[0]),
                            direct_visual_contributors=int(dn_direct_vis[dn_pos[int(i)]]),
                            visual_projection_inputs=int(dn_vp_inputs[dn_pos[int(i)]]),
                            t5_inputs=int(dn_t5_inputs[dn_pos[int(i)]])) for i in idx]

    p20 = json.loads((ROOT / "results/experiment-13-population-event/population.json").read_text())
    p20_rows = []
    for nr in p20["neurons"]:
        i = int(nr["brain_index"])
        p20_rows.append(dict(neuron_row(i), historical_rank=nr["historical_rank"],
                             rank_among_dn=int(np.flatnonzero(dn_order == i)[0]),
                             direct_visual_contributors=int(dn_direct_vis[dn_pos[i]]),
                             visual_projection_inputs=int(dn_vp_inputs[dn_pos[i]]),
                             t5_inputs=int(dn_t5_inputs[dn_pos[i]])))

    out = dict(
        method="restart diffusion of |W| row-normalised input, alpha=%.2f; label-free" % ALPHA,
        alpha=ALPHA, source_r16_count=len(source_r16), t5_count=len(t5), t4_count=len(t4),
        vp_count=len(vp), dn_count=len(dn),
        retina_mapping_sha=diagnostics.get("source_sha256"),
        dn_influence_summary=dict(max=float(a.influence[dn].max()), median=float(np.median(a.influence[dn])),
                                  min=float(a.influence[dn].min())),
        dn_ranked=dn_ranked, vp_ranked=vp_ranked[:250],
        selected_dn=selected, selected_dn_indices=selected_idx, selected_vp=vp_selected,
        named_neurons=named, p20=p20_rows, top_k=list(TOP_K))
    (RESULTS / "connectivity-audit.json").write_text(json.dumps(out, indent=2) + "\n")

    print("\n=== Top 20 DNs by visual influence ===", flush=True)
    for row in dn_ranked[:20]:
        print(f"  #{row['rank']:>3} {row['cell_type']:<9} {row['side']} "
              f"infl={row['influence']:.4f} hops={row['hops_from_visual']} "
              f"visContrib={row['direct_visual_contributors']} vpIn={row['visual_projection_inputs']} "
              f"t5In={row['t5_inputs']}", flush=True)
    print("\n=== Named controller/reference DNs ===", flush=True)
    for name, rows in named.items():
        for row in rows:
            print(f"  {name:<7} {row['side']} rank={row['rank_among_dn']:>4}/{len(dn)} "
                  f"infl={row['influence']:.4f} hops={row['hops_from_visual']} "
                  f"visContrib={row['direct_visual_contributors']} vpIn={row['visual_projection_inputs']} "
                  f"t5In={row['t5_inputs']}", flush=True)
    print("\nsaved connectivity-audit.json", flush=True)


if __name__ == "__main__":
    main()
