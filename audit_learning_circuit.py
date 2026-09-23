"""Experiment 26 Phase 1-2: label-free audit of MaleCNS mushroom-body learning circuitry.

Resolves every population against the local Flymon MaleCNS v1.0 sources only:
  redfly-benchmark/data/brain.npz                              (FlyBrain neuron index, types)
  raw/body-annotations-male-cns-v1.0-minconf-0.5.feather       (type, class, superclass)
  raw/body-neurotransmitters-male-cns-v1.0.feather             (consensus / predicted NT)
  raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather     (raw synapse counts)
  redfly-benchmark/data/weights.npz                            (FlyBrain signed, normalised W)
No neuron IDs are copied from any other project. Writes results/experiment-26/circuit-audit.json
(machine-readable) and the proposed plastic edge set with a deterministic hash.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf
from scipy import sparse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "redfly-benchmark" / "data"
RAW = DATA / "raw"
OUT = ROOT / "results" / "experiment-26"
CONTROLLER_DN_TYPES = ("DNa02", "DNg100", "MDN", "DNp01")      # E14 frozen controller populations
POPS = {"KC": r"^KC", "MBON": r"^MBON", "PAM": r"^PAM\d", "PPL1": r"^PPL1\d", "PPL2": r"^PPL2\d",
        "DPM": r"^DPM$", "APL": r"^APL$", "OA": r"^OA-", "T4": r"^T4[abcd]$"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = np.load(DATA / "brain.npz")
    ids, ct, side, supc = m["ids"], m["cell_type"].astype(str), m["side"].astype(str), m["superclass"].astype(str)
    index = {int(b): i for i, b in enumerate(ids)}
    ann = {r["bodyId"]: r for r in pf.read_table(RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather", columns=[
        "bodyId", "type", "flywireType", "hemibrainType", "class", "subclass", "superclass", "instance"]).to_pylist()}
    nt = {r["body"]: r for r in pf.read_table(RAW / "body-neurotransmitters-male-cns-v1.0.feather", columns=[
        "body", "consensus_nt", "predicted_nt", "predicted_nt_confidence", "ground_truth",
        "celltype_predicted_nt", "celltype_predicted_nt_confidence"]).to_pylist()}
    pops = {k: np.array([i for i, t in enumerate(ct) if re.match(p, t)], np.int64) for k, p in POPS.items()}
    dn = np.array([i for i, t in enumerate(ct) if t in CONTROLLER_DN_TYPES], np.int64)

    # ---------------- population records ----------------
    records = {}
    for k in ("KC", "MBON", "PAM", "PPL1", "PPL2", "DPM", "APL", "OA"):
        rows = []
        for i in pops[k]:
            b = int(ids[i]); a = ann.get(b, {}); n = nt.get(b, {})
            rows.append(dict(flybrain_index=int(i), body_id=b, cell_type=ct[i], side=side[i],
                             annotation_type=a.get("type"), flywire_type=a.get("flywireType"),
                             hemibrain_type=a.get("hemibrainType"), cls=a.get("class"), superclass=a.get("superclass"),
                             consensus_nt=n.get("consensus_nt"), predicted_nt=n.get("predicted_nt"),
                             predicted_nt_confidence=n.get("predicted_nt_confidence"), ground_truth_nt=n.get("ground_truth"),
                             celltype_predicted_nt=n.get("celltype_predicted_nt")))
        records[k] = rows

    # ---------------- raw synapse counts among/around the MB populations ----------------
    mbset = np.concatenate([pops[k] for k in ("KC", "MBON", "PAM", "PPL1", "PPL2", "DPM", "APL")])
    mb_bodies = pa.array(ids[mbset].astype(np.int64))
    edges = pf.read_table(RAW / "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
                          columns=["body_pre", "body_post", "weight"], memory_map=True)
    keep = pc.or_(pc.is_in(edges["body_pre"], value_set=mb_bodies), pc.is_in(edges["body_post"], value_set=mb_bodies))
    sub = edges.filter(keep)
    pre = np.array([index.get(int(b), -1) for b in sub["body_pre"].to_numpy()])
    post = np.array([index.get(int(b), -1) for b in sub["body_post"].to_numpy()])
    w = sub["weight"].to_numpy().astype(np.int64)
    ok = (pre >= 0) & (post >= 0)
    pre, post, w = pre[ok], post[ok], w[ok]
    pop_of = np.full(len(ids), "", dtype=object)
    for k in ("KC", "MBON", "PAM", "PPL1", "PPL2", "DPM", "APL"):
        pop_of[pops[k]] = k
    ttype = ct

    def block(a, b):
        sel = (pop_of[pre] == a) & (pop_of[post] == b)
        return pre[sel], post[sel], w[sel]

    summary = {}
    for a in ("KC", "PAM", "PPL1", "PPL2", "DPM", "APL", "MBON"):
        for b in ("KC", "MBON", "PAM", "PPL1", "APL", "DPM"):
            p_, q_, w_ = block(a, b)
            if len(w_):
                summary[f"{a}->{b}"] = dict(edges=int(len(w_)), synapses=int(w_.sum()))
    # KC input composition by presynaptic superclass (is there visual drive?)
    kcin = pop_of[post] == "KC"
    comp = collections.Counter()
    for i, ww in zip(pre[kcin], w[kcin]):
        comp[supc[i] if pop_of[i] == "" else pop_of[i]] += int(ww)
    kc_by_type_visual = {}
    vis_sc = {"visual_projection", "visual_centrifugal", "ol_intrinsic"}
    for t in sorted(set(ct[pops["KC"]])):
        tsel = kcin & (ttype[post] == t)
        tot = int(w[tsel].sum()); vis = int(w[tsel & np.isin(supc[pre], list(vis_sc))].sum())
        kc_by_type_visual[t] = dict(n=int((ct[pops["KC"]] == t).sum()), input_synapses=tot,
                                    visual_projection_input_fraction=vis / tot if tot else 0.0)
    # DAN -> MBON and DAN -> KC by type (compartment proxy), KC -> MBON by MBON type
    dan_mbon = collections.defaultdict(int)
    for a in ("PAM", "PPL1"):
        p_, q_, w_ = block(a, "MBON")
        for i, j, ww in zip(p_, q_, w_):
            dan_mbon[(ct[i], ct[j])] += int(ww)
    dan_kc = collections.defaultdict(lambda: collections.Counter())
    for a in ("PAM", "PPL1"):
        p_, q_, w_ = block(a, "KC")
        for i, j, ww in zip(p_, q_, w_):
            dan_kc[ct[i]][ct[j]] += int(ww)
    kc_mbon = collections.defaultdict(lambda: collections.Counter())
    p_, q_, w_ = block("KC", "MBON")
    for i, j, ww in zip(p_, q_, w_):
        kc_mbon[ct[j]][ct[i]] += int(ww)

    # ---------------- FlyBrain reachability (hops in the simulated W) ----------------
    W = sparse.load_npz(DATA / "weights.npz").tocsr()          # (post, pre)
    Wt = (W != 0).T.tocsr()                                     # pre -> post adjacency

    def hops_from(start, maxh=6):
        dist = np.full(len(ids), -1, np.int16); frontier = np.unique(start); dist[frontier] = 0
        for h in range(1, maxh + 1):
            nxt = np.unique(Wt[frontier].indices)
            nxt = nxt[dist[nxt] < 0]
            if not len(nxt):
                break
            dist[nxt] = h; frontier = nxt
        return dist
    reach = {}
    for src in ("T4", "KC", "MBON"):
        d = hops_from(pops[src])
        reach[src] = {tgt: (int(d[idx][d[idx] >= 0].min()) if (d[idx] >= 0).any() else None,
                            float(np.mean(d[idx] >= 0)))
                      for tgt, idx in (("KC", pops["KC"]), ("MBON", pops["MBON"]), ("PAM", pops["PAM"]),
                                       ("PPL1", pops["PPL1"]), ("controller_DN", dn))}
    mbon_dn_direct = {}
    for j in pops["MBON"]:
        row = W[dn][:, [j]]
        if row.nnz:
            mbon_dn_direct[ct[j]] = mbon_dn_direct.get(ct[j], 0) + int(row.nnz)

    audit = dict(
        sources={p.name: sha256_file(p) for p in (DATA / "brain.npz", DATA / "weights.npz",
                                                  RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
                                                  RAW / "body-neurotransmitters-male-cns-v1.0.feather")},
        population_counts={k: dict(n=len(v), types=len({r["cell_type"] for r in v}),
                                   consensus_nt=dict(collections.Counter(r["consensus_nt"] for r in v)),
                                   cls=dict(collections.Counter(r["cls"] for r in v)),
                                   ground_truth_nt=dict(collections.Counter(r["ground_truth_nt"] for r in v)))
                           for k, v in records.items()},
        block_synapses=summary, kc_input_composition=dict(comp.most_common(15)), kc_types=kc_by_type_visual,
        dan_to_mbon={f"{a}->{b}": s for (a, b), s in sorted(dan_mbon.items(), key=lambda t: -t[1])[:80]},
        dan_to_kc={a: dict(c.most_common(6)) for a, c in dan_kc.items()},
        kc_to_mbon={b: dict(c.most_common(6)) | {"_total": int(sum(c.values()))} for b, c in kc_mbon.items()},
        reachability_hops_min_and_fraction=reach, mbon_to_controller_dn_direct_edges=mbon_dn_direct,
        flybrain_sign_rule="FlyBrain build: sign -1 iff consensus_nt matches gaba|glutamate|histamine, else +1 "
                           "(dopamine and octopamine are therefore simulated as excitatory)")
    (OUT / "circuit-audit.json").write_text(json.dumps(audit, indent=1, default=str) + "\n")
    import gzip
    with gzip.open(OUT / "population-records.json.gz", "wt") as fh:
        json.dump(records, fh, default=str)
    print(json.dumps({k: audit[k] for k in ("population_counts", "block_synapses", "kc_input_composition",
                                            "reachability_hops_min_and_fraction", "mbon_to_controller_dn_direct_edges")},
                     indent=1, default=str))
    print(json.dumps(audit["kc_types"], indent=1))
    print(json.dumps(dict(list(audit["dan_to_mbon"].items())[:40]), indent=1))
    print(json.dumps(audit["dan_to_kc"], indent=1))


if __name__ == "__main__":
    main()
