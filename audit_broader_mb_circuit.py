"""Experiment 29 static audit: every Kenyon-cell class, visual-projection input, KC->MBON connectivity,
static MBON leverage and DAN compartment topology (connectome only; no simulation).

Sources are the same local MaleCNS v1.0 files as the E26 audit (audit_learning_circuit.py):
  redfly-benchmark/data/brain.npz, weights.npz (FlyBrain signed, input-normalised W; rows = post)
  raw/body-annotations-, body-neurotransmitters-, connectome-weights-male-cns-v1.0 feathers
Writes (results/experiment-29-broader-mb-circuit-audit/):
  visual-projection-set.json, kc-population-audit.json, kc-mbon-connectivity.json,
  static-leverage-screen.json, dan-compartment-audit.json
Rules are fixed in preregistration.md before any E29 dynamic result.
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
OUT = ROOT / "results" / "experiment-29-broader-mb-circuit-audit"
KC_RE, MBON_RE = r"^KC", r"^MBON"
DAN_RE = r"^(PAM\d|PPL1\d|PPL2\d)"
VP_SUPERCLASS = "visual_projection"      # same rule as the E15 pathway audit (run_pathway_audit.py)
SUBSTANTIAL_VISUAL_FRACTION = 0.01       # >= 1 % of class input |W| from the visual-projection set
DAN_MIN_SYNAPSES = 50                    # topology match: DAN->MBON and DAN->KC-class each >= 50 raw synapses


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def jdump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float) + "\n")


class Connectome:
    def __init__(self):
        m = np.load(DATA / "brain.npz")
        self.ids = m["ids"].astype(np.int64); self.ct = m["cell_type"].astype(str)
        self.side = m["side"].astype(str); self.sc = m["superclass"].astype(str)
        self.W = sparse.load_npz(DATA / "weights.npz").tocsr()          # (post, pre)
        self.Wabs = abs(self.W).tocsr()
        self.index = {int(b): i for i, b in enumerate(self.ids)}
        self.kc = np.flatnonzero([bool(re.match(KC_RE, t)) for t in self.ct])
        self.mbon = np.flatnonzero([bool(re.match(MBON_RE, t)) for t in self.ct])
        self.dan = np.flatnonzero([bool(re.match(DAN_RE, t)) for t in self.ct])
        self.vp = np.flatnonzero(self.sc == VP_SUPERCLASS)
        self.kc_types = sorted(set(self.ct[self.kc])); self.mbon_types = sorted(set(self.ct[self.mbon]))
        self.dan_types = sorted(set(self.ct[self.dan]))
        nt = pf.read_table(RAW / "body-neurotransmitters-male-cns-v1.0.feather",
                           columns=["body", "consensus_nt", "ground_truth"]).to_pylist()
        self.nt = {r["body"]: r for r in nt}
        ann = pf.read_table(RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
                            columns=["bodyId", "type", "hemibrainType", "flywireType", "class"]).to_pylist()
        self.ann = {r["bodyId"]: r for r in ann}

    def of(self, t):
        return np.flatnonzero(self.ct == t)

    def raw_edges(self, bodies_post=None, bodies_pre=None):
        """Raw synapse counts (FlyBrain-indexed) with post in / pre in the given body sets."""
        tab = pf.read_table(RAW / "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
                            columns=["body_pre", "body_post", "weight"], memory_map=True)
        keep = None
        for col, s in (("body_post", bodies_post), ("body_pre", bodies_pre)):
            if s is not None:
                k = pc.is_in(tab[col], value_set=pa.array(np.asarray(s, np.int64)))
                keep = k if keep is None else pc.or_(keep, k)
        sub = tab.filter(keep)
        pre = np.array([self.index.get(int(b), -1) for b in sub["body_pre"].to_numpy()])
        post = np.array([self.index.get(int(b), -1) for b in sub["body_post"].to_numpy()])
        w = sub["weight"].to_numpy().astype(np.int64)
        ok = (pre >= 0) & (post >= 0)
        return pre[ok], post[ok], w[ok]


def visual_projection_set(C):
    from flymon.pathway import hop_distance
    import run_generation_zero as G
    t4 = G.sensory_setup("native").brain_index
    hops = hop_distance(C.W, np.unique(np.asarray(t4, np.int64)), max_hops=8)
    rows = [dict(flybrain_index=int(i), body_id=int(C.ids[i]), cell_type=C.ct[i], side=C.side[i],
                 hops_from_frozen_t4=int(hops[i]) if hops[i] <= 8 else None) for i in C.vp]
    ids = sorted(int(C.ids[i]) for i in C.vp)
    h = [r["hops_from_frozen_t4"] for r in rows]
    return dict(rule=f"every FlyBrain neuron with superclass == '{VP_SUPERCLASS}' (MaleCNS v1.0 annotation); "
                     "identical to the E15 pathway-audit rule (run_pathway_audit.py); no graph-depth filter; "
                     "hops from the frozen T4 injection set are recorded descriptively only",
                source_categories=[VP_SUPERCLASS], excluded_categories=["ol_intrinsic", "visual_centrifugal",
                                                                         "visual_projection_tbc"],
                n=len(rows), types=len({r["cell_type"] for r in rows}),
                hemisphere=dict(collections.Counter(r["side"] for r in rows)),
                t4_hop_histogram={str(k): v for k, v in sorted(collections.Counter(h).items(), key=lambda t: (t[0] is None, t[0] or 0))},
                frozen_t4_cells=int(len(np.unique(t4))), body_ids=ids, sha256=sha(ids), cells=rows)


def kc_population_audit(C, vp):
    vpmask = np.zeros(len(C.ids), bool); vpmask[vp] = True
    raw_pre, raw_post, raw_w = C.raw_edges(bodies_post=C.ids[C.kc], bodies_pre=C.ids[C.kc])
    types = {}
    all_ids = []
    for t in C.kc_types:
        ix = C.of(t); all_ids += [int(b) for b in C.ids[ix]]
        rows_in = C.Wabs[ix]                                   # inputs to the class
        cols_out = C.Wabs[:, ix]
        vin = rows_in[:, vp]
        tot_cell = np.asarray(rows_in.sum(1)).ravel(); vis_cell = np.asarray(vin.sum(1)).ravel()
        frac_cell = np.divide(vis_cell, tot_cell, out=np.zeros_like(vis_cell), where=tot_cell > 0)
        vin_coo = vin.tocoo()
        into = np.isin(raw_post, ix); outof = np.isin(raw_pre, ix)
        vis_raw = int(raw_w[into & vpmask[raw_pre]].sum()); tot_raw = int(raw_w[into].sum())
        nts = collections.Counter((C.nt.get(int(b)) or {}).get("consensus_nt") for b in C.ids[ix])
        frac = float(vis_cell.sum() / tot_cell.sum()) if tot_cell.sum() else 0.0
        types[t] = dict(
            n=int(len(ix)), left=int((C.side[ix] == "L").sum()), right=int((C.side[ix] == "R").sum()),
            other_side=int((~np.isin(C.side[ix], ["L", "R"])).sum()),
            consensus_nt=dict(collections.Counter({str(k): v for k, v in nts.items()})),
            incoming_edges=int(rows_in.nnz), incoming_abs_weight=float(tot_cell.sum()),
            outgoing_edges=int(cols_out.nnz), outgoing_abs_weight=float(cols_out.sum()),
            incoming_raw_synapses=tot_raw, outgoing_raw_synapses=int(raw_w[outof].sum()),
            static_visual_input=dict(
                direct_vp_edges=int(vin.nnz), distinct_vp_presynaptic_cells=int(len(np.unique(vin_coo.col))),
                vp_abs_weight=float(vis_cell.sum()), total_abs_weight=float(tot_cell.sum()),
                vp_fraction_of_input=frac, vp_raw_synapses=vis_raw,
                vp_raw_fraction=float(vis_raw / tot_raw) if tot_raw else 0.0,
                vp_abs_weight_per_cell=float(vis_cell.mean()),
                per_cell_fraction=dict(median=float(np.median(frac_cell)), q75=float(np.quantile(frac_cell, .75)),
                                       max=float(frac_cell.max()), cells_with_any_vp_input=int((vis_cell > 0).sum())),
                category=("none" if vin.nnz == 0 else "substantial" if frac >= SUBSTANTIAL_VISUAL_FRACTION else "sparse")),
            body_ids=sorted(int(b) for b in C.ids[ix]))
        # top VP presynaptic types by |W| into this class
        agg = collections.Counter()
        for c, v in zip(vin_coo.col, vin_coo.data):
            agg[C.ct[vp[c]]] += float(v)
        types[t]["static_visual_input"]["top_vp_types"] = dict(agg.most_common(8))
    total = int(len(C.kc))
    assert sum(v["n"] for v in types.values()) == total
    return dict(selector=KC_RE, total_kc=total, n_types=len(types), sum_of_type_counts=sum(v["n"] for v in types.values()),
                population_sha256=sha(sorted(all_ids)), visual_category_rule=
                f"none: no direct VP edge; substantial: VP |W| fraction >= {SUBSTANTIAL_VISUAL_FRACTION}; else sparse",
                types=types)


def kc_mbon_connectivity(C, raw):
    pre, post, w = raw
    rawmap = collections.defaultdict(int)
    for a, b, x in zip(pre, post, w):
        rawmap[(a, b)] += int(x)
    out = {}
    for k in C.kc_types:
        ki = C.of(k)
        for m in C.mbon_types:
            mi = C.of(m)
            sub = C.W[mi][:, ki].tocoo()
            if sub.nnz == 0:
                continue
            vals = sub.data; per_m = np.zeros(len(mi)); np.add.at(per_m, sub.row, np.abs(vals))
            hemi = collections.Counter(f"{C.side[ki[c]]}{C.side[mi[r]]}" for r, c in zip(sub.row, sub.col))
            raw_syn = sum(rawmap.get((int(ki[c]), int(mi[r])), 0) for r, c in zip(sub.row, sub.col))
            out[f"{k}->{m}"] = dict(kc_type=k, mbon_type=m, kc_cells=int(len(ki)), mbon_cells=int(len(mi)),
                                    edges=int(sub.nnz), source_kc_cells=int(len(np.unique(sub.col))),
                                    target_mbon_cells=int(len(np.unique(sub.row))),
                                    weight_sum=float(vals.sum()), abs_weight_sum=float(np.abs(vals).sum()),
                                    median_edge_weight=float(np.median(vals)), max_edge_weight=float(vals.max()),
                                    per_target_mbon_cell_abs=per_m.tolist(), hemisphere_pattern=dict(hemi),
                                    raw_synapses=int(raw_syn))
    return out


def static_leverage(C, conn):
    out = {}
    tot = {m: np.asarray(C.Wabs[C.of(m)].sum(1)).ravel() for m in C.mbon_types}
    for key, r in conn.items():
        num = np.asarray(r["per_target_mbon_cell_abs"]); den = tot[r["mbon_type"]]
        frac = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        out[key] = dict(kc_type=r["kc_type"], mbon_type=r["mbon_type"], pooled=float(num.sum() / den.sum()),
                        per_cell=frac.tolist(), median=float(np.median(frac)), max=float(frac.max()),
                        min=float(frac.min()))
    return out


def dan_audit(C, raw_all):
    pre, post, w = raw_all
    syn = collections.defaultdict(int)
    for a, b, x in zip(C.ct[pre], C.ct[post], w):
        syn[(a, b)] += int(x)
    dans = {}
    for d in C.dan_types:
        ix = C.of(d)
        b0 = int(C.ids[ix[0]]); a = C.ann.get(b0, {})
        dans[d] = dict(n=int(len(ix)), left=int((C.side[ix] == "L").sum()), right=int((C.side[ix] == "R").sum()),
                       consensus_nt=dict(collections.Counter(str((C.nt.get(int(b)) or {}).get("consensus_nt")) for b in C.ids[ix])),
                       ground_truth_nt=dict(collections.Counter(str((C.nt.get(int(b)) or {}).get("ground_truth")) for b in C.ids[ix])),
                       metadata_hemibrain_type=a.get("hemibrainType"), metadata_flywire_type=a.get("flywireType"))
    mbons = {}
    for m in C.mbon_types:
        ix = C.of(m); a = C.ann.get(int(C.ids[ix[0]]), {})
        mbons[m] = dict(n=int(len(ix)), metadata_hemibrain_type=a.get("hemibrainType"), metadata_flywire_type=a.get("flywireType"),
                        consensus_nt=dict(collections.Counter(str((C.nt.get(int(b)) or {}).get("consensus_nt")) for b in C.ids[ix])))
    pairs = {}
    for k in C.kc_types:
        for m in C.mbon_types:
            rows = []
            for d in C.dan_types:
                dm, dk = syn.get((d, m), 0), syn.get((d, k), 0)
                if dm == 0 and dk == 0:
                    continue
                rows.append(dict(dan_type=d, dan_to_mbon=dm, dan_to_kc=dk, kc_to_dan=syn.get((k, d), 0),
                                 mbon_to_dan=syn.get((m, d), 0),
                                 topology_match=bool(dm >= DAN_MIN_SYNAPSES and dk >= DAN_MIN_SYNAPSES)))
            rows.sort(key=lambda r: (-r["topology_match"], -r["dan_to_mbon"], -r["dan_to_kc"]))
            matched = [r["dan_type"] for r in rows if r["topology_match"]]
            pairs[f"{k}->{m}"] = dict(dan_rows=rows[:8], topology_matched_dans=matched,
                                      primary_dan=matched[0] if matched else None)
    return dict(rule=f"topology-derived match: DAN type with >= {DAN_MIN_SYNAPSES} raw synapses onto the MBON type AND "
                     f">= {DAN_MIN_SYNAPSES} onto the KC class (DAN innervates both halves of the compartment); primary = "
                     "most DAN->MBON synapses among matches. Metadata (hemibrain/flyWire names) is reported separately "
                     "and never used for matching.",
                dan_populations=dans, mbon_populations=mbons, pairs=pairs)


def main():
    C = Connectome()
    vp = visual_projection_set(C)
    jdump("visual-projection-set.json", vp)
    kca = kc_population_audit(C, C.vp)
    jdump("kc-population-audit.json", kca)
    mb_bodies = np.concatenate([C.ids[C.kc], C.ids[C.mbon], C.ids[C.dan]])
    raw = C.raw_edges(bodies_post=mb_bodies)
    rawkm = tuple(x for x in raw)
    conn = kc_mbon_connectivity(C, rawkm)
    jdump("kc-mbon-connectivity.json", dict(weights="FlyBrain W (signed, input-normalised); raw_synapses from the "
                                            "minconf-0.5 feather", kc_types=C.kc_types, mbon_types=C.mbon_types,
                                            n_pairs_with_edges=len(conn), matrix_sha256=sha({k: (v["edges"], round(v["abs_weight_sum"], 12))
                                                                                             for k, v in conn.items()}),
                                            pairs=conn))
    lev = static_leverage(C, conn)
    jdump("static-leverage-screen.json", dict(metric="candidate |W| from KC class / total |W| into the MBON (per cell and "
                                              "pooled over the MBON type); static screening only", pairs=lev))
    dan = dan_audit(C, raw)
    jdump("dan-compartment-audit.json", dan)
    print("KC", kca["total_kc"], "types", kca["n_types"], "VP", vp["n"], vp["sha256"][:16])
    for t, v in kca["types"].items():
        s = v["static_visual_input"]
        print(f"{t:<12} n={v['n']:<5} vp_frac={s['vp_fraction_of_input']:.4f} raw={s['vp_raw_fraction']:.4f} {s['category']}")


if __name__ == "__main__":
    main()
