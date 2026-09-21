"""Label-free biological connectivity tracing on the MaleCNS v1.0 graph.

Experiment 15 uses this module to rank neurons by how much of their synaptic
input can be traced back to the photoreceptors that actually carry the Pokemon
image (the Experiment-5 mapped R1-R6 set), using only the released connectome.
No stimulus, no neural activity, and no class label enters any function here, so
any neuron subset chosen from these scores is anatomically selected and cannot
leak test labels into the downstream classifier.

Graph convention (matches flybrain.FlyBrain): the weight matrix W is indexed
W[post, pre]; synaptic current flows pre -> post. `A = |W|` is the nonnegative
magnitude graph. `A @ v` therefore spreads a value forward, from presynaptic to
postsynaptic neurons (downstream).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


def load_graph(data_dir: Path):
    """Return (W, meta) where W[post, pre] is the signed CSR weight matrix."""
    meta = np.load(Path(data_dir) / "brain.npz")
    W = sparse.load_npz(Path(data_dir) / "weights.npz").tocsr()
    return W, meta


def row_normalized_magnitude(W: sparse.csr_matrix) -> sparse.csr_matrix:
    """|W| with each postsynaptic row scaled to sum to 1 over its inputs.

    (A_hat @ v)[i] is the input-weighted mean of v over the presynaptic partners
    of neuron i, i.e. the fraction of i's input drawn from a source carrying v."""
    A = W.copy()
    A.data = np.abs(A.data)
    indeg = np.asarray(A.sum(axis=1)).ravel()
    scale = np.zeros_like(indeg)
    nz = indeg > 0
    scale[nz] = 1.0 / indeg[nz]
    return sparse.diags(scale.astype(np.float32)) @ A


def visual_influence(W: sparse.csr_matrix, source: np.ndarray, alpha: float = 0.85,
                     iterations: int = 40, tol: float = 1e-7) -> np.ndarray:
    """Restart diffusion of visual drive downstream through the connectome.

    r = (1 - alpha) * s + alpha * (A_hat @ r), where s is the unit-mass restart
    on `source` neurons. r[i] in [0, 1] is the anatomical fraction of neuron i's
    recurrent input that traces back to the visual source; source neurons stay
    near 1 and the score decays with directed distance and input dilution."""
    Ahat = row_normalized_magnitude(W)
    s = np.zeros(W.shape[0], np.float64)
    s[source] = 1.0
    r = s.copy()
    for _ in range(iterations):
        r_next = (1 - alpha) * s + alpha * (Ahat @ r)
        if np.max(np.abs(r_next - r)) < tol:
            r = r_next
            break
        r = r_next
    return r


def hop_distance(W: sparse.csr_matrix, source: np.ndarray, max_hops: int = 12) -> np.ndarray:
    """Shortest directed hop count from `source` to every neuron (downstream).

    Uses boolean frontier expansion on the pre->post graph. Unreached neurons get
    the sentinel `max_hops + 1`."""
    A = W.copy()
    A.data = np.ones_like(A.data)  # presence only
    n = W.shape[0]
    dist = np.full(n, max_hops + 1, np.int16)
    frontier = np.zeros(n, bool)
    frontier[source] = True
    dist[source] = 0
    reached = frontier.copy()
    for hop in range(1, max_hops + 1):
        nxt = (A @ frontier.astype(np.float32)) > 0  # downstream neighbours
        nxt &= ~reached
        if not nxt.any():
            break
        dist[nxt] = hop
        reached |= nxt
        frontier = nxt
    return dist


def input_contributor_count(W: sparse.csr_matrix, source_mask: np.ndarray,
                            targets: np.ndarray) -> np.ndarray:
    """For each target, count distinct presynaptic partners that are in `source_mask`.

    A direct-connection anatomical support measure (one hop)."""
    src = sparse.csr_matrix((np.ones(source_mask.sum(), np.float32),
                             (np.flatnonzero(source_mask), np.zeros(int(source_mask.sum()), int))),
                            shape=(W.shape[0], 1))
    present = W.copy()
    present.data = (present.data != 0).astype(np.float32)
    counts = np.asarray((present @ src).todense()).ravel()
    return counts[targets]


@dataclass
class PathwayAudit:
    influence: np.ndarray          # visual-influence score per neuron
    influence_from_t5: np.ndarray  # visual-influence score seeded on T5
    hops_from_visual: np.ndarray   # directed hop distance from mapped R1-R6
    hops_from_t5: np.ndarray       # directed hop distance from T5


def audit(W: sparse.csr_matrix, source_r16: np.ndarray, t5: np.ndarray,
          alpha: float = 0.85) -> PathwayAudit:
    return PathwayAudit(
        influence=visual_influence(W, source_r16, alpha=alpha),
        influence_from_t5=visual_influence(W, t5, alpha=alpha),
        hops_from_visual=hop_distance(W, source_r16),
        hops_from_t5=hop_distance(W, t5),
    )
