"""Family-wise permutation null for Experiment 4 temporal DN-subset search."""
from __future__ import annotations

import json
import time

import numpy as np

from analyze_temporal_dn import RESULTS, NAMES, SEEDS, representation
from validate_dn_subpopulation import shuffled_population_accuracies


def main():
    started = time.perf_counter()
    cpath = RESULTS / "classification.json"
    ppath = RESULTS / "permutation-results.json"
    c = json.loads(cpath.read_text())
    p = json.loads(ppath.read_text())
    archive = np.load(RESULTS / "dn-bins-5.npz")
    conditions = list(archive["conditions"])
    raw = archive["counts"][[conditions.index(name) for name in NAMES]].transpose(1, 0, 2, 3)
    reps = {steps: representation(raw, steps) for steps in (5, 10, 20, 50)}
    labels = np.tile(np.arange(5), (len(SEEDS), 1))
    rng = np.random.default_rng(20260922)
    null = []
    for iteration in range(500):
        shuffled = np.stack([rng.permutation(row) for row in labels])
        null.append(float(max(shuffled_population_accuracies(data, shuffled).max()
                              for data in reps.values())))
        if (iteration + 1) % 100 == 0:
            print("Temporal subset null", iteration + 1, "of 500", flush=True)
    null = np.array(null)
    observed = c["nested_population"]["accuracy"]
    p["temporal_selected_population"] = dict(observed=observed,
        null_mean=float(null.mean()), null_95th=float(np.quantile(null, .95)),
        empirical_p=float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1)),
        permutations=500, null_accuracies=null.tolist(),
        shuffle="labels permuted independently within seed; DN F rankings refit from training seeds; maximum grouped accuracy over all 4 temporal bin sizes × 5 population sizes per shuffle (conservative family-wise null)")
    c["performance"]["temporal_population_validation_ms"] = (time.perf_counter() - started) * 1000
    cpath.write_text(json.dumps(c, indent=2) + "\n")
    ppath.write_text(json.dumps(p, indent=2) + "\n")
    print("Nested temporal selected-DN accuracy", observed,
          "max-search permutation p", p["temporal_selected_population"]["empirical_p"], flush=True)


if __name__ == "__main__":
    main()
