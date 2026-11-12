"""The full monitoring sweep: fit a model once on data before a cutoff, then walk forward through
sequential real time windows measuring input drift (genre PSI) and output quality (NDCG@10,
Recall@10) independently -- on `compass-recsys`'s real MovieLens-1M dataset, no synthetic drift.
"""

from __future__ import annotations

import numpy as np

from sentinel.drift import genre_distribution, psi
from sentinel.quality import fit_frozen_model, load_dataset, quality_on_window, window_split


def run_sweep(quantiles: list[float], min_history: int = 5) -> list[dict]:
    ds = load_dataset()
    all_t = np.concatenate(ds.times)
    cuts = [int(np.quantile(all_t, q)) for q in quantiles]

    model = fit_frozen_model(ds, cuts[1])  # trained once, on everything before the first cutoff

    # reference genre distribution: the training window itself
    train_items = np.concatenate([s[t < cuts[1]] for s, t in zip(ds.seqs, ds.times)])
    ref_dist, _ = genre_distribution(train_items, ds.catalog.genres)

    rows = []
    for i in range(1, len(quantiles) - 1):
        lo_q, hi_q = quantiles[i], quantiles[i + 1]
        lo, hi = cuts[i], cuts[i + 1]

        window_items = np.concatenate([s[(t >= lo) & (t < hi)] for s, t in zip(ds.seqs, ds.times)])
        dist, n_interactions = genre_distribution(window_items, ds.catalog.genres)
        genre_psi = psi(ref_dist, dist)

        split = window_split(ds, lo_q, hi_q, min_history=min_history)
        q = quality_on_window(model, split)

        rows.append({
            "window": f"[{lo_q:.2f}, {hi_q:.2f})",
            "lo_q": lo_q, "hi_q": hi_q,
            "n_interactions": int(n_interactions),
            "genre_psi": genre_psi,
            **q,
        })  # fmt: skip
        print(f"[pipeline] window {rows[-1]['window']:>14} n_interactions={n_interactions:>7} "
              f"genre_psi={genre_psi:.4f} ndcg@10={q['ndcg@10']:.4f} n_eval={q['n_eval']}")
    return rows
