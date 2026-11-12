"""Integration tests on `compass.data.synthetic()` -- no MovieLens download, so these run in CI.
They check the pipeline executes correctly and returns sane numbers; the actual PSI-spike finding
reported in the README is checked against the real dataset by `make eval`, not here.
"""

from __future__ import annotations

import compass.data as D
import numpy as np

from sentinel.drift import genre_distribution, psi
from sentinel.quality import fit_frozen_model, quality_on_window, window_split


def _ds():
    return D.synthetic(n_users=200, n_items=60, seed=0, min_len=10, max_len=30)


def test_fit_and_score_on_a_real_temporal_window():
    ds = _ds()
    all_t = np.concatenate(ds.times)
    cutoff = int(np.quantile(all_t, 0.5))
    model = fit_frozen_model(ds, cutoff)

    split = window_split(ds, 0.5, 1.0, min_history=3)
    assert len(split.eval_users) > 0, "the synthetic dataset should yield some evaluable users"

    q = quality_on_window(model, split)
    assert 0.0 <= q["ndcg@10"] <= 1.0
    assert 0.0 <= q["recall@10"] <= 1.0
    assert q["n_eval"] == len(split.eval_users)


def test_genre_drift_is_zero_when_window_equals_reference():
    ds = _ds()
    items = np.concatenate(ds.seqs)
    dist, n = genre_distribution(items, ds.catalog.genres)
    assert n == len(items)
    assert psi(dist, dist) == 0.0


def test_fitting_on_more_history_does_not_crash_on_cold_start_users():
    """A user with zero pre-cutoff history (cold start) must not break scoring."""
    ds = _ds()
    all_t = np.concatenate(ds.times)
    very_early_cutoff = int(all_t.min())  # ~everyone is cold-start at this cutoff
    model = fit_frozen_model(ds, very_early_cutoff)
    split = window_split(ds, 0.0, 1.0, min_history=0)
    q = quality_on_window(model, split)
    assert q["n_eval"] > 0
