"""The output-quality side of monitoring: a frozen model's real ranking quality on a real
temporal window, computed with `compass-recsys`'s own evaluation code (not reimplemented).
"""

from __future__ import annotations

from pathlib import Path

import compass.data as D
import numpy as np
from compass.baselines import EASE
from compass.data import Dataset, Split
from compass.metrics import evaluate

COMPASS_DATA_DIR = Path(__file__).resolve().parents[2] / "compass-recsys" / "data"


def load_dataset(root: Path = COMPASS_DATA_DIR) -> Dataset:
    return D.load(root=root)


def training_split(ds: Dataset, cutoff: int) -> Split:
    """Everything before `cutoff`, as a Split with no eval targets -- for fitting only."""
    hist, ht, hr = [], [], []
    for s, t, r in zip(ds.seqs, ds.times, ds.ratings):
        before = t < cutoff
        hist.append(s[before])
        ht.append(t[before])
        hr.append(r[before])
    return Split(
        name="train",
        history=hist,
        history_times=ht,
        history_ratings=hr,
        target=np.full(ds.n_users, -1, dtype=np.int64),
        target_time=np.full(ds.n_users, -1, dtype=np.int64),
        n_items=ds.n_items,
    )


def scorer(model, split: Split):
    """Exactly compass.train's own `scorer` -- a model's `.score` takes histories, not user ids,
    and `metrics.evaluate` only passes user ids, so this closes over the split to bridge them.
    """
    return lambda u: model.score([split.history[i] for i in u], u)


def fit_frozen_model(ds: Dataset, cutoff: int) -> EASE:
    return EASE().fit(training_split(ds, cutoff))


def window_split(ds: Dataset, lo_q: float, hi_q: float, min_history: int = 5) -> Split:
    """A temporal window as a `Split`: history = everything before `lo_q`, target = each user's
    first interaction in [lo_q, hi_q). Thin wrapper over `compass.data.temporal`, which already
    implements exactly this -- used here at arbitrary adjacent quantiles instead of just val/test.
    """
    return D.temporal(ds, val_q=lo_q, test_q=hi_q, min_history=min_history)[0]


def quality_on_window(model: EASE, split: Split, rec_k: int = 10) -> dict:
    result = evaluate(model.name if hasattr(model, "name") else "model", scorer(model, split), split, rec_k=rec_k)
    return {
        "n_eval": int(len(split.eval_users)),
        "ndcg@10": float(result.per_user("ndcg", rec_k).mean()),
        "recall@10": float(result.per_user("recall", rec_k).mean()),
    }
