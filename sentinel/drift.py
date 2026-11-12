"""Population Stability Index (PSI) -- the standard input-drift statistic in production credit
and recommendation systems (a convention, not a universal law: readings below are reported as
measured numbers, not graded against a threshold this repo invented).

PSI(reference, current) = sum( (current_i - reference_i) * log(current_i / reference_i) )

over a shared set of bins/categories. It's 0 when the two distributions are identical and grows
as they diverge; unlike a KS-test it doesn't give a p-value, but it composes cleanly across many
categories (here: movie genres) into one number per time window, which is what a dashboard needs.
"""

from __future__ import annotations

import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, eps: float = 1e-6) -> float:
    """Both inputs are proportions over the same categories (sum to ~1, same length, same order)."""
    reference = np.clip(np.asarray(reference, dtype=np.float64), eps, None)
    current = np.clip(np.asarray(current, dtype=np.float64), eps, None)
    return float(np.sum((current - reference) * np.log(current / reference)))


def genre_distribution(items: np.ndarray, genre_matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """The proportion of genre tags among a set of (possibly repeated) item indices.

    A film can have more than one genre, so this sums to more than 1 across genres for a single
    film and is normalised by the *number of interactions*, not the number of genre tags --
    "what fraction of interactions this period touched each genre."
    """
    n = len(items)
    if n == 0:
        return np.zeros(genre_matrix.shape[1]), 0
    counts = genre_matrix[items].sum(axis=0)
    return counts / n, n
