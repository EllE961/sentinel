import numpy as np

from sentinel.drift import genre_distribution, psi


def test_psi_is_zero_for_identical_distributions():
    d = np.array([0.3, 0.5, 0.2])
    assert psi(d, d) == 0.0


def test_psi_is_nonnegative():
    """PSI = KL(cur||ref) + KL(ref||cur), the symmetrised KL divergence (Jeffreys divergence) --
    a sum of two non-negative KL terms, so it can never go negative for valid distributions.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.dirichlet(np.ones(5))
        b = rng.dirichlet(np.ones(5))
        assert psi(a, b) >= -1e-9


def test_psi_is_symmetric():
    rng = np.random.default_rng(1)
    a, b = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4))
    assert abs(psi(a, b) - psi(b, a)) < 1e-9


def test_psi_grows_with_distance_from_reference():
    ref = np.array([0.25, 0.25, 0.25, 0.25])
    near = np.array([0.22, 0.28, 0.24, 0.26])
    far = np.array([0.70, 0.10, 0.10, 0.10])
    assert psi(ref, near) < psi(ref, far)


def test_genre_distribution_sums_to_one_interaction_weighted():
    genres = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)  # 3 films, 2 genres
    items = np.array([0, 0, 1, 2])  # 4 interactions
    dist, n = genre_distribution(items, genres)
    assert n == 4
    # genre 0 appears in items 0,0,2 (3 of 4); genre 1 appears in items 1,2 (2 of 4)
    assert np.allclose(dist, [3 / 4, 2 / 4])


def test_genre_distribution_handles_no_interactions():
    genres = np.zeros((5, 3), dtype=np.float32)
    dist, n = genre_distribution(np.array([], dtype=np.int64), genres)
    assert n == 0
    assert np.all(dist == 0)
