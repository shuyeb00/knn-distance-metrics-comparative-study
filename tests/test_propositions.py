import numpy as np
import pytest

from knnmetrics.distances import pairwise_block
from knnmetrics.neighbour_engine import kneighbors

rng = np.random.default_rng(2026)


@pytest.mark.parametrize("p,metric", [(1, "l1"), (2, "l2"), (3, "l3"), (0.5, "l0.5"), (np.inf, "linf")])
def test_p1_diagonal_affine_map_is_weighted_lp(p, metric):
    X, Y = rng.normal(size=(40, 7)), rng.normal(size=(30, 7))
    a, b = rng.uniform(0.1, 5, size=7), rng.normal(size=7)
    lhs = pairwise_block(X * a + b, Y * a + b, metric)
    diff = np.abs(X[:, None, :] - Y[None, :, :]) * a
    rhs = diff.max(-1) if np.isinf(p) else (diff ** p).sum(-1) ** (1 / p)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


@pytest.mark.parametrize("metric", ["l0.5", "l1", "l2", "l3", "linf"])
def test_p1_minmax_and_zscore_agree_when_range_to_sd_ratio_is_constant(metric):
    # Every training column is a permutation of the same values, so range/sd is identical across features.
    col = rng.normal(size=200)
    tr = np.column_stack([rng.permutation(col) for _ in range(6)]) * rng.uniform(0.5, 20, size=6) + rng.normal(size=6)
    te = rng.normal(size=(100, 6)) * tr.std(0) + tr.mean(0)
    lo, hi = tr.min(0), tr.max(0)
    mm = lambda Z: (Z - lo) / (hi - lo)
    mu, sd = tr.mean(0), tr.std(0)
    zs = lambda Z: (Z - mu) / sd
    ratio = (hi - lo) / sd
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
    a = kneighbors(mm(tr), mm(te), 15, metric)
    b = kneighbors(zs(tr), zs(te), 15, metric)
    np.testing.assert_allclose(a.distances * ratio[0], b.distances, rtol=1e-9)
    D = np.sort(pairwise_block(mm(te), mm(tr), metric), 1)
    clean = (D[:, 15] - D[:, 14]) > 1e-9
    np.testing.assert_array_equal(np.sort(a.indices[clean], 1), np.sort(b.indices[clean], 1))


def test_p2_l2_and_cosine_identical_on_a_sphere_including_ties():
    v = np.array([3.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    X = np.array([rng.permutation(v) * rng.choice([-1, 1], size=8) for _ in range(600)])
    tr, te = X[:450], X[450:]
    a = kneighbors(tr, te, 21, "l2")
    b = kneighbors(tr, te, 21, "cosine")
    assert (a.boundary_ties > 1).mean() > 0.5
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.boundary_ties, b.boundary_ties)
    r2 = v @ v
    np.testing.assert_allclose(a.distances ** 2, 2 * r2 * b.distances, rtol=1e-12, atol=1e-12)


def test_p2_differences_arise_only_from_norm_variation():
    X = rng.normal(size=(500, 10))
    unit = X / np.linalg.norm(X, axis=1, keepdims=True)
    same = kneighbors(unit[:400], unit[400:], 7, "l2").indices
    assert (same == kneighbors(unit[:400], unit[400:], 7, "cosine").indices).mean() > 0.999
    s = np.exp(rng.uniform(-2, 2, size=(500, 1)))
    cos_scaled = kneighbors(unit[:400] * s[:400], unit[400:] * s[400:], 7, "cosine").indices
    np.testing.assert_array_equal(cos_scaled, kneighbors(unit[:400], unit[400:], 7, "cosine").indices)
    l2_scaled = kneighbors(unit[:400] * s[:400], unit[400:] * s[400:], 7, "l2").indices
    assert (l2_scaled != same).any()


def test_p3_correlation_is_cosine_after_row_centering_and_affine_invariant():
    X, Y = rng.normal(size=(50, 12)), rng.normal(size=(40, 12))
    corr = pairwise_block(X, Y, "correlation")
    cent = pairwise_block(X - X.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True), "cosine")
    np.testing.assert_allclose(corr, cent, atol=1e-12)
    alpha = rng.uniform(0.2, 5, size=(50, 1))
    beta = rng.normal(size=(50, 1)) * 10
    np.testing.assert_allclose(pairwise_block(alpha * X + beta, Y, "correlation"), corr, atol=1e-10)
    np.testing.assert_allclose(pairwise_block(alpha * X, Y, "cosine"), pairwise_block(X, Y, "cosine"), atol=1e-12)
    assert not np.allclose(pairwise_block(X + beta, Y, "cosine"), pairwise_block(X, Y, "cosine"))


@pytest.mark.parametrize("q", [2, 4, 16, 256])
def test_p4_linf_takes_at_most_q_values_on_an_equally_spaced_grid(q):
    D_dim = 64
    X = rng.integers(0, q, size=(800, D_dim)).astype(np.float64) * 0.37 + 1.5
    Dinf = pairwise_block(X[:100], X[100:], "linf")
    assert len(np.unique(Dinf)) <= q
    assert len(np.unique(pairwise_block(X[:100], X[100:], "l1"))) > q or q == 256


def test_p4_bound_requires_equal_spacing():
    levels = np.array([0.0, 1.0, 3.0, 7.0])
    X = levels[rng.integers(0, 4, size=(400, 1))]
    values = np.unique(pairwise_block(X, X, "linf"))
    assert len(values) == 7 > len(levels)
    assert len(values) <= len(levels) * (len(levels) - 1) // 2 + 1
