import numpy as np
import pytest

from knnmetrics.distances import METRICS, pairwise_block
from knnmetrics.neighbour_engine import kneighbors
from knnmetrics.voting import predict_from_neighbours

KMAX = 31


def reference_neighbours(Xtr, Xq, k, metric, priority=None):
    from knnmetrics.distances import MahalanobisMap
    if metric == "mahalanobis":
        m = MahalanobisMap().fit(Xtr)
        Xtr, Xq, metric = m.transform(Xtr), m.transform(Xq), "l2"
    D = pairwise_block(Xq, Xtr, metric)
    pr = np.arange(Xtr.shape[0]) if priority is None else priority
    order = np.lexsort((np.broadcast_to(pr, D.shape), D), axis=1)[:, :k]
    return np.take_along_axis(D, order, axis=1), order


@pytest.mark.parametrize("metric", METRICS)
def test_matches_full_lexicographic_sort(metric, bundled_tasks, quantized_task):
    for Xtr, ytr, Xq, yq in list(bundled_tasks.values()) + [quantized_task]:
        res = kneighbors(Xtr, Xq, KMAX, metric, chunk_bytes=2**16)
        d_ref, i_ref = reference_neighbours(Xtr, Xq, KMAX, metric)
        np.testing.assert_array_equal(res.indices, i_ref)
        np.testing.assert_array_equal(res.distances, d_ref)


@pytest.mark.parametrize("metric", ["l1", "l2", "linf"])
def test_prefix_property_and_tie_counts(metric, quantized_task):
    Xtr, _, Xq, _ = quantized_task
    full = kneighbors(Xtr, Xq, KMAX, metric)
    for K in (1, 5, 11):
        part = kneighbors(Xtr, Xq, K, metric)
        np.testing.assert_array_equal(part.indices, full.indices[:, :K])
        D = pairwise_block(Xq, Xtr, metric)
        kth = part.distances[:, -1:]
        np.testing.assert_array_equal(part.boundary_ties, (D == kth).sum(axis=1))
        np.testing.assert_array_equal(part.below_boundary, (D < kth).sum(axis=1))


def test_random_tie_break_is_seeded_and_changes_only_tied_sets(quantized_task):
    Xtr, _, Xq, _ = quantized_task
    a = kneighbors(Xtr, Xq, 5, "l2", tie_break="random", seed=7)
    b = kneighbors(Xtr, Xq, 5, "l2", tie_break="random", seed=7)
    base = kneighbors(Xtr, Xq, 5, "l2")
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(np.sort(a.distances, 1), np.sort(base.distances, 1))
    untied = base.boundary_ties + base.below_boundary == 5
    assert untied.any() and (~untied).any()
    np.testing.assert_array_equal(np.sort(a.indices[untied], 1), np.sort(base.indices[untied], 1))
    assert (np.sort(a.indices[~untied], 1) != np.sort(base.indices[~untied], 1)).any()


SKLEARN_METRICS = {"l1": ("manhattan", {}), "l2": ("euclidean", {}), "linf": ("chebyshev", {}),
                   "l3": ("minkowski", {"p": 3}), "cosine": ("cosine", {})}


@pytest.mark.parametrize("metric", list(SKLEARN_METRICS))
def test_agrees_with_sklearn_up_to_tie_order(metric, bundled_tasks, quantized_task):
    """scikit-learn's brute search returns the same distances. Index sets can differ only where
    the k-th distance is tied, because scikit-learn's selection among tied rows is not stable
    (it depends on chunking and thread count)."""
    from sklearn.neighbors import NearestNeighbors
    name, kw = SKLEARN_METRICS[metric]
    for Xtr, ytr, Xq, yq in list(bundled_tasks.values()) + [quantized_task]:
        res = kneighbors(Xtr, Xq, KMAX, metric)
        nn = NearestNeighbors(n_neighbors=KMAX, algorithm="brute", metric=name, **kw)
        d_sk, i_sk = nn.fit(Xtr).kneighbors(Xq)
        np.testing.assert_allclose(np.sort(d_sk, 1), res.distances, rtol=1e-9, atol=1e-12)
        D = pairwise_block(Xq, Xtr, metric)
        gap = np.sort(D, 1)
        separated = np.abs(gap[:, KMAX] - gap[:, KMAX - 1]) > 1e-9 * np.maximum(1, gap[:, KMAX - 1])
        assert separated.any()
        for r in np.flatnonzero(separated):
            assert set(i_sk[r]) == set(res.indices[r])


@pytest.mark.parametrize("weighting", ["uniform", "distance"])
@pytest.mark.parametrize("metric", ["l1", "l2", "linf"])
def test_predictions_match_sklearn_without_ties(metric, weighting, bundled_tasks):
    from sklearn.neighbors import KNeighborsClassifier
    name, kw = SKLEARN_METRICS[metric]
    for Xtr, ytr, Xq, yq in bundled_tasks.values():
        classes, ytr_codes = np.unique(ytr, return_inverse=True)
        res = kneighbors(Xtr, Xq, KMAX, metric)
        for K in (1, 5, 15):
            pred, vote_tied = predict_from_neighbours(ytr_codes[res.indices], res.distances, K,
                                                      len(classes), weighting)
            clf = KNeighborsClassifier(n_neighbors=K, algorithm="brute", metric=name,
                                       weights=weighting).fit(Xtr, ytr_codes)
            sk = clf.predict(Xq)
            D = np.sort(pairwise_block(Xq, Xtr, metric), 1)
            clean = (D[:, K] > D[:, K - 1]) & ~vote_tied
            np.testing.assert_array_equal(pred[clean], sk[clean])


def test_vote_tie_rule_lowest_class():
    labels = np.array([[2, 1, 1, 2], [0, 2, 2, 0]])
    dist = np.ones_like(labels, dtype=float)
    pred, tied = predict_from_neighbours(labels, dist, 4, 3)
    np.testing.assert_array_equal(pred, [1, 0])
    np.testing.assert_array_equal(tied, [True, True])
