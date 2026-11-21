import numpy as np
import pytest

from knnmetrics.neighbour_engine import kneighbors
from knnmetrics.preprocessing import EMBEDDING, TABULAR, FoldPreprocessor

rng = np.random.default_rng(11)
METHODS = sorted(set(TABULAR) | set(EMBEDDING))


def _data():
    tr = rng.lognormal(size=(300, 8))
    tr[rng.random(tr.shape) < 0.03] = np.nan
    tr[:, 5] = 2.0
    te = rng.lognormal(size=(120, 8)) * 3
    te[:, 5] = rng.normal(size=120)
    return tr, te


@pytest.mark.parametrize("method", METHODS)
def test_transform_of_a_row_does_not_depend_on_other_query_rows(method):
    tr, te = _data()
    te = np.nan_to_num(te)
    p = FoldPreprocessor(method).fit(tr)
    full = p.transform(te)
    alone = np.vstack([p.transform(te[i:i + 1]) for i in range(len(te))])
    np.testing.assert_allclose(full, alone, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_fit_ignores_query_rows(method):
    tr, te = _data()
    a = FoldPreprocessor(method).fit(tr).transform(tr)
    b = FoldPreprocessor(method).fit(tr.copy()).transform(tr)
    np.testing.assert_array_equal(a, b)
    pooled = FoldPreprocessor(method).fit(np.vstack([tr, np.nan_to_num(te) * 50])).transform(tr)
    if method not in ("none", "l2"):
        assert pooled.shape != a.shape or not np.allclose(pooled, a)


def test_training_constant_column_is_dropped_and_imputation_uses_training_medians():
    tr, te = _data()
    p = FoldPreprocessor("zscore").fit(tr)
    assert p.keep_.sum() == 7 and not p.keep_[5]
    np.testing.assert_allclose(p.medians_, np.nanmedian(tr, axis=0))
    out = p.transform(np.full((1, 8), np.nan))
    np.testing.assert_allclose(out, p.model_.transform(p.medians_[p.keep_][None, :]))


def test_mahalanobis_uses_training_rows_only():
    tr = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))
    q1 = rng.normal(size=(30, 5))
    q2 = np.vstack([q1, rng.normal(size=(500, 5)) * 100])
    a = kneighbors(tr, q1, 7, "mahalanobis")
    b = kneighbors(tr, q2, 7, "mahalanobis")
    np.testing.assert_array_equal(a.indices, b.indices[:30])
    np.testing.assert_allclose(a.distances, b.distances[:30], rtol=1e-12)
