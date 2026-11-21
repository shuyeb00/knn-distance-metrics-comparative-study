import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

from knnmetrics.evaluation import deduplicate, outer_splits, scores_from_predictions, stratified_cap

rng = np.random.default_rng(5)


@pytest.mark.parametrize("n_classes", [2, 3, 7])
def test_scores_match_sklearn(n_classes):
    for _ in range(20):
        y = rng.integers(0, n_classes, size=200)
        p = np.where(rng.random(200) < 0.6, y, rng.integers(0, n_classes, size=200))
        s = scores_from_predictions(y, p, n_classes)
        assert s["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, p), abs=1e-12)
        assert s["accuracy"] == pytest.approx(accuracy_score(y, p), abs=1e-12)
        assert s["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"), abs=1e-12)
        assert s["kappa"] == pytest.approx(cohen_kappa_score(y, p), abs=1e-12)


def test_deduplicate_rules():
    X = np.array([[0, 0], [1, 1], [0, 0], [2, 2], [2, 2], [3, np.nan], [3, np.nan]], dtype=float)
    y = np.array([0, 1, 0, 0, 1, 1, 1])
    keep, info = deduplicate(X, y)
    np.testing.assert_array_equal(keep, [0, 1, 5])
    assert info["n_rows_conflicting_removed"] == 2 and info["n_rows_duplicate_removed"] == 2


def test_splits_are_sorted_stratified_and_cover_every_row():
    y = rng.integers(0, 4, size=500)
    seen = np.zeros((3, 500), dtype=int)
    for r, f, tr, te in outer_splits(y):
        assert np.all(np.diff(tr) > 0) and np.all(np.diff(te) > 0)
        assert len(np.intersect1d(tr, te)) == 0
        seen[r, te] += 1
    assert (seen == 1).all()
    cap = stratified_cap(y, 100)
    assert len(cap) == 100 and np.all(np.diff(cap) > 0)
