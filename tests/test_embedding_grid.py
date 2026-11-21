import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from knnmetrics.data_embeddings import save_split, write_manifest
from knnmetrics.distances import METRICS
from knnmetrics.embedding_grid import (EMBEDDING_WEIGHTINGS, build_index, compute_unit, make_units, unit_path,
                                       write_unit)
from knnmetrics.evaluation import K_GRID

rng = np.random.default_rng(11)


def blobs(n, d, c, shift=2.0):
    y = np.arange(n) % c
    centres = rng.normal(size=(c, d)) * shift
    return (centres[y] + rng.normal(size=(n, d))).astype(np.float32), y


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    main, clip = tmp_path_factory.mktemp("features"), tmp_path_factory.mktemp("clip")
    Xtr, ytr = blobs(150, 8, 4)
    Xte, yte = blobs(60, 8, 4)
    save_split(main, "banking77", "minilm_l6", "train", Xtr, ytr)
    save_split(main, "banking77", "minilm_l6", "test", Xte, yte)
    save_split(main, "cifar10", "clip_b32", "train", Xtr, ytr)
    save_split(main, "cifar10", "clip_b32", "test", Xte, yte)
    save_split(clip, "cifar10", "clip_b32", "train", Xtr + 1, ytr)
    save_split(clip, "cifar10", "clip_b32", "test", Xte + 1, yte)
    Xf, yf = blobs(90, 6, 3)
    save_split(main, "olivetti", "raw", "all", Xf, yf)
    write_manifest(main)
    write_manifest(clip)
    return main, clip


def test_clip_comes_only_from_the_corrected_root(roots):
    main, clip = roots
    idx = build_index([main], [clip])
    assert idx[("cifar10", "clip_b32")].parent.parent == clip
    assert ("cifar10", "clip_b32") not in build_index([main])


def test_units_and_skips(roots):
    main, clip = roots
    units, skipped = make_units(build_index([main], [clip]), preprocessings=("none", "l2"))
    names = {(u["dataset"], u["representation"], u["preprocessing"], u["repeat"], u["fold"]) for u in units}
    assert ("banking77", "minilm_l6", "none", 0, 0) in names
    assert sum(1 for n in names if n[0] == "olivetti") == 2 * 15
    assert sum(1 for n in names if n[0] == "cifar10") == 2
    units, skipped = make_units(build_index([main]), preprocessings=("none",))
    assert not any(u["dataset"] == "cifar10" for u in units)


def test_official_unit_matches_sklearn_and_stores_predictions(roots, tmp_path):
    main, clip = roots
    units, _ = make_units(build_index([main], [clip]), datasets=["banking77"], preprocessings=("none",))
    df, predictions, timings = compute_unit(units[0], 2**24)
    pred = predictions["pred"]
    assert pred.shape == (len(METRICS), len(K_GRID), len(EMBEDDING_WEIGHTINGS), 60)
    assert set(df["weighting"]) == set(EMBEDDING_WEIGHTINGS)
    assert set(df["split"]) == {"outer_test", "inner_validation"}
    Xtr, ytr = blobs(150, 8, 4)
    from knnmetrics.data_embeddings import load_split
    Xtr, ytr, _ = load_split(main, "banking77", "minilm_l6", "train")
    Xte, yte, _ = load_split(main, "banking77", "minilm_l6", "test")
    for K, w, wname in ((5, "uniform", "uniform"), (11, "distance", "distance")):
        sk = KNeighborsClassifier(n_neighbors=K, weights=w, algorithm="brute").fit(Xtr.astype(float), ytr)
        expected = sk.predict(Xte.astype(float))
        got = pred[METRICS.index("l2"), K_GRID.index(K), EMBEDDING_WEIGHTINGS.index(wname)]
        np.testing.assert_array_equal(got, expected)
        row = df[(df.split == "outer_test") & (df.metric == "l2") & (df.K == K) & (df.weighting == wname)]
        assert row["balanced_accuracy"].iloc[0] == pytest.approx(balanced_accuracy_score(yte, expected), abs=1e-12)
    write_unit(unit_path(tmp_path, units[0]), df, predictions, timings, {"config_hash": "x"})
    assert unit_path(tmp_path, units[0]).exists()
    z = np.load(unit_path(tmp_path, units[0]).with_name("none_predictions.npz"))
    np.testing.assert_array_equal(z["y_test"], yte)


def test_face_fold_unit_uses_outer_cv(roots):
    main, clip = roots
    units, _ = make_units(build_index([main]), datasets=["olivetti"], preprocessings=("l2",))
    df, predictions, timings = compute_unit(units[0], 2**24)
    assert predictions is None
    assert set(df["split"]) == {"outer_test", "inner_validation"}
    assert df["fold"].nunique() == 1 and set(df["weighting"]) == set(EMBEDDING_WEIGHTINGS)
