import os
import subprocess
import sys

import numpy as np
import pytest

from knnmetrics.distances import METRICS, MahalanobisMap
from knnmetrics.neighbour_engine import kneighbors

torch = pytest.importorskip("torch")

from knnmetrics.certified_search import (FAST_METRICS, error_model, input_properties, power_sums,  # noqa: E402
                                         row_norms)

TIE_KS = (1, 3, 5, 7, 11, 15, 21, 31)


def assert_same(a, b):
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.distances, b.distances)
    np.testing.assert_array_equal(a.boundary_ties, b.boundary_ties)
    np.testing.assert_array_equal(a.below_boundary, b.below_boundary)
    assert set(a.ties_at) == set(b.ties_at)
    for K in a.ties_at:
        np.testing.assert_array_equal(a.ties_at[K], b.ties_at[K])


def both(Xtr, Xq, k, metric, chunk_bytes=2**20, tie_ks=TIE_KS):
    tie_ks = tuple(K for K in tie_ks if K <= k)
    ref = kneighbors(Xtr, Xq, k, metric, tie_ks=tie_ks, backend="torch", device="cpu", chunk_bytes=chunk_bytes)
    stats = {}
    fast = kneighbors(Xtr, Xq, k, metric, tie_ks=tie_ks, backend="torch_certified", device="cpu",
                      chunk_bytes=chunk_bytes, stats=stats)
    return ref, fast, stats


def datasets():
    rng = np.random.default_rng(3)
    out = {}
    out["gaussian"] = rng.normal(size=(900, 24))
    out["offset"] = rng.normal(size=(900, 12)) * 1e-3 + 1e3
    out["heavy_tail"] = rng.standard_t(2, size=(900, 16))
    q = rng.integers(0, 4, size=(900, 20)).astype(np.float64)
    out["quantized"] = q
    pix = rng.integers(0, 256, size=(900, 48)).astype(np.float64)
    out["pixels"] = pix
    dup = rng.normal(size=(60, 10))
    out["duplicates"] = dup[rng.integers(0, 60, size=900)]
    const = rng.normal(size=(900, 15))
    const[::7] = 0.0
    const[1::9] = 2.5
    const[:, 3] = 4.0
    out["constant_rows"] = const
    lat = np.round(rng.normal(size=(900, 9)) * 4) / 8
    out["dyadic"] = lat
    out["embedding_like"] = (rng.normal(size=(900, 64)) * 0.3 + rng.normal(size=64)).astype(np.float16).astype(
        np.float64)
    return out


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("name", list(datasets()))
def test_identical_to_float64_engine(metric, name):
    X = datasets()[name]
    Xtr, Xq = X[:700], X[700:]
    ref, fast, stats = both(Xtr, Xq, 31, metric)
    assert_same(ref, fast)
    if metric in FAST_METRICS or metric == "mahalanobis":
        assert stats["queries"] == len(Xq)
        assert stats.get("stage1", 0) + stats.get("stage2", 0) + stats.get("full_float64", 0) == len(Xq)


@pytest.mark.parametrize("metric", METRICS)
def test_identical_on_bundled_and_quantized_tasks(metric, bundled_tasks, quantized_task):
    for Xtr, _, Xq, _ in list(bundled_tasks.values()) + [quantized_task]:
        k = min(31, len(Xtr) - 1)
        ref, fast, _ = both(Xtr, Xq, k, metric)
        assert_same(ref, fast)


@pytest.mark.parametrize("metric", ["l1", "l2", "linf", "l0.5"])
def test_every_stage_is_exercised_and_exact(metric, quantized_task):
    from knnmetrics.certified_search import kneighbors_certified
    Xtr, _, Xq, _ = quantized_task
    ref = kneighbors(Xtr, Xq, 31, metric, tie_ks=TIE_KS, backend="torch", device="cpu")
    for margin, wide in ((1, 40), (1, 600), (33, 512)):
        stats = {}
        fast = kneighbors_certified(Xtr, Xq, 31, metric, "cpu", 2**20, TIE_KS, margin=margin, wide=wide, stats=stats)
        assert_same(ref, fast)
    if metric == "linf":
        assert stats["full_float64"] > 0


def test_small_training_sets_fall_back(bundled_tasks):
    Xtr, _, Xq, _ = bundled_tasks["iris"]
    for n in (5, 31, 40, 64):
        ref, fast, stats = both(Xtr[:n], Xq, min(31, n), "l1")
        assert_same(ref, fast)


def test_mahalanobis_maps_before_search(bundled_tasks):
    Xtr, _, Xq, _ = bundled_tasks["breast_cancer"]
    ref, fast, stats = both(Xtr, Xq, 31, "mahalanobis")
    assert_same(ref, fast)
    m = MahalanobisMap().fit(Xtr)
    ref2, fast2, _ = both(m.transform(Xtr), m.transform(Xq), 31, "l2")
    assert_same(fast, fast2)


def exact_power_sums(Xq, Xtr, metric):
    from fractions import Fraction
    out = np.empty((Xq.shape[0], Xtr.shape[0]))
    for i, q in enumerate(Xq):
        for j, t in enumerate(Xtr):
            d = [abs(Fraction(a) - Fraction(b)) for a, b in zip(q.tolist(), t.tolist())]
            if metric == "linf":
                out[i, j] = float(max(d))
            elif metric == "l1":
                out[i, j] = float(sum(d))
            elif metric == "l2":
                out[i, j] = float(sum(x * x for x in d))
            elif metric == "l3":
                out[i, j] = float(sum(x * x * x for x in d))
            else:
                out[i, j] = float(sum(np.sqrt(float(x)) for x in d))
    return out


@pytest.mark.parametrize("metric", FAST_METRICS)
@pytest.mark.parametrize("name", ["gaussian", "offset", "pixels", "embedding_like"])
def test_float32_error_bound_holds(metric, name):
    X = datasets()[name]
    Xtr, Xq = X[:40], X[700:712]
    Ctr, Cq, props = input_properties(Xtr, Xq)
    A, B, C, norm = error_model(metric, X.shape[1], props)
    S = power_sums(torch.as_tensor(Cq.astype(np.float32)), torch.as_tensor(Ctr.astype(np.float32)), metric,
                   kernel="torch").double().numpy()
    exact = exact_power_sums(Xq, Xtr, metric)
    bound = A * S + B * (row_norms(Cq, norm)[:, None] + row_norms(Ctr, norm)[None, :]) + C
    tol = 1e-12 * np.abs(exact) if metric == "l0.5" else 0.0
    assert (np.abs(S - exact) <= bound + tol).all()


def test_triton_kernel_in_interpreter_matches_bound():
    pytest.importorskip("triton")
    code = (
        "import numpy as np, torch\n"
        "from knnmetrics.certified_search import power_sums\n"
        "rng = np.random.default_rng(0)\n"
        "Q = torch.as_tensor(rng.normal(size=(37, 29)), dtype=torch.float32)\n"
        "T = torch.as_tensor(rng.normal(size=(70, 29)), dtype=torch.float32)\n"
        "for m in ('l1', 'l3', 'l0.5', 'linf'):\n"
        "    a = power_sums(Q, T, m, kernel='triton').double()\n"
        "    b = power_sums(Q, T, m, kernel='torch').double()\n"
        "    assert float(((a - b).abs() / b).max()) < 1e-5, m\n"
        "print('ok')\n")
    env = dict(os.environ, TRITON_INTERPRET="1")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")


def test_grid_fold_identical_across_backends(bundled_tasks):
    import pandas as pd
    from knnmetrics.embedding_grid import EMBEDDING_WEIGHTINGS
    from knnmetrics.evaluation import outer_splits, run_outer_fold
    from knnmetrics.preprocessing import EMBEDDING
    Xtr, ytr, Xq, yq = bundled_tasks["digits"]
    X, y = np.concatenate([Xtr, Xq]), np.concatenate([ytr, yq])
    r, f, tr, te = next(outer_splits(y))
    kw = dict(preprocessings=EMBEDDING, weightings=EMBEDDING_WEIGHTINGS, device="cpu", chunk_bytes=2**20)
    stats = {}
    a = run_outer_fold(X, y, 10, r, f, tr, te, backend="torch", **kw)
    b = run_outer_fold(X, y, 10, r, f, tr, te, backend="torch_certified", search_stats=stats, **kw)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert {m for m, v in stats.items() if v} == {m for m in METRICS if m in FAST_METRICS or m == "mahalanobis"}
    assert all(v["queries"] == v.get("stage1", 0) + v.get("stage2", 0) + v.get("full_float64", 0)
               for v in stats.values() if v)
