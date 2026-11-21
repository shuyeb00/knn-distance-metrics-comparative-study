import numpy as np
import pytest

from knnmetrics.distances import METRICS
from knnmetrics.neighbour_engine import kneighbors

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("metric", METRICS)
def test_torch_backend_matches_numpy_on_integer_data(metric, quantized_task):
    Xtr, _, Xq, _ = quantized_task
    a = kneighbors(Xtr, Xq, 31, metric, tie_ks=(1, 5, 31))
    b = kneighbors(Xtr, Xq, 31, metric, tie_ks=(1, 5, 31), backend="torch", device="cpu", chunk_bytes=2**18)
    np.testing.assert_allclose(b.distances, a.distances, rtol=1e-12, atol=1e-12)
    if metric in ("l1", "l2", "linf", "cosine", "correlation"):
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_array_equal(a.distances, b.distances)
        for K in (1, 5, 31):
            np.testing.assert_array_equal(a.ties_at[K], b.ties_at[K])
    else:
        assert (a.indices == b.indices).all(axis=1).mean() > 0.99


@pytest.mark.parametrize("metric", METRICS)
def test_torch_backend_matches_numpy_on_continuous_data(metric, bundled_tasks):
    for Xtr, _, Xq, _ in bundled_tasks.values():
        a = kneighbors(Xtr, Xq, 15, metric)
        b = kneighbors(Xtr, Xq, 15, metric, backend="torch", device="cpu")
        np.testing.assert_allclose(b.distances, a.distances, rtol=1e-10, atol=1e-12)
        assert (a.indices == b.indices).all(axis=1).mean() > 0.99
