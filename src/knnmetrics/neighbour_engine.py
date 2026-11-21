from dataclasses import dataclass

import numpy as np

from .distances import MahalanobisMap, pairwise_block


@dataclass
class NeighbourResult:
    distances: np.ndarray
    indices: np.ndarray
    boundary_ties: np.ndarray
    below_boundary: np.ndarray
    ties_at: dict = None

    @property
    def tie_mass(self):
        """tau_k = |{j : d_j = d_(k)}| / k at the k-th boundary."""
        return self.boundary_ties / self.indices.shape[1]


def _select(D, k, priority):
    # Exactly k columns per row: everything strictly closer than the k-th distance, then tied
    # columns in increasing priority until k are taken. Columns are already in priority order.
    kth = np.partition(D, k - 1, axis=1)[:, k - 1:k]
    less = D < kth
    equal = D == kth
    n_less = less.sum(axis=1)
    n_equal = equal.sum(axis=1)
    need = (k - n_less)[:, None]
    over = n_equal > need[:, 0]
    if over.any():
        rows = np.flatnonzero(over)
        sub = equal[rows]
        sub &= np.cumsum(sub, axis=1, dtype=np.int32) <= need[rows]
        equal_taken = equal.copy()
        equal_taken[rows] = sub
    else:
        equal_taken = equal
    less |= equal_taken
    # Exactly k entries per row are set, so the row-major nonzero positions reshape to (m, k), ascending.
    cols = np.nonzero(less)[1].reshape(D.shape[0], k)
    d = np.take_along_axis(D, cols, axis=1)
    order = np.lexsort((priority[cols], d), axis=1)
    cols = np.take_along_axis(cols, order, axis=1)
    d = np.take_along_axis(d, order, axis=1)
    return d, cols, n_equal, n_less


def kneighbors(X_train, X_query, k, metric, tie_break="index", seed=None, chunk_bytes=2**30, tie_ks=(),
               backend="numpy", device="cpu"):
    """Exact k nearest training rows for every query row, sorted by (distance, tie priority).

    tie_break="index" keeps the lower training index on equal distances, so the neighbour list
    for any K <= k is the prefix of this list. tie_break="random" uses a seeded permutation of
    the training rows as the priority instead. tie_ks lists further K values at which the number of
    training rows tied with the K-th distance is counted (stored in ties_at).
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_query = np.asarray(X_query, dtype=np.float64)
    n = X_train.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k={k} must be between 1 and n_train={n}")

    if metric == "mahalanobis":
        mapping = MahalanobisMap().fit(X_train)
        X_train, X_query, metric = mapping.transform(X_train), mapping.transform(X_query), "l2"

    if backend == "torch":
        if tie_break != "index":
            raise ValueError("the torch backend supports only the canonical tie rule")
        from .torch_backend import kneighbors_torch
        return kneighbors_torch(X_train, X_query, k, metric, device, chunk_bytes, tie_ks)

    if tie_break == "index":
        perm = np.arange(n)
    elif tie_break == "random":
        perm = np.random.default_rng(seed).permutation(n)
    else:
        raise ValueError(f"unknown tie_break {tie_break!r}")
    X_ordered = X_train[perm]
    priority = np.arange(n)

    m = X_query.shape[0]
    rows_per_chunk = max(1, int(chunk_bytes // (n * 8 * 4)))
    dist = np.empty((m, k))
    idx = np.empty((m, k), dtype=np.int64)
    ties = np.empty(m, dtype=np.int64)
    below = np.empty(m, dtype=np.int64)
    ties_at = {K: np.empty(m, dtype=np.int64) for K in tie_ks}
    for start in range(0, m, rows_per_chunk):
        stop = min(start + rows_per_chunk, m)
        D = pairwise_block(X_query[start:stop], X_ordered, metric)
        d, cols, t, b = _select(D, k, priority)
        dist[start:stop], idx[start:stop] = d, perm[cols]
        ties[start:stop], below[start:stop] = t, b
        for K in tie_ks:
            ties_at[K][start:stop] = (D == d[:, K - 1:K]).sum(axis=1)
    return NeighbourResult(dist, idx, ties, below, ties_at)
