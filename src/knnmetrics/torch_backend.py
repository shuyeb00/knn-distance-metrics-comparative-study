import numpy as np


def _torch():
    import torch
    return torch


def pairwise_block_torch(Q, T, metric):
    from .distances import snap
    return snap(_pairwise_block_torch(Q, T, metric), _torch())


def _pairwise_block_torch(Q, T, metric):
    """float64 distances on the given device. L_p use torch.cdist without the matrix-product shortcut,
    so integer-valued inputs give exact sums as in the SciPy backend."""
    torch = _torch()
    p = {"l1": 1.0, "l2": 2.0, "l3": 3.0, "l0.5": 0.5, "linf": float("inf")}
    if metric in p:
        return torch.cdist(Q, T, p=p[metric], compute_mode="donot_use_mm_for_euclid_dist")
    if metric in ("cosine", "correlation"):
        from .distances import angular_block
        return angular_block(Q, T, metric, torch)
    raise ValueError(f"unknown metric {metric!r}")


def select_torch(D, k, tie_ks=()):
    """Same rule as the NumPy engine: lowest column index among tied distances."""
    torch = _torch()
    kth = torch.kthvalue(D, k, dim=1, keepdim=True).values
    less = D < kth
    equal = D == kth
    need = k - less.sum(dim=1, keepdim=True)
    take = less | (equal & (torch.cumsum(equal, dim=1) <= need))
    cols = torch.sort((~take).to(torch.int8), dim=1, stable=True).indices[:, :k]
    d = torch.gather(D, 1, cols)
    d, order = torch.sort(d, dim=1, stable=True)
    cols = torch.gather(cols, 1, order)
    ties_at = {K: (D == d[:, K - 1:K]).sum(dim=1) for K in tie_ks}
    return d, cols, equal.sum(dim=1), less.sum(dim=1), ties_at


def kneighbors_torch(X_train, X_query, k, metric, device, chunk_bytes=2**30, tie_ks=()):
    torch = _torch()
    from .neighbour_engine import NeighbourResult
    T = torch.as_tensor(np.ascontiguousarray(X_train), dtype=torch.float64, device=device)
    Qall = torch.as_tensor(np.ascontiguousarray(X_query), dtype=torch.float64, device=device)
    n, m = T.shape[0], Qall.shape[0]
    rows = max(1, int(chunk_bytes // (n * 8 * 3)))
    out = {"d": [], "i": [], "t": [], "b": [], "ta": {K: [] for K in tie_ks}}
    for s in range(0, m, rows):
        D = pairwise_block_torch(Qall[s:s + rows], T, metric)
        d, c, t, b, ta = select_torch(D, k, tie_ks)
        out["d"].append(d.cpu()); out["i"].append(c.cpu()); out["t"].append(t.cpu()); out["b"].append(b.cpu())
        for K in tie_ks:
            out["ta"][K].append(ta[K].cpu())
        del D
    cat = lambda xs: torch.cat(xs).numpy()
    return NeighbourResult(cat(out["d"]), cat(out["i"]).astype(np.int64), cat(out["t"]).astype(np.int64),
                           cat(out["b"]).astype(np.int64), {K: cat(v).astype(np.int64) for K, v in out["ta"].items()})
