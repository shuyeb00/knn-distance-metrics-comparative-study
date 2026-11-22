"""Exact k-NN search for L_p metrics with a float32 candidate pass and float64 refinement.

The float32 pass computes, for every (query, training row) pair, an approximation S~ of the power sum
S = sum_i |q_i - t_i|^p (the maximum for p = inf, the squared distance for p = 2). Every candidate is then
recomputed with the float64 engine (same kernel, same 40-bit snapping), and the neighbour list is chosen
by (distance, training index) among the candidates. A query is accepted only if a rigorous bound

    |S~ - S| <= A * S~ + B * (N(q) + N(t)) + C

shows that every non-candidate row has a float64 distance strictly above the k-th selected distance. Then
the selected list, the distances and every tie count equal those of a full float64 search. Queries that
fail are retried with a wider candidate set and finally searched in full float64.

The bounds hold for any summation order. They assume IEEE single-precision arithmetic with unit roundoff
u = 2^-24 for +, -, *, a square root within 2u, and absolute error at most 2^-126 per operation on
underflow (flush to zero included). Inputs are centred before rounding to float32; L_p distances are
translation invariant, and the centring error is part of the bound.
"""
import numpy as np

from .distances import snap

FAST_METRICS = ("l0.5", "l1", "l2", "l3", "linf")
_P = {"l0.5": 0.5, "l1": 1.0, "l2": 2.0, "l3": 3.0, "linf": float("inf")}
_KERNEL_CODE = {"linf": 0, "l1": 1, "l3": 3, "l0.5": 5}
_U = 2.0 ** -24
_U64 = 2.0 ** -53
_TINY = 2.0 ** -126
_SAFE = 2.0 ** -45
_MAX_ABS = 2.0 ** 30
_MIN_ABS = 2.0 ** -100


def _torch():
    import torch
    return torch


def _gamma(n):
    return n * _U / (1 - n * _U)


def input_properties(Xtr, Xq):
    """Centre on the training mean (rounded to an integer for integer data, so centring is exact) and
    report which error terms of the float32 pass vanish."""
    integer = bool(np.array_equal(Xtr, np.round(Xtr)) and np.array_equal(Xq, np.round(Xq)))
    mu = Xtr.mean(axis=0)
    if integer:
        mu = np.round(mu)
    Ctr, Cq = Xtr - mu, Xq - mu
    a = max(float(np.abs(Ctr).max(initial=0)), float(np.abs(Cq).max(initial=0)))
    nz = [np.abs(c[c != 0]) for c in (Ctr, Cq)]
    smallest = min((float(v.min()) for v in nz if v.size), default=1.0)
    exact32 = all(np.array_equal(c.astype(np.float32).astype(np.float64), c) for c in (Ctr, Cq))
    return Ctr, Cq, {
        "supported": bool(np.isfinite(a) and a <= _MAX_ABS and smallest >= _MIN_ABS),
        "exact32": exact32,
        "exact_diff": bool(integer and a < 2.0 ** 23),
        "centred_exactly": integer and a < 2.0 ** 52,
    }


def error_model(metric, dim, props):
    """(A, B, C, norm) such that |S~ - S| <= A S~ + B (norm(q) + norm(t)) + C for every pair.

    delta bounds the relative error of a float32 difference of exactly represented inputs; eta bounds,
    relative to |q_i| + |t_i|, the error from centring in float64 and rounding the inputs to float32."""
    D = dim
    delta = 0.0 if props["exact_diff"] else _U / (1 - _U)
    eta = (0.0 if props["exact32"] else _U) + (0.0 if props["centred_exactly"] else 2 * _U64)
    eta_w = delta * (1 + _U) ** 2 + eta
    C = 8 * (D + 4) * _TINY
    if metric == "l1":
        g = _gamma(D)
        A, B, norm = (g + delta) / (1 - g), eta, "l1"
    elif metric == "linf":
        A, B, norm = delta, eta, "linf"
    elif metric == "l3":
        g = _gamma(D + 1)
        if eta == 0:
            A, B = (g + 3 * delta * (1 + delta) ** 2) / (1 - g), 0.0
        else:
            A, B = g / (1 - g), 12 * eta_w * (1 + _U) ** 4
        norm = "l3"
    elif metric == "l0.5":
        g = _gamma(D + 1)
        if eta == 0:
            A, B = (g + delta) / (1 - g), 0.0
        else:
            A, B = g / (1 - g), float(np.sqrt(eta_w))
        norm = "l0.5"
    elif metric == "l2":
        K = _gamma(D + 4) * (1 + _U) ** 2 + eta * (2 + eta)
        A, B, norm = 0.0, 2 * K, "l2"
    else:
        raise ValueError(f"no float32 error model for {metric!r}")
    inflate = 1 + 2.0 ** -20
    return A * inflate, B * inflate, C, norm


def row_norms(X, norm):
    a = np.abs(X)
    if norm == "l1":
        v = a.sum(axis=1)
    elif norm == "linf":
        v = a.max(axis=1, initial=0.0)
    elif norm == "l3":
        v = (a * a * a).sum(axis=1)
    elif norm == "l0.5":
        v = np.sqrt(a).sum(axis=1)
    elif norm == "l2":
        v = (a * a).sum(axis=1)
    return v * (1 + (X.shape[1] + 4) * 2 * _U64)


def reference_rho(dim):
    """Bound on the relative error of the float64 engine (summation, pow, root and 40-bit snapping)."""
    return 2.0 ** -38 + 8 * (dim + 16) * _U64


def _distance_from_power_sum(S, metric, torch):
    S = torch.clamp(S, min=0.0)
    if metric == "l2":
        return torch.sqrt(S)
    if metric == "l3":
        return torch.pow(S, 1.0 / 3.0)
    if metric == "l0.5":
        return S * S
    return S


_TRITON = {}


def _triton_kernel():
    if "kernel" in _TRITON:
        return _TRITON["kernel"]
    try:
        import triton
        import triton.language as tl
    except Exception:
        _TRITON["kernel"] = None
        return None

    @triton.jit
    def power_sum_kernel(Q, T, S, m, n, D, sqm, sqd, stn, std, ssm, ssn,
                         P: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            q = tl.load(Q + rm[:, None] * sqm + rk[None, :] * sqd, mask=(rm[:, None] < m) & (rk[None, :] < D),
                        other=0.0)
            t = tl.load(T + rn[:, None] * stn + rk[None, :] * std, mask=(rn[:, None] < n) & (rk[None, :] < D),
                        other=0.0)
            diff = tl.abs(q[:, None, :] - t[None, :, :])
            if P == 0:
                acc = tl.maximum(acc, tl.max(diff, axis=2))
            elif P == 1:
                acc += tl.sum(diff, axis=2)
            elif P == 3:
                acc += tl.sum(diff * diff * diff, axis=2)
            else:
                acc += tl.sum(tl.sqrt_rn(diff), axis=2)
        tl.store(S + rm[:, None] * ssm + rn[None, :] * ssn, acc, mask=(rm[:, None] < m) & (rn[None, :] < n))

    _TRITON["kernel"] = (triton, power_sum_kernel)
    return _TRITON["kernel"]


TRITON_TILE = {"BM": 32, "BN": 32, "BK": 8, "num_warps": 4}


def power_sums(Q32, T32, metric, kernel="auto", tile_elems=2 ** 24):
    """float32 S~ for every pair: the squared distance for l2, the maximum for linf, else sum |d|^p."""
    torch = _torch()
    m, n = Q32.shape[0], T32.shape[0]
    if metric == "l2":
        S = (Q32 * Q32).sum(dim=1)[:, None] + (T32 * T32).sum(dim=1)[None, :]
        return S.addmm_(Q32, T32.T, beta=1.0, alpha=-2.0)
    use_triton = kernel == "triton" or (kernel == "auto" and Q32.is_cuda)
    tk = _triton_kernel() if use_triton else None
    if tk is not None:
        triton, fn = tk
        S = torch.empty((m, n), dtype=torch.float32, device=Q32.device)
        t = TRITON_TILE
        grid = (triton.cdiv(m, t["BM"]), triton.cdiv(n, t["BN"]))
        try:
            fn[grid](Q32, T32, S, m, n, Q32.shape[1], Q32.stride(0), Q32.stride(1), T32.stride(0), T32.stride(1),
                     S.stride(0), S.stride(1), P=_KERNEL_CODE[metric], BM=t["BM"], BN=t["BN"], BK=t["BK"],
                     num_warps=t["num_warps"])
            return S
        except Exception as e:
            if kernel == "triton":
                raise
            import warnings
            warnings.warn(f"triton kernel failed ({type(e).__name__}: {e}); using the torch kernel from now on")
            _TRITON["kernel"] = None
    if kernel == "triton":
        raise RuntimeError("triton is not available")
    S = torch.empty((m, n), dtype=torch.float32, device=Q32.device)
    D = Q32.shape[1]
    bt = max(1, min(n, tile_elems // max(D, 1)))
    bq = max(1, tile_elems // (bt * max(D, 1)))
    for i in range(0, m, bq):
        q = Q32[i:i + bq, None, :]
        for j in range(0, n, bt):
            diff = (q - T32[None, j:j + bt, :]).abs_()
            if metric == "linf":
                S[i:i + bq, j:j + bt] = diff.amax(dim=2)
                continue
            if metric == "l3":
                diff = diff * diff * diff
            elif metric == "l0.5":
                diff.sqrt_()
            S[i:i + bq, j:j + bt] = diff.sum(dim=2)
    return S


def _refine(Q64, T64, rows, cand, metric, budget=2 ** 25):
    from .torch_backend import pairwise_block_torch
    torch = _torch()
    c, D = cand.shape[1], T64.shape[1]
    out = torch.empty(cand.shape, dtype=torch.float64, device=T64.device)
    per = max(1, budget // (c * D))
    for s in range(0, rows.shape[0], per):
        r, cc = rows[s:s + per], cand[s:s + per]
        out[s:s + per] = pairwise_block_torch(Q64[r][:, None, :], T64[cc], metric)[:, 0, :]
    return out


def _lower_bound_outside(S, vals, cand, Bt, Bq, A, B, C, budget=2 ** 24):
    """Lower bound of the exact power sum over all rows that are not candidates, per query."""
    torch = _torch()
    a = 1.0 - A - _SAFE
    if B == 0:
        return vals.double() * a - C * (1 + _SAFE)
    out = torch.empty(S.shape[0], dtype=torch.float64, device=S.device)
    per = max(1, budget // S.shape[1])
    for s in range(0, S.shape[0], per):
        L = S[s:s + per].double() * a - Bt[None, :]
        L.scatter_(1, cand[s:s + per], float("inf"))
        out[s:s + per] = L.min(dim=1).values
    return out - (Bq + C) * (1 + _SAFE)


def _select_candidates(ref, cand, k, tie_ks):
    torch = _torch()
    by_index = torch.argsort(cand, dim=1)
    cand = torch.gather(cand, 1, by_index)
    ref = torch.gather(ref, 1, by_index)
    ref_sorted, order = torch.sort(ref, dim=1, stable=True)
    cand_sorted = torch.gather(cand, 1, order)
    dk = ref_sorted[:, k - 1:k]
    ties = (ref == dk).sum(dim=1)
    below = (ref < dk).sum(dim=1)
    ties_at = {K: (ref == ref_sorted[:, K - 1:K]).sum(dim=1) for K in tie_ks}
    return ref_sorted[:, :k], cand_sorted[:, :k], ties, below, ties_at


def kneighbors_certified(X_train, X_query, k, metric, device="cuda", chunk_bytes=2 ** 30, tie_ks=(), margin=33,
                         wide=512, kernel="auto", stats=None):
    """Same output as torch_backend.kneighbors_torch, bit for bit, for the L_p metrics."""
    torch = _torch()
    from .neighbour_engine import NeighbourResult
    from .torch_backend import kneighbors_torch, pairwise_block_torch, select_torch

    n, m, D = X_train.shape[0], X_query.shape[0], X_train.shape[1]
    c1 = k + margin
    props = None
    if metric in FAST_METRICS and c1 < n and D > 0:
        Ctr, Cq, props = input_properties(X_train, X_query)
    if props is None or not props["supported"]:
        if stats is not None and metric in FAST_METRICS:
            stats["queries"] = stats.get("queries", 0) + m
            stats["full_float64"] = stats.get("full_float64", 0) + m
        return kneighbors_torch(X_train, X_query, k, metric, device, chunk_bytes, tie_ks)

    A, B, C, norm = error_model(metric, D, props)
    rho = reference_rho(D)
    to = dict(device=device)
    T64 = torch.as_tensor(np.ascontiguousarray(X_train), dtype=torch.float64, **to)
    Q64 = torch.as_tensor(np.ascontiguousarray(X_query), dtype=torch.float64, **to)
    T32 = torch.as_tensor(np.ascontiguousarray(Ctr, dtype=np.float32), **to)
    Q32 = torch.as_tensor(np.ascontiguousarray(Cq, dtype=np.float32), **to)
    Bt = torch.as_tensor(B * (1 + _SAFE) * row_norms(Ctr, norm), dtype=torch.float64, **to)
    Bq_all = torch.as_tensor(B * row_norms(Cq, norm), dtype=torch.float64, **to)
    c2 = wide if c1 < wide < n else None

    dist = torch.empty((m, k), dtype=torch.float64, **to)
    idx = torch.empty((m, k), dtype=torch.int64, **to)
    ties = torch.empty(m, dtype=torch.int64, **to)
    below = torch.empty(m, dtype=torch.int64, **to)
    ties_at = {K: torch.empty(m, dtype=torch.int64, **to) for K in tie_ks}
    counts = {"stage1": 0, "stage2": 0, "full_float64": 0}

    def store(rows, d, i, t, b, ta):
        dist[rows], idx[rows], ties[rows], below[rows] = d, i, t.to(torch.int64), b.to(torch.int64)
        for K in tie_ks:
            ties_at[K][rows] = ta[K].to(torch.int64)

    rows_per_chunk = max(1, int(chunk_bytes // (n * 4 * 3)))
    for s in range(0, m, rows_per_chunk):
        e = min(s + rows_per_chunk, m)
        S = power_sums(Q32[s:e], T32, metric, kernel)
        pending = torch.arange(s, e, device=T64.device)
        for c in (c1, c2):
            if c is None or pending.numel() == 0:
                continue
            Sp = S[pending - s]
            vals, cand = torch.topk(Sp, c + 1, dim=1, largest=False, sorted=True)
            cand, lim = cand[:, :c], vals[:, c]
            low = _lower_bound_outside(Sp, lim, cand, Bt, Bq_all[pending], A, B, C)
            ref = _refine(Q64, T64, pending, cand, metric)
            d, i, t, b, ta = _select_candidates(ref, cand, k, tie_ks)
            ok = _distance_from_power_sum(low, metric, torch) * (1 - 2 * rho) > d[:, k - 1]
            if ok.any():
                store(pending[ok], d[ok], i[ok], t[ok], b[ok], {K: v[ok] for K, v in ta.items()})
            counts["stage1" if c == c1 else "stage2"] += int(ok.sum())
            pending = pending[~ok]
        del S
        if pending.numel():
            per = max(1, int(chunk_bytes // (n * 8 * 3)))
            for a in range(0, pending.numel(), per):
                r = pending[a:a + per]
                d, i, t, b, ta = select_torch(pairwise_block_torch(Q64[r], T64, metric), k, tie_ks)
                store(r, d, i, t, b, ta)
            counts["full_float64"] += int(pending.numel())

    if stats is not None:
        stats["queries"] = stats.get("queries", 0) + m
        for key, v in counts.items():
            stats[key] = stats.get(key, 0) + v
    cpu = lambda x: x.cpu().numpy()
    return NeighbourResult(cpu(dist), cpu(idx), cpu(ties), cpu(below), {K: cpu(v) for K, v in ties_at.items()})
