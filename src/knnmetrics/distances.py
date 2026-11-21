import numpy as np
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf

METRICS = ("l0.5", "l1", "l2", "l3", "linf", "cosine", "correlation", "mahalanobis")


class MahalanobisMap:
    """Ledoit-Wolf precision fitted on the training rows; distances are L2 after x -> L^T x with P = L L^T."""

    def fit(self, X_train):
        cov = LedoitWolf().fit(X_train).covariance_
        precision = np.linalg.inv(cov)
        precision = (precision + precision.T) / 2
        self.factor_ = np.linalg.cholesky(precision)
        return self

    def transform(self, X):
        return X @ self.factor_


SNAP_BITS = 40


def snap(D, xp=np):
    """Round distances to 40 significant bits. Distances that are equal in exact arithmetic but differ in
    the last bits because of summation order (BLAS blocking, CPU versus GPU) become exactly equal, so the
    tie rule, not floating-point noise, decides between them. The relative change is below 1e-12."""
    m, e = xp.frexp(D)
    if xp is np:
        return np.ldexp(np.round(m * 2.0 ** SNAP_BITS), e - SNAP_BITS)
    return xp.ldexp(xp.round(m * 2.0 ** SNAP_BITS), (e - SNAP_BITS).to(D.dtype))


def pairwise_block(Q, T, metric):
    return snap(_pairwise_block(Q, T, metric))


def _pairwise_block(Q, T, metric):
    """Exact float64 distances between query rows Q and training rows T.

    scipy's cdist evaluates each pair directly rather than through a matrix product, so results do not
    depend on chunking or thread count. Cosine and correlation on integer-valued data use the exact
    shared formula in angular_block instead, so that mathematically tied distances stay tied.
    Zero-norm rows under cosine (and constant rows under correlation) get distance 1, the value
    an epsilon-normalised zero vector would give.
    """
    if metric == "l1":
        return cdist(Q, T, "cityblock")
    if metric == "l2":
        return cdist(Q, T, "euclidean")
    if metric == "linf":
        return cdist(Q, T, "chebyshev")
    if metric == "l3":
        return cdist(Q, T, "minkowski", p=3)
    if metric == "l0.5":
        return cdist(Q, T, "minkowski", p=0.5)
    if metric in ("cosine", "correlation"):
        if is_integer_valued(Q, T):
            return angular_block(Q, T, metric, np)
        with np.errstate(invalid="ignore", divide="ignore"):
            D = cdist(Q, T, metric)
        return np.where(np.isnan(D), 1.0, D)
    raise ValueError(f"unknown metric {metric!r}")


def dyadic_scale(*arrays, max_shift=24):
    """Smallest k such that every value times 2**k is an integer below 2**20 in magnitude, else None.
    Integer-valued data (k = 0) and integers rescaled by a power of two qualify; for such data the
    sums in angular_block are exact."""
    if any(a.size and not np.isfinite(a).all() for a in arrays):
        return None
    for k in range(max_shift + 1):
        scaled = [a * 2.0 ** k for a in arrays]
        if all(a.size == 0 or float(np.abs(a).max()) >= 2**20 for a in scaled):
            return None
        if all((a == np.round(a)).all() and float(np.abs(a).max(initial=0)) < 2**20 for a in scaled):
            return k
    return None


def is_integer_valued(*arrays):
    return dyadic_scale(*arrays) is not None


def angular_block(Q, T, metric, xp):
    """Cosine and correlation distance with one formula shared by the NumPy and PyTorch backends.

    On integer-valued inputs every sum below is an exact integer in float64, and the remaining
    operations are single correctly rounded IEEE operations, so both backends return bit-identical
    distances and exact ties survive. Correlation then uses the uncentred moment form; on other inputs
    it centres the rows first to avoid cancellation. Zero-norm (or constant) rows get distance 1.
    """
    k = dyadic_scale(Q, T) if xp is np else dyadic_scale(Q.cpu().numpy(), T.cpu().numpy())
    exact = k is not None
    if exact and k:
        Q, T = Q * 2.0 ** k, T * 2.0 ** k
    if metric == "correlation" and not exact:
        Q = Q - Q.mean(1, keepdims=True) if xp is np else Q - Q.mean(dim=1, keepdim=True)
        T = T - T.mean(1, keepdims=True) if xp is np else T - T.mean(dim=1, keepdim=True)
    dot = Q @ T.T
    qq, tt = (Q * Q).sum(1), (T * T).sum(1)
    if metric == "correlation" and exact:
        d = Q.shape[1]
        sq, st = Q.sum(1), T.sum(1)
        dot = d * dot - sq[:, None] * st[None, :]
        qq, tt = d * qq - sq * sq, d * tt - st * st
    if xp is np:
        rq, rt = np.sqrt(qq), np.sqrt(tt)
    else:
        # torch's vectorised CPU sqrt is not always correctly rounded; NumPy's is, so roots are taken there.
        rq = xp.as_tensor(np.sqrt(qq.cpu().numpy()), device=qq.device)
        rt = xp.as_tensor(np.sqrt(tt.cpu().numpy()), device=tt.device)
    denom = rq[:, None] * rt[None, :]
    zero = denom == 0
    safe = xp.where(zero, xp.ones_like(denom), denom)
    D = 1.0 - dot / safe
    return xp.where(zero, xp.ones_like(D), D)


def degenerate_rows(X, metric):
    if metric == "cosine":
        return int(np.sum(np.linalg.norm(X, axis=1) == 0))
    if metric == "correlation":
        return int(np.sum(np.ptp(X, axis=1) == 0))
    return 0
