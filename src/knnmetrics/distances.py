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


def pairwise_block(Q, T, metric):
    """Exact float64 distances between query rows Q and training rows T, one pair at a time.

    scipy's cdist evaluates each pair directly rather than through a matrix product, so equal
    inputs give bit-identical distances regardless of thread count; exact ties are preserved.
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
        with np.errstate(invalid="ignore", divide="ignore"):
            D = cdist(Q, T, metric)
        return np.where(np.isnan(D), 1.0, D)
    raise ValueError(f"unknown metric {metric!r}")


def degenerate_rows(X, metric):
    if metric == "cosine":
        return int(np.sum(np.linalg.norm(X, axis=1) == 0))
    if metric == "correlation":
        return int(np.sum(np.ptp(X, axis=1) == 0))
    return 0
