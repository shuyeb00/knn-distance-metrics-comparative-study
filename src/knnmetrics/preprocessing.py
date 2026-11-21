import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler

TABULAR = ("none", "minmax", "zscore", "robust", "quantile")
EMBEDDING = ("none", "l2", "center_l2", "zscore", "pca_whiten")

_EPS = 1e-12


def _row_l2(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, _EPS)


class FoldPreprocessor:
    """Training-fold-only preprocessing: median imputation, removal of columns that are constant
    on the training fold, then the named transform. Nothing is estimated from rows passed to transform."""

    def __init__(self, method, seed=0):
        if method not in set(TABULAR) | set(EMBEDDING):
            raise ValueError(f"unknown preprocessing {method!r}")
        self.method = method
        self.seed = seed

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.medians_ = np.nanmedian(X, axis=0) if np.isnan(X).any() else None
        Xi = self._impute(X)
        self.keep_ = np.ptp(Xi, axis=0) > 0
        Xk = Xi[:, self.keep_]
        m = self.method
        if m == "minmax":
            self.model_ = MinMaxScaler().fit(Xk)
        elif m == "zscore":
            self.model_ = StandardScaler().fit(Xk)
        elif m == "robust":
            self.model_ = RobustScaler().fit(Xk)
        elif m == "quantile":
            self.model_ = QuantileTransformer(
                n_quantiles=min(1000, Xk.shape[0]), output_distribution="normal",
                subsample=None, random_state=self.seed).fit(Xk)
        elif m == "center_l2":
            self.mean_ = Xk.mean(axis=0)
        elif m == "pca_whiten":
            self.model_ = PCA(whiten=True, svd_solver="full", random_state=self.seed).fit(Xk)
            # Components with negligible variance are dropped so whitening does not amplify noise.
            ev = self.model_.explained_variance_
            self.n_components_ = int(np.sum(ev > ev[0] * 1e-10))
        return self

    def _impute(self, X):
        if self.medians_ is None:
            if np.isnan(X).any():
                raise ValueError("missing values at transform time but none in the training fold")
            return X
        return np.where(np.isnan(X), self.medians_, X)

    def transform(self, X):
        X = self._impute(np.asarray(X, dtype=np.float64))[:, self.keep_]
        m = self.method
        if m == "none":
            return X
        if m in ("minmax", "zscore", "robust", "quantile"):
            return self.model_.transform(X)
        if m == "l2":
            return _row_l2(X)
        if m == "center_l2":
            return _row_l2(X - self.mean_)
        if m == "pca_whiten":
            return self.model_.transform(X)[:, :self.n_components_]
        raise AssertionError

    def fit_transform(self, X):
        return self.fit(X).transform(X)
