import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.preprocessing import StandardScaler


def _split(X, y, seed=0, frac=0.7):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    cut = int(frac * len(y))
    return X[perm[:cut]], y[perm[:cut]], X[perm[cut:]], y[perm[cut:]]


@pytest.fixture(scope="session")
def bundled_tasks():
    tasks = {}
    for name, loader in [("iris", load_iris), ("wine", load_wine),
                         ("breast_cancer", load_breast_cancer), ("digits", load_digits)]:
        X, y = loader(return_X_y=True)
        X = X.astype(np.float64)
        if name != "digits":
            X = StandardScaler().fit_transform(X)
        tasks[name] = _split(X, y)
    return tasks


@pytest.fixture(scope="session")
def quantized_task():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 4, size=(1500, 20)).astype(np.float64)
    y = (X[:, :3].sum(axis=1) + rng.integers(0, 2, size=1500)) % 3
    return _split(X, y)
