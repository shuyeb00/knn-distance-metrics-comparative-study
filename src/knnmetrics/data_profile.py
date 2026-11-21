import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .data_tabular import array_sha256


def duplicate_summary(X, y):
    df = pd.DataFrame(X).copy()
    df["__y"] = y
    feat = list(range(X.shape[1]))
    dup_any = df.duplicated(subset=feat, keep=False)
    groups = df[dup_any].groupby(feat, dropna=False)["__y"].nunique()
    conflicting = groups[groups > 1]
    conflict_rows = 0
    if len(conflicting):
        key = pd.MultiIndex.from_frame(df.loc[dup_any, feat])
        conflict_rows = int(key.isin(conflicting.index).sum())
    return {
        "n_duplicate_rows": int(df.duplicated(subset=feat, keep="first").sum()),
        "n_duplicate_groups": int(len(groups)),
        "n_conflicting_groups": int(len(conflicting)),
        "n_rows_in_conflicting_groups": conflict_rows,
    }


def label_order_ratio(codes):
    # Ratio of observed to expected label changes between adjacent rows under a random row order;
    # values near 0 mean the file is sorted by class, which makes stable tie-breaking label-dependent.
    n = len(codes)
    counts = np.bincount(codes)
    expected = (n - 1) * (1 - np.sum(counts * (counts - 1)) / (n * (n - 1)))
    observed = int(np.sum(codes[1:] != codes[:-1]))
    return observed / expected if expected > 0 else np.nan


def id_like_columns(X, tol=0.999):
    n = X.shape[0]
    idx = np.arange(n, dtype=float)
    out = []
    for j in range(X.shape[1]):
        col = X[:, j]
        ok = ~np.isnan(col)
        if ok.sum() < 3:
            continue
        nunique = len(np.unique(col[ok]))
        if nunique < 0.95 * ok.sum():
            continue
        rho = pd.Series(col[ok]).rank().corr(pd.Series(idx[ok]).rank())
        if abs(rho) >= tol:
            out.append(j)
    return out


def one_nn_balanced_accuracy(X, codes, max_rows=5000, seed=0):
    if X.shape[0] > max_rows:
        X, _, codes, _ = train_test_split(X, codes, train_size=max_rows, stratify=codes, random_state=seed)
    min_class = np.bincount(codes)[np.bincount(codes) > 0].min()
    cv = StratifiedKFold(n_splits=int(min(5, max(2, min_class))), shuffle=True, random_state=seed)
    pred = np.empty_like(codes)
    for tr, te in cv.split(X, codes):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        sc = StandardScaler().fit(imp.transform(X[tr]))
        Xtr, Xte = sc.transform(imp.transform(X[tr])), sc.transform(imp.transform(X[te]))
        pred[te] = KNeighborsClassifier(n_neighbors=1, algorithm="brute").fit(Xtr, codes[tr]).predict(Xte)
    return balanced_accuracy_score(codes, pred)


def profile_numeric_task(X_df, y):
    X = X_df.to_numpy(dtype=np.float64)
    codes = y.cat.codes.to_numpy().astype(np.int64)
    std = np.nanstd(X, axis=0)
    nunique = X_df.nunique(dropna=True).to_numpy()
    row = {
        "n_rows": X.shape[0],
        "n_cols": X.shape[1],
        "x_sha256": array_sha256(X),
        "y_sha256": array_sha256(codes),
        "n_constant_cols": int(np.sum((std == 0) | np.isnan(std))),
        "median_distinct_per_feature": float(np.median(nunique)),
        "median_distinct_fraction": float(np.median(nunique) / X.shape[0]),
        "n_integer_valued_cols": int(np.sum([np.all(np.nan_to_num(X[:, j]) == np.round(np.nan_to_num(X[:, j]))) for j in range(X.shape[1])])),
        "label_order_ratio": label_order_ratio(codes),
        "id_like_cols": ";".join(str(X_df.columns[j]) for j in id_like_columns(X)),
    }
    row.update(duplicate_summary(X, codes))
    row["one_nn_balanced_accuracy"] = one_nn_balanced_accuracy(X, codes)
    return row
