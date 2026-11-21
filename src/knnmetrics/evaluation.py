import time

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

from .distances import METRICS
from .neighbour_engine import kneighbors
from .preprocessing import TABULAR, FoldPreprocessor
from .voting import predict_from_neighbours

SEED = 20261121
K_GRID = (1, 3, 5, 7, 11, 15, 21, 31)
TABULAR_WEIGHTINGS = ("uniform", "distance")
N_REPEATS, N_SPLITS, N_INNER = 3, 5, 5


def deduplicate(X, y):
    """Keep the first copy of each group of identical rows; drop every copy of groups with conflicting labels.
    Returns kept row indices in their original order and a summary."""
    df = pd.DataFrame(X)
    key = pd.util.hash_pandas_object(df, index=False).to_numpy()
    frame = pd.DataFrame({"key": key, "y": y, "row": np.arange(len(y))})
    n_labels = frame.groupby("key")["y"].transform("nunique").to_numpy()
    conflicting = n_labels > 1
    first = ~frame.duplicated("key", keep="first").to_numpy()
    keep = np.flatnonzero(first & ~conflicting)
    # Hash collisions are possible in principle, so groups are confirmed on the actual rows.
    kept = df.iloc[keep]
    if kept.duplicated().any():
        raise RuntimeError("hash collision in deduplication")
    return keep, {
        "n_rows_original": int(len(y)),
        "n_rows_conflicting_removed": int(conflicting.sum()),
        "n_rows_duplicate_removed": int((~first & ~conflicting).sum()),
        "n_rows_after_dedup": int(len(keep)),
    }


def stratified_cap(y, cap, seed=SEED):
    idx = np.arange(len(y))
    if len(y) <= cap:
        return idx
    keep, _ = train_test_split(idx, train_size=cap, stratify=y, random_state=seed)
    return np.sort(keep)


def outer_splits(y):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    for i, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        yield i // N_SPLITS, i % N_SPLITS, np.sort(tr), np.sort(te)


def inner_splits(y, repeat, fold):
    cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED + 100 * repeat + fold)
    for tr, va in cv.split(np.zeros(len(y)), y):
        yield np.sort(tr), np.sort(va)


def scores_from_predictions(y_true, y_pred, n_classes):
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    present = support > 0
    recall = tp[present] / support[present]
    used = present | (predicted > 0)
    denom = support + predicted
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)[used]
    n = cm.sum()
    po = tp.sum() / n
    pe = (support * predicted).sum() / n ** 2
    return {
        "balanced_accuracy": float(recall.mean()),
        "accuracy": float(po),
        "macro_f1": float(f1.mean()),
        "kappa": float((po - pe) / (1 - pe)) if pe < 1 else 0.0,
    }


def evaluate_neighbours(res, y_train, y_test, n_classes, ks=K_GRID, weightings=TABULAR_WEIGHTINGS,
                        predictions=None):
    rows = []
    labels = y_train[res.indices]
    for K in ks:
        if K > res.indices.shape[1]:
            continue
        ties = res.ties_at.get(K) if res.ties_at else None
        for w in weightings:
            pred, vote_tied = predict_from_neighbours(labels, res.distances, K, n_classes, w)
            if predictions is not None:
                predictions[(K, w)] = pred
            row = {"K": K, "weighting": w, **scores_from_predictions(y_test, pred, n_classes),
                   "vote_tie_rate": float(vote_tied.mean())}
            if ties is not None:
                row["tie_mass_mean"] = float((ties / K).mean())
                row["boundary_tie_rate"] = float((ties > 1).mean())
            rows.append(row)
    return rows


def run_outer_fold(X, y, n_classes, repeat, fold, tr, te, metrics=METRICS, preprocessings=TABULAR,
                   inner=True, kmax=max(K_GRID), chunk_bytes=2**28, backend="numpy", device="cpu",
                   weightings=TABULAR_WEIGHTINGS, predictions=None, timings=None):
    """All grid cells for one outer fold: outer test scores and inner-CV validation scores.
    If `predictions` is a dict it receives the outer-test predictions keyed by (preprocessing, metric, K, weighting);
    `timings` receives the seconds spent per metric."""
    rows = []
    for prep in preprocessings:
        pp = FoldPreprocessor(prep, seed=SEED).fit(X[tr])
        Xtr, Xte = pp.transform(X[tr]), pp.transform(X[te])
        inner_sets = []
        if inner:
            for itr, iva in inner_splits(y[tr], repeat, fold):
                ipp = FoldPreprocessor(prep, seed=SEED).fit(X[tr][itr])
                inner_sets.append((itr, iva, ipp.transform(X[tr][itr]), ipp.transform(X[tr][iva])))
        for metric in metrics:
            t0 = time.perf_counter()
            k = min(kmax, len(tr))
            res = kneighbors(Xtr, Xte, k, metric, tie_ks=tuple(K for K in K_GRID if K <= k),
                             chunk_bytes=chunk_bytes, backend=backend, device=device)
            store = {} if predictions is not None else None
            for r in evaluate_neighbours(res, y[tr], y[te], n_classes, weightings=weightings, predictions=store):
                rows.append({"split": "outer_test", "inner_fold": -1, "preprocessing": prep, "metric": metric, **r})
            if predictions is not None:
                predictions.update({(prep, metric, K, w): p for (K, w), p in store.items()})
            for j, (itr, iva, Xi, Xv) in enumerate(inner_sets):
                ki = min(kmax, len(itr))
                ires = kneighbors(Xi, Xv, ki, metric, chunk_bytes=chunk_bytes, backend=backend, device=device)
                for r in evaluate_neighbours(ires, y[tr][itr], y[tr][iva], n_classes, weightings=weightings):
                    rows.append({"split": "inner_validation", "inner_fold": j, "preprocessing": prep,
                                 "metric": metric, **r})
            if timings is not None:
                timings[metric] = timings.get(metric, 0.0) + time.perf_counter() - t0
    out = pd.DataFrame(rows)
    keys = ["split", "preprocessing", "metric", "K", "weighting"]
    inner_mean = (out[out["split"] == "inner_validation"].groupby(keys, sort=False)
                  .mean(numeric_only=True).reset_index().assign(inner_fold=-1))
    out = pd.concat([out[out["split"] == "outer_test"], inner_mean], ignore_index=True)
    out.insert(0, "repeat", repeat)
    out.insert(1, "fold", fold)
    return out
