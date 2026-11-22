import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_embeddings import load_split
from .distances import METRICS
from .evaluation import K_GRID, N_INNER, N_REPEATS, N_SPLITS, SEED, outer_splits, run_outer_fold
from .preprocessing import EMBEDDING

EMBEDDING_WEIGHTINGS = ("uniform", "distance", "softmax")
IMAGE_OFFICIAL = ("cifar10", "cifar100", "stl10", "pets", "flowers102", "dtd", "eurosat")
TEXT_OFFICIAL = ("ag_news", "dbpedia14", "banking77", "newsgroups20")
FACES = ("lfw", "olivetti")
IMAGE_REPRESENTATIONS = ("raw", "resnet50", "dinov2_s14", "dinov2_b14", "clip_b32")
TEXT_REPRESENTATIONS = ("minilm_l6", "mpnet_base", "e5_base")
CLIP = "clip_b32"


def representations_for(dataset):
    if dataset in TEXT_OFFICIAL:
        return TEXT_REPRESENTATIONS
    if dataset in FACES:
        return IMAGE_REPRESENTATIONS + ("facenet",)
    return IMAGE_REPRESENTATIONS


def grid_config():
    return {"metrics": list(METRICS), "preprocessings": list(EMBEDDING), "k_grid": list(K_GRID),
            "weightings": list(EMBEDDING_WEIGHTINGS), "softmax_temperature": 0.07,
            "outer_faces": [N_REPEATS, N_SPLITS], "inner": N_INNER, "seed": SEED,
            "tie_break": "index", "vote_tie": "lowest", "distance_rounding_bits": 40}


def find_roots(paths):
    """Directories that hold a feature_manifest.csv, searched recursively below each given path."""
    roots = []
    for p in paths:
        p = Path(p)
        if (p / "feature_manifest.csv").exists():
            roots.append(p)
        else:
            roots += sorted(m.parent for m in p.rglob("feature_manifest.csv"))
    return roots


def build_index(roots, clip_roots=()):
    """Map (dataset, representation) to its feature directory.

    CLIP features come only from `clip_roots`: the first extraction used a mismatched activation
    function and is superseded, so a CLIP folder found in an ordinary root is ignored."""
    index = {}

    def add(root, only=None, skip=None):
        for meta in sorted(Path(root).glob("*/*/*_meta.json")):
            d = meta.parent
            key = (d.parent.name, d.name)
            if (only and d.name != only) or (skip and d.name == skip):
                continue
            if key in index and index[key] != d:
                raise ValueError(f"{key} found in two places: {index[key]} and {d}")
            index[key] = d

    for r in roots:
        add(r, skip=CLIP)
    for r in clip_roots:
        add(r, only=CLIP)
    return index


def split_names(dataset):
    return ("all",) if dataset in FACES else ("train", "test")


def feature_shapes(index, dataset, representation):
    d = index[(dataset, representation)]
    metas = {s: json.loads((d / f"{s}_meta.json").read_text(encoding="utf-8")) for s in split_names(dataset)
             if (d / f"{s}_meta.json").exists()}
    return metas


def make_units(index, datasets=None, representations=None, preprocessings=EMBEDDING):
    """Units ordered so that cheap datasets finish first; every unit is one (dataset, representation,
    preprocessing) for official splits and one outer fold of those for the two face datasets."""
    units, skipped = [], []
    for (ds, rep), d in sorted(index.items()):
        if datasets and ds not in datasets:
            continue
        if representations and rep not in representations:
            continue
        if ds not in IMAGE_OFFICIAL + TEXT_OFFICIAL + FACES or rep not in representations_for(ds):
            skipped.append((ds, rep, "not part of the design"))
            continue
        metas = feature_shapes(index, ds, rep)
        if set(metas) != set(split_names(ds)):
            skipped.append((ds, rep, f"missing splits, found {sorted(metas)}"))
            continue
        if ds in FACES:
            n = metas["all"]["n"]
            cost = n * n * 0.8 * metas["all"]["dim"]
            folds = [(r, f) for r in range(N_REPEATS) for f in range(N_SPLITS)]
        else:
            ntr, nte, dim = metas["train"]["n"], metas["test"]["n"], metas["train"]["dim"]
            cost = ntr * (nte + (N_INNER - 1) / N_INNER * ntr * 1.0) * dim
            folds = [(0, 0)]
        for prep in preprocessings:
            for r, f in folds:
                units.append({"dataset": ds, "representation": rep, "preprocessing": prep, "repeat": r, "fold": f,
                              "cost": float(cost), "dir": str(d)})
    dataset_cost = {}
    for u in units:
        dataset_cost[u["dataset"]] = dataset_cost.get(u["dataset"], 0.0) + u["cost"]
    units.sort(key=lambda u: (dataset_cost[u["dataset"]], u["dataset"], u["representation"], u["preprocessing"],
                              u["repeat"], u["fold"]))
    return units, skipped


def unit_path(out_dir, unit):
    stem = unit["preprocessing"]
    if unit["dataset"] in FACES:
        stem += f"_r{unit['repeat']}_f{unit['fold']}"
    return Path(out_dir) / unit["dataset"] / unit["representation"] / f"{stem}.parquet"


def _load(unit):
    d = Path(unit["dir"])
    root, rep, ds = d.parent.parent, d.name, d.parent.name
    if ds in FACES:
        X, y, _ = load_split(root, ds, rep, "all")
        return X, y
    Xtr, ytr, _ = load_split(root, ds, rep, "train")
    Xte, yte, _ = load_split(root, ds, rep, "test")
    return np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte]), len(ytr)


def compute_unit(unit, chunk_bytes, backend="numpy", device="cpu", inner=True, search_stats=None):
    """Returns the score table, the outer-test predictions (official splits only) and per-metric seconds.
    With the certified backend, `search_stats` receives per-metric counts of queries settled at each stage."""
    prep = unit["preprocessing"]
    timings, store = {}, {}
    if unit["dataset"] in FACES:
        X, y = _load(unit)
        X = X.astype(np.float64)
        n_classes = int(y.max()) + 1
        for r, f, tr, te in outer_splits(y):
            if (r, f) == (unit["repeat"], unit["fold"]):
                break
        df = run_outer_fold(X, y, n_classes, r, f, tr, te, preprocessings=(prep,), chunk_bytes=chunk_bytes,
                            backend=backend, device=device, weightings=EMBEDDING_WEIGHTINGS, timings=timings,
                            inner=inner, search_stats=search_stats)
        return df, None, timings
    X, y, ntr = _load(unit)
    X = X.astype(np.float64)
    n_classes = int(y.max()) + 1
    tr, te = np.arange(ntr), np.arange(ntr, len(y))
    df = run_outer_fold(X, y, n_classes, 0, 0, tr, te, preprocessings=(prep,), chunk_bytes=chunk_bytes,
                        backend=backend, device=device, weightings=EMBEDDING_WEIGHTINGS, predictions=store,
                        timings=timings, inner=inner, search_stats=search_stats)
    dtype = np.uint8 if n_classes <= 256 else np.uint16
    pred = np.zeros((len(METRICS), len(K_GRID), len(EMBEDDING_WEIGHTINGS), len(te)), dtype=dtype)
    for (_, metric, K, w), p in store.items():
        pred[list(METRICS).index(metric), K_GRID.index(K), EMBEDDING_WEIGHTINGS.index(w)] = p
    return df, {"pred": pred, "y_test": y[te].astype(dtype)}, timings


def write_unit(out_path, df, predictions, timings, extra, search_stats=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")
    if predictions is not None:
        tmp = Path(str(stem) + "_predictions.tmp.npz")
        np.savez_compressed(tmp, metrics=np.array(list(METRICS)), ks=np.array(K_GRID),
                            weightings=np.array(EMBEDDING_WEIGHTINGS), **predictions)
        tmp.replace(Path(str(stem) + "_predictions.npz"))
    for k, v in extra.items():
        df[k] = v
    Path(str(stem) + "_timing.json").write_text(json.dumps({k: round(v, 1) for k, v in timings.items()}) + "\n",
                                                  encoding="utf-8")
    search_stats = {k: v for k, v in (search_stats or {}).items() if v}
    if search_stats:
        Path(str(stem) + "_search.json").write_text(json.dumps(search_stats, sort_keys=True) + "\n", encoding="utf-8")
    tmp = out_path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)


def fallback_share(search_stats):
    """Share of queries that needed a full float64 search, over all metrics of a unit."""
    q = sum(v.get("queries", 0) for v in search_stats.values())
    return sum(v.get("full_float64", 0) for v in search_stats.values()) / q if q else 0.0


def merge_scores(out_dir):
    frames = [pd.read_parquet(p) for p in sorted(Path(out_dir).rglob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
