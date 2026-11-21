from pathlib import Path

import openml

CC18_SUITE_ID = 99
MAX_ROWS = 20_000


def configure_openml_cache(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(root))
    return root


def suite_task_ids(suite_id=CC18_SUITE_ID):
    return sorted(int(t) for t in openml.study.get_suite(suite_id).tasks)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def task_metadata(task_id):
    task = openml.tasks.get_task(
        task_id, download_splits=False, download_data=False,
        download_qualities=False, download_features_meta_data=False,
    )
    ds = openml.datasets.get_dataset(
        task.dataset_id, download_data=False,
        download_qualities=True, download_features_meta_data=True,
    )
    target = task.target_name
    # Row identifiers and ignore-flagged columns are dropped, as OpenML marks them unusable as predictors.
    excluded = set(_as_list(ds.row_id_attribute)) | set(_as_list(ds.ignore_attribute)) | {target}
    used = [f for f in ds.features.values() if f.name not in excluded]
    by_type = {t: sum(f.data_type == t for f in used) for t in ("numeric", "nominal", "string", "date")}
    target_feature = next(f for f in ds.features.values() if f.name == target)

    q = ds.qualities or {}
    n = int(q.get("NumberOfInstances", float("nan")))
    missing_cells = int(sum(f.number_missing_values for f in used))
    majority = q.get("MajorityClassSize")
    minority = q.get("MinorityClassSize")

    return {
        "task_id": int(task_id),
        "dataset_id": int(task.dataset_id),
        "name": ds.name,
        "version": int(ds.version),
        "target": target,
        "n_instances": n,
        "n_features": len(used),
        "n_numeric": by_type["numeric"],
        "n_nominal": by_type["nominal"],
        "n_string": by_type["string"],
        "n_date": by_type["date"],
        "n_classes": len(target_feature.nominal_values or []),
        "majority_class_size": int(majority) if majority is not None else None,
        "minority_class_size": int(minority) if minority is not None else None,
        "imbalance_ratio": (majority / minority) if majority and minority else None,
        "missing_cells": missing_cells,
        "missing_rate": missing_cells / (n * len(used)) if n and used else None,
        "n_instances_with_missing": int(q.get("NumberOfInstancesWithMissingValues", 0) or 0),
        "row_id_attribute": ";".join(_as_list(ds.row_id_attribute)),
        "ignore_attributes": ";".join(_as_list(ds.ignore_attribute)),
        "all_numeric": by_type["numeric"] == len(used) and len(used) > 0,
        "exceeds_row_cap": n > MAX_ROWS,
        "dense_float64_mb": n * len(used) * 8 / 2**20,
    }


def load_task(task_id):
    task = openml.tasks.get_task(
        task_id, download_splits=False, download_data=False,
        download_qualities=False, download_features_meta_data=False,
    )
    ds = openml.datasets.get_dataset(
        task.dataset_id, download_data=True,
        download_qualities=False, download_features_meta_data=True,
    )
    X, y, _, names = ds.get_data(
        target=task.target_name, include_row_id=False, include_ignore_attribute=False,
    )
    y = y.astype("category")
    return X, y, task, ds


def array_sha256(a):
    import hashlib
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def prepare_task(task_id, cache_dir, max_rows=MAX_ROWS):
    """Load, deduplicate and cap one task; cached as float64 arrays plus the kept source-row indices."""
    import json
    import numpy as np
    from .evaluation import SEED, deduplicate, stratified_cap

    cache_dir = Path(cache_dir)
    path = cache_dir / f"task_{task_id}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return z["X"], z["y"], json.loads(str(z["info"]))
    X_df, y, task, ds = load_task(task_id)
    X = X_df.to_numpy(dtype=np.float64)
    codes = y.cat.codes.to_numpy().astype(np.int64)
    info = {"task_id": int(task_id), "name": ds.name, "x_sha256_raw": array_sha256(X),
            "y_sha256_raw": array_sha256(codes), "n_classes_total": int(len(y.cat.categories))}
    keep, summary = deduplicate(X, codes)
    info.update(summary)
    cap = stratified_cap(codes[keep], max_rows, SEED)
    rows = keep[cap]
    info["n_rows_final"] = int(len(rows))
    info["subsampled"] = bool(len(cap) < len(keep))
    X, codes = X[rows], codes[rows]
    present = np.unique(codes)
    remap = np.full(codes.max() + 1, -1)
    remap[present] = np.arange(len(present))
    codes = remap[codes]
    info["n_classes"] = int(len(present))
    info["n_features"] = int(X.shape[1])
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=codes, source_rows=rows, info=json.dumps(info))
    return X, codes, info
