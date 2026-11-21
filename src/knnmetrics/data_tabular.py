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
