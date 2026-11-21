import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype}|{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def feature_dir(root, dataset, representation):
    return Path(root) / dataset / representation


def save_split(root, dataset, representation, split, X, y, meta=None):
    out = feature_dir(root, dataset, representation)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{split}_X.npy", X)
    np.save(out / f"{split}_y.npy", np.asarray(y, dtype=np.int64))
    record = {
        "dataset": dataset, "representation": representation, "split": split,
        "n": int(X.shape[0]), "dim": int(np.prod(X.shape[1:])), "dtype": str(X.dtype),
        "x_sha256": sha256_array(X), "y_sha256": sha256_array(np.asarray(y, dtype=np.int64)),
    }
    if meta:
        record.update(meta)
    (out / f"{split}_meta.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def load_split(root, dataset, representation, split, verify=True):
    d = feature_dir(root, dataset, representation)
    X = np.load(d / f"{split}_X.npy")
    y = np.load(d / f"{split}_y.npy")
    meta = json.loads((d / f"{split}_meta.json").read_text(encoding="utf-8"))
    if verify and (sha256_array(X) != meta["x_sha256"] or sha256_array(y) != meta["y_sha256"]):
        raise ValueError(f"checksum mismatch for {d}/{split}")
    return X, y, meta


def write_manifest(root):
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(root).rglob("*_meta.json"))]
    keys = ["dataset", "representation", "split", "n", "dim", "dtype", "x_sha256", "y_sha256"]
    lines = [",".join(keys)] + [",".join(str(r.get(k, "")) for k in keys) for r in rows]
    (Path(root) / "feature_manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)
