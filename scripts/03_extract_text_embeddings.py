import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from knnmetrics.data_embeddings import save_split, write_manifest

SEED = 20261121
MODELS = {
    "minilm_l6": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "mpnet_base": ("sentence-transformers/all-mpnet-base-v2", ""),
    "e5_base": ("intfloat/e5-base-v2", "query: "),
}
DATASETS = ("ag_news", "dbpedia14", "banking77", "newsgroups20")


def stratified_subsample(y, n, seed):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    if n >= len(y):
        return idx
    keep, _ = train_test_split(idx, train_size=n, stratify=y, random_state=seed)
    return np.sort(keep)


def load_text_dataset(name, cache):
    """Returns {split: (texts, labels, original_indices)} with the official splits."""
    from datasets import load_dataset
    out = {}
    if name == "ag_news":
        ds = load_dataset("fancyzhx/ag_news", cache_dir=cache)
        for split in ("train", "test"):
            out[split] = (list(ds[split]["text"]), np.array(ds[split]["label"]), np.arange(len(ds[split])))
    elif name == "dbpedia14":
        ds = load_dataset("fancyzhx/dbpedia_14", cache_dir=cache)
        for split, n in (("train", 50_000), ("test", 10_000)):
            y = np.array(ds[split]["label"])
            keep = stratified_subsample(y, n, SEED)
            titles, contents = ds[split]["title"], ds[split]["content"]
            texts = [f"{titles[i]}. {contents[i]}".strip() for i in keep]
            out[split] = (texts, y[keep], keep)
    elif name == "banking77":
        ds = load_dataset("mteb/banking77", cache_dir=cache)
        for split in ("train", "test"):
            out[split] = (list(ds[split]["text"]), np.array(ds[split]["label"]), np.arange(len(ds[split])))
    elif name == "newsgroups20":
        from sklearn.datasets import fetch_20newsgroups
        for split in ("train", "test"):
            b = fetch_20newsgroups(subset=split, remove=("headers", "footers", "quotes"),
                                   data_home=cache, shuffle=False)
            out[split] = (list(b.data), np.asarray(b.target), np.arange(len(b.target)))
    else:
        raise ValueError(name)
    return out


def load_unnormalised_model(model_id, device):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Normalize
    model = SentenceTransformer(model_id, device=device)
    # The packaged pipelines end in an L2 normalisation layer; it is removed so that vector norms are
    # kept, since norm information is one of the studied factors. Normalising is a preprocessing level.
    kept = [m for m in model if not isinstance(m, Normalize)]
    removed = len(model) - len(kept)
    model = SentenceTransformer(modules=kept, device=device)
    return model, removed


def main():
    p = argparse.ArgumentParser(description="Extract sentence embeddings for the text benchmark.")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default="/tmp/knnmetrics_text_cache")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="debug only: first n documents per split")
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    env = {"torch": torch.__version__, "python": platform.python_version(),
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}

    for dname in args.datasets:
        t0 = time.perf_counter()
        splits = load_text_dataset(dname, args.cache)
        if args.limit:
            splits = {s: (t[:args.limit], y[:args.limit], i[:args.limit]) for s, (t, y, i) in splits.items()}
        print(f"{dname}: " + ", ".join(f"{s}={len(v[0])}" for s, v in splits.items())
              + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
        for s, (texts, y, idx) in splits.items():
            idx_path = out / dname / f"{s}_source_indices.npy"
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(idx_path, np.asarray(idx, dtype=np.int64))
            n_empty = sum(1 for t in texts if not t.strip())
            if n_empty:
                print(f"  {s}: {n_empty} empty documents kept as empty strings", flush=True)

        for mname in args.models:
            model_id, prefix = MODELS[mname]
            if (out / dname / mname / "test_meta.json").exists():
                print(f"  {mname}: already done", flush=True)
                continue
            model, removed = load_unnormalised_model(model_id, device)
            if device == "cuda":
                model.half()
            for s, (texts, y, _) in splits.items():
                t0 = time.perf_counter()
                X = model.encode([prefix + t for t in texts], batch_size=args.batch_size,
                                 convert_to_numpy=True, normalize_embeddings=False,
                                 show_progress_bar=False).astype(np.float16)
                meta = {"model_id": model_id, "prefix": prefix, "max_seq_length": model.max_seq_length,
                        "normalize_layers_removed": removed, "seconds": round(time.perf_counter() - t0, 1),
                        **env}
                rec = save_split(out, dname, mname, s, X, y, meta)
                norms = np.linalg.norm(X.astype(np.float32), axis=1)
                print(f"  {mname} {s}: {X.shape} in {meta['seconds']}s, norm mean {norms.mean():.3f} "
                      f"cv {norms.std() / norms.mean():.3f}", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    n = write_manifest(out)
    print(f"manifest: {n} feature files", flush=True)
    (out / "text_environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
