import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_DEVICE = {"name": "cpu"}


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def init_worker(device_queue):
    if device_queue is not None:
        _DEVICE["name"] = device_queue.get()


def run_unit(unit, out_dir, chash, chunk_bytes, backend):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "2")
    from knnmetrics.embedding_grid import compute_unit, unit_path, write_unit
    out = unit_path(out_dir, unit)
    if out.exists():
        return unit, 0.0, "skipped", {}
    t0 = time.perf_counter()
    df, predictions, timings = compute_unit(unit, chunk_bytes, backend, _DEVICE["name"])
    protocol = "repeated_cv" if predictions is None else "official_split"
    write_unit(out, df, predictions, timings,
               {"dataset": unit["dataset"], "representation": unit["representation"], "protocol": protocol,
                "backend": backend, "config_hash": chash})
    return unit, time.perf_counter() - t0, "done", timings


def main():
    p = argparse.ArgumentParser(description="Embedding main grid: every (preprocessing, metric, K, weighting) cell.")
    p.add_argument("--features", nargs="+", required=True, help="directories searched for feature_manifest.csv")
    p.add_argument("--clip-features", nargs="*", default=[], help="directories holding the corrected CLIP features")
    p.add_argument("--out", default=str(REPO / "results" / "embeddings" / "grid"))
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--representations", nargs="*")
    p.add_argument("--preprocessings", nargs="*")
    p.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    p.add_argument("--devices", nargs="*", default=["cuda:0"], help="torch backend: one worker per device")
    p.add_argument("--jobs", type=int, default=2, help="numpy backend: worker processes")
    p.add_argument("--chunk-mb", type=int, default=512)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--estimate", action="store_true", help="list the units and exit")
    args = p.parse_args()

    from knnmetrics.embedding_grid import build_index, find_roots, grid_config, make_units, unit_path
    from knnmetrics.preprocessing import EMBEDDING

    roots, clip_roots = find_roots(args.features), find_roots(args.clip_features)
    index = build_index(roots, clip_roots)
    preps = tuple(args.preprocessings) if args.preprocessings else EMBEDDING
    units, skipped = make_units(index, args.datasets, args.representations, preps)
    for s in skipped:
        print("skipped", *s, flush=True)

    cfg = grid_config()
    chash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "grid_config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text())["config_hash"] != chash:
        sys.exit(f"{out_dir} holds results from a different configuration; use a new --out or remove it")
    cfg_path.write_text(json.dumps({**cfg, "config_hash": chash}, indent=2) + "\n")

    todo = [u for u in units if not unit_path(out_dir, u).exists()]
    print(f"{len(roots)} feature roots, {len(clip_roots)} clip roots, {len(index)} (dataset, representation) pairs, "
          f"config {chash}, commit {git_commit()[:10]}", flush=True)
    print(f"{len(units)} units, {len(todo)} to run", flush=True)
    if args.estimate:
        by = {}
        for u in todo:
            k = (u["dataset"], u["representation"])
            by[k] = by.get(k, 0) + 1
        for (d, r), n in by.items():
            print(f"  {d:<12} {r:<12} {n} units", flush=True)
        return

    import multiprocessing as mp
    if args.backend == "torch":
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        for d in args.devices:
            queue.put(d)
        n_workers = len(args.devices)
    else:
        ctx, queue, n_workers = mp.get_context(), None, args.jobs
    t0 = time.perf_counter()
    deadline = t0 + args.max_hours * 3600 if args.max_hours else None
    done, pending, it = 0, set(), iter(todo)
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=init_worker,
                             initargs=(queue,)) as ex:
        def submit_next():
            if deadline and time.perf_counter() > deadline:
                return False
            u = next(it, None)
            if u is None:
                return False
            pending.add(ex.submit(run_unit, u, str(out_dir), chash, args.chunk_mb * 2**20, args.backend))
            return True
        for _ in range(n_workers):
            submit_next()
        while pending:
            fut = next(as_completed(pending))
            pending.discard(fut)
            u, sec, status, timings = fut.result()
            done += 1
            el = time.perf_counter() - t0
            per_metric = " ".join(f"{m}={s:.0f}" for m, s in timings.items())
            print(f"[{done}/{len(todo)}] {u['dataset']} {u['representation']} {u['preprocessing']} "
                  f"r{u['repeat']}f{u['fold']} {status} {sec:.0f}s | {per_metric} | elapsed {el / 60:.1f} min",
                  flush=True)
            submit_next()
    left = sum(not unit_path(out_dir, u).exists() for u in units)
    print(f"finished: {len(units) - left} of {len(units)} units present, {left} left", flush=True)


if __name__ == "__main__":
    main()
