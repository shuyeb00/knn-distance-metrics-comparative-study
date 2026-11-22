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


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def grid_config():
    from knnmetrics.distances import METRICS
    from knnmetrics.evaluation import K_GRID, N_INNER, N_REPEATS, N_SPLITS, SEED, TABULAR_WEIGHTINGS
    from knnmetrics.preprocessing import TABULAR
    return {"metrics": list(METRICS), "preprocessings": list(TABULAR), "k_grid": list(K_GRID),
            "weightings": list(TABULAR_WEIGHTINGS), "outer": [N_REPEATS, N_SPLITS], "inner": N_INNER,
            "seed": SEED, "tie_break": "index", "vote_tie": "lowest",
            "distance_rounding_bits": 40, "angular_exact_on_dyadic_data": True}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def unit_path(out_dir, task_id, repeat, fold, prep):
    return Path(out_dir) / f"task_{task_id}" / f"repeat{repeat}_fold{fold}_{prep}.parquet"


_DEVICE = {"name": "cpu"}


def init_worker(device_queue):
    if device_queue is not None:
        _DEVICE["name"] = device_queue.get()


def run_unit(task_id, repeat, fold, prep, prepared_dir, out_dir, chash, chunk_bytes, backend="numpy"):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    import numpy as np
    from knnmetrics.data_tabular import prepare_task
    from knnmetrics.evaluation import outer_splits, run_outer_fold

    out = unit_path(out_dir, task_id, repeat, fold, prep)
    if out.exists():
        return task_id, repeat, fold, prep, 0.0, "skipped"
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    t0 = time.perf_counter()
    X, y, info = prepare_task(task_id, prepared_dir)
    n_classes = int(y.max()) + 1
    for r, f, tr, te in outer_splits(y):
        if (r, f) == (repeat, fold):
            break
    df = run_outer_fold(X, y, n_classes, repeat, fold, tr, te, preprocessings=(prep,), chunk_bytes=chunk_bytes,
                        backend=backend, device=_DEVICE["name"])
    df["backend"] = backend
    df.insert(0, "task_id", task_id)
    df["config_hash"] = chash
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return task_id, repeat, fold, prep, time.perf_counter() - t0, "done"


def main():
    p = argparse.ArgumentParser(description="Tabular main grid: every (preprocessing, metric, K, weighting) cell.")
    p.add_argument("--tasks", type=int, nargs="*")
    p.add_argument("--max-dim", type=int, default=None)
    p.add_argument("--min-dim", type=int, default=None)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--prepared", default=str(REPO / "cache" / "prepared"))
    p.add_argument("--out", default=str(REPO / "results" / "tabular" / "grid"))
    p.add_argument("--openml-cache", default=str(REPO / "cache" / "openml"))
    p.add_argument("--chunk-mb", type=int, default=256)
    p.add_argument("--backend", choices=["numpy", "torch", "torch_certified"], default="numpy")
    p.add_argument("--devices", nargs="*", default=["cuda:0"], help="torch backend: one worker per device")
    p.add_argument("--max-hours", type=float, default=None, help="stop starting new units after this many hours")
    p.add_argument("--seed-from", nargs="*", default=[], help="directories with unit files from earlier sessions")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()

    import yaml
    from knnmetrics.data_tabular import configure_openml_cache, prepare_task
    from knnmetrics.evaluation import N_REPEATS, N_SPLITS
    from knnmetrics.preprocessing import TABULAR

    configure_openml_cache(args.openml_cache)
    bench = yaml.safe_load(open(REPO / "configs" / "benchmark_tabular.yaml", encoding="utf-8"))
    tasks = [t for t in bench["primary"]
             if (not args.tasks or t["task_id"] in args.tasks)
             and (args.max_dim is None or t["n_features"] <= args.max_dim)
             and (args.min_dim is None or t["n_features"] >= args.min_dim)]
    cfg = grid_config()
    chash = config_hash(cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "grid_config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text())["config_hash"] != chash:
        sys.exit(f"{out_dir} holds results from a different configuration; use a new --out or remove it")
    cfg_path.write_text(json.dumps({**cfg, "config_hash": chash}, indent=2) + "\n")

    print(f"{len(tasks)} tasks, config {chash}, commit {git_commit()[:10]}", flush=True)
    summaries = []
    for t in tasks:
        X, y, info = prepare_task(t["task_id"], args.prepared)
        if t.get("x_sha256") and info["x_sha256_raw"] != t["x_sha256"]:
            sys.exit(f"checksum mismatch for task {t['task_id']} ({t['name']}); data differ from the frozen list")
        summaries.append(info)
        print(f"  prepared {t['task_id']:>6} {info['name'][:28]:<28} rows {info['n_rows_original']:>6} -> "
              f"{info['n_rows_final']:>6} (dup -{info['n_rows_duplicate_removed']}, "
              f"conflict -{info['n_rows_conflicting_removed']}) D={info['n_features']} C={info['n_classes']}", flush=True)
    import pandas as pd
    summary_path = out_dir.parent / "prepared_task_summary.csv"
    old = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    merged = pd.concat([old, pd.DataFrame(summaries)]).drop_duplicates("task_id", keep="last").sort_values("task_id")
    merged.to_csv(summary_path, index=False)
    if args.prepare_only:
        return

    cost = {s["task_id"]: s["n_rows_final"] ** 2 * s["n_features"] for s in summaries}
    units = [(t["task_id"], r, f, pr) for t in tasks for r in range(N_REPEATS) for f in range(N_SPLITS)
             for pr in TABULAR]
    units.sort(key=lambda u: -cost[u[0]])
    todo = [u for u in units if not unit_path(out_dir, *u).exists()]
    for src in args.seed_from:
        import shutil
        for f in Path(src).rglob("repeat*_fold*_*.parquet"):
            dst = out_dir / f.parent.name / f.name
            if f.parent.name.startswith("task_") and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
    todo = [u for u in units if not unit_path(out_dir, *u).exists()]

    import multiprocessing as mp
    if args.backend.startswith("torch"):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        for d in args.devices:
            queue.put(d)
        n_workers = len(args.devices)
    else:
        ctx, queue, n_workers = mp.get_context(), None, args.jobs
    print(f"{len(units)} units, {len(todo)} to run on {n_workers} {args.backend} workers", flush=True)
    t0 = time.perf_counter()
    deadline = t0 + args.max_hours * 3600 if args.max_hours else None
    done, pending, it, owner, failed = 0, set(), iter(todo), {}, []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=init_worker,
                             initargs=(queue,)) as ex:
        def submit_next():
            if deadline and time.perf_counter() > deadline:
                return False
            u = next(it, None)
            if u is None:
                return False
            fut = ex.submit(run_unit, *u, args.prepared, str(out_dir), chash, args.chunk_mb * 2**20, args.backend)
            owner[fut] = u
            pending.add(fut)
            return True
        for _ in range(n_workers * 2):
            submit_next()
        while pending:
            fut = next(as_completed(pending))
            pending.discard(fut)
            done += 1
            el = time.perf_counter() - t0
            try:
                tid, r, f, pr, sec, status = fut.result()
            except Exception as e:
                failed.append((owner[fut], f"{type(e).__name__}: {e}"))
                print(f"[{done}/{len(todo)}] FAILED {owner[fut]} {type(e).__name__}: {e}", flush=True)
                submit_next()
                continue
            print(f"[{done}/{len(todo)}] task {tid} r{r} f{f} {pr} {status} {sec:.0f}s | elapsed {el / 60:.1f} min, "
                  f"eta {el / done * (len(todo) - done) / 60:.0f} min", flush=True)
            submit_next()
    left = sum(not unit_path(out_dir, *u).exists() for u in units)
    print(f"finished: {len(units) - left} of {len(units)} units present, {left} left", flush=True)
    if failed:
        print(f"{len(failed)} units failed:", flush=True)
        for u, msg in failed:
            print("  ", u, msg, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
