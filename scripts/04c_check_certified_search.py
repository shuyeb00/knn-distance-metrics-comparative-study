import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TIE_KS = (1, 3, 5, 7, 11, 15, 21, 31)


def compare(a, b):
    same = (np.array_equal(a.indices, b.indices) and np.array_equal(a.distances, b.distances)
            and np.array_equal(a.boundary_ties, b.boundary_ties) and np.array_equal(a.below_boundary, b.below_boundary)
            and set(a.ties_at) == set(b.ties_at) and all(np.array_equal(a.ties_at[K], b.ties_at[K]) for K in a.ties_at))
    rows = float((a.indices == b.indices).all(axis=1).mean())
    return same, rows


def synchronize(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def timed(fn, device):
    synchronize(device)
    t0 = time.perf_counter()
    out = fn()
    synchronize(device)
    return out, time.perf_counter() - t0


def run_pair(Xtr, Xq, metric, device, chunk_bytes, k=31):
    from knnmetrics.neighbour_engine import kneighbors
    ref, t_ref = timed(lambda: kneighbors(Xtr, Xq, k, metric, tie_ks=TIE_KS, backend="torch", device=device,
                                          chunk_bytes=chunk_bytes), device)
    stats = {}
    fast, t_fast = timed(lambda: kneighbors(Xtr, Xq, k, metric, tie_ks=TIE_KS, backend="torch_certified",
                                            device=device, chunk_bytes=chunk_bytes, stats=stats), device)
    same, rows = compare(ref, fast)
    return {"identical": same, "row_agreement": rows, "seconds_float64": round(t_ref, 3),
            "seconds_certified": round(t_fast, 3), **stats}


def synthetic_sets(seed=5):
    rng = np.random.default_rng(seed)
    out = {}
    out["gaussian_d256"] = rng.normal(size=(10000, 256))
    out["offset_d64"] = rng.normal(size=(10000, 64)) * 1e-3 + 1e3
    out["pixels_d3072"] = rng.integers(0, 256, size=(6000, 3072)).astype(np.float64)
    out["quantized_d64"] = rng.integers(0, 4, size=(10000, 64)).astype(np.float64)
    base = rng.normal(size=(400, 128))
    out["duplicates_d128"] = base[rng.integers(0, 400, size=10000)]
    c = rng.normal(size=(10000, 96))
    c[::11] = 0.0
    c[3::13] = 1.5
    c[:, 7] = 2.0
    out["constant_rows_d96"] = c
    centres = rng.normal(size=(20, 768))
    out["embedding_like_d768"] = (centres[rng.integers(0, 20, 10000)] + rng.normal(size=(10000, 768)) * 0.8).astype(
        np.float16).astype(np.float64)
    return out


def check_synthetic(device, chunk_bytes, report):
    from knnmetrics.distances import METRICS
    for name, X in synthetic_sets().items():
        n = int(0.8 * len(X))
        for metric in METRICS:
            r = run_pair(X[:n], X[n:], metric, device, chunk_bytes)
            report["synthetic"].append({"set": name, "metric": metric, **r})
            print(f"  {name:<20} {metric:<12} identical={r['identical']} float64 {r['seconds_float64']:.1f}s "
                  f"certified {r['seconds_certified']:.1f}s  stage1={r.get('stage1', '-')} "
                  f"stage2={r.get('stage2', '-')} full={r.get('full_float64', '-')}", flush=True)


def load_official(index, dataset, rep):
    from knnmetrics.data_embeddings import load_split
    d = index[(dataset, rep)]
    root = d.parent.parent
    Xtr, ytr, _ = load_split(root, dataset, rep, "train")
    Xte, yte, _ = load_split(root, dataset, rep, "test")
    return Xtr.astype(np.float64), Xte.astype(np.float64)


def check_real(index, targets, preps, device, chunk_bytes, report, max_queries):
    from knnmetrics.distances import METRICS
    from knnmetrics.preprocessing import FoldPreprocessor
    for target in targets:
        ds, rep = target.split(":")
        if (ds, rep) not in index:
            print(f"  {target}: not found, skipped", flush=True)
            continue
        Xtr, Xte = load_official(index, ds, rep)
        Xte = Xte[:max_queries]
        for prep in preps:
            pp = FoldPreprocessor(prep, seed=0).fit(Xtr)
            A, B = pp.transform(Xtr), pp.transform(Xte)
            for metric in METRICS:
                r = run_pair(A, B, metric, device, chunk_bytes)
                report["real"].append({"dataset": ds, "representation": rep, "preprocessing": prep, "metric": metric,
                                       "n_train": len(A), "n_query": len(B), "dim": A.shape[1], **r})
                print(f"  {ds:<11} {rep:<11} {prep:<10} {metric:<12} identical={r['identical']} "
                      f"float64 {r['seconds_float64']:.1f}s certified {r['seconds_certified']:.1f}s "
                      f"stage1={r.get('stage1', '-')} stage2={r.get('stage2', '-')} full={r.get('full_float64', '-')}",
                      flush=True)


def benchmark(index, targets, device, chunk_bytes, report, n_ref, n_fast):
    from knnmetrics.distances import METRICS
    from knnmetrics.neighbour_engine import kneighbors
    for target in targets:
        ds, rep = target.split(":")
        if (ds, rep) not in index:
            print(f"  {target}: not found, skipped", flush=True)
            continue
        Xtr, Xte = load_official(index, ds, rep)
        n, D = Xtr.shape
        for metric in METRICS:
            q_ref, q_fast = Xte[:n_ref], Xte[:n_fast]
            _, t_ref = timed(lambda: kneighbors(Xtr, q_ref, 31, metric, tie_ks=TIE_KS, backend="torch", device=device,
                                                chunk_bytes=chunk_bytes), device)
            stats = {}
            _, t_fast = timed(lambda: kneighbors(Xtr, q_fast, 31, metric, tie_ks=TIE_KS, backend="torch_certified",
                                                 device=device, chunk_bytes=chunk_bytes, stats=stats), device)
            rate_ref, rate_fast = n * len(q_ref) * D / t_ref, n * len(q_fast) * D / t_fast
            report["benchmark"].append({"dataset": ds, "representation": rep, "metric": metric, "n_train": n, "dim": D,
                                        "queries_float64": len(q_ref), "queries_certified": len(q_fast),
                                        "seconds_float64": round(t_ref, 3), "seconds_certified": round(t_fast, 3),
                                        "rate_float64": rate_ref, "rate_certified": rate_fast, **stats})
            print(f"  {ds:<11} {rep:<11} D={D:<5} {metric:<12} float64 {rate_ref:.2e}/s certified {rate_fast:.2e}/s "
                  f"speed-up {rate_fast / rate_ref:5.1f}x  full={stats.get('full_float64', '-')}/{stats.get('queries', '-')}",
                  flush=True)


def tile_sweep(device, report, n=50000, m=2048, D=768):
    import torch
    from knnmetrics import certified_search as cs
    g = torch.Generator(device="cpu").manual_seed(0)
    T = torch.randn(n, D, generator=g).to(device)
    Q = torch.randn(m, D, generator=g).to(device)
    base = dict(cs.TRITON_TILE)
    for BM, BN, BK, warps in [(32, 32, 8, 4), (64, 64, 4, 4), (64, 64, 8, 8), (32, 64, 8, 4), (64, 32, 8, 4),
                              (16, 64, 16, 4), (64, 64, 16, 8), (128, 64, 4, 8)]:
        cs.TRITON_TILE.update(BM=BM, BN=BN, BK=BK, num_warps=warps)
        try:
            cs.power_sums(Q[:64], T[:512], "l1", kernel="triton")
            _, t = timed(lambda: cs.power_sums(Q, T, "l1", kernel="triton"), device)
            _, t3 = timed(lambda: cs.power_sums(Q, T, "l0.5", kernel="triton"), device)
            rate, rate3 = n * m * D / t, n * m * D / t3
            report["tiles"].append({"BM": BM, "BN": BN, "BK": BK, "num_warps": warps, "rate_l1": rate,
                                    "rate_l0.5": rate3})
            print(f"  tile {BM}x{BN}x{BK} warps {warps}: l1 {rate:.2e}/s, l0.5 {rate3:.2e}/s", flush=True)
        except Exception as e:
            print(f"  tile {BM}x{BN}x{BK} warps {warps}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    cs.TRITON_TILE.update(base)
    _, t = timed(lambda: cs.power_sums(Q, T, "l2"), device)
    report["tiles"].append({"gemm_l2_rate": n * m * D / t})
    print(f"  l2 float32 GEMM: {n * m * D / t:.2e}/s", flush=True)


def main():
    p = argparse.ArgumentParser(description="Check that the certified float32/float64 search equals the float64 "
                                            "engine bit for bit, and measure its speed.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--chunk-mb", type=int, default=1024)
    p.add_argument("--features", nargs="*", default=[])
    p.add_argument("--clip-features", nargs="*", default=[])
    p.add_argument("--real", nargs="*", default=["flowers102:resnet50", "dtd:dinov2_b14", "stl10:raw",
                                                 "pets:clip_b32", "banking77:e5_base"])
    p.add_argument("--real-preprocessings", nargs="*", default=["none", "zscore", "l2"])
    p.add_argument("--real-max-queries", type=int, default=3000)
    p.add_argument("--bench", nargs="*", default=["cifar10:dinov2_s14", "cifar10:dinov2_b14", "cifar10:clip_b32",
                                                  "cifar10:resnet50", "cifar10:raw", "ag_news:mpnet_base"])
    p.add_argument("--bench-queries", type=int, nargs=2, default=[256, 2048], metavar=("FLOAT64", "CERTIFIED"))
    p.add_argument("--skip", nargs="*", default=[], choices=["synthetic", "real", "tiles", "bench"])
    p.add_argument("--out", default="certified_search_check.json")
    args = p.parse_args()

    import torch
    from knnmetrics import certified_search as cs
    report = {"torch": torch.__version__, "python": platform.python_version(), "device": args.device,
              "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu",
              "triton": cs._triton_kernel() is not None, "synthetic": [], "real": [], "tiles": [], "benchmark": []}
    try:
        import triton
        report["triton_version"] = triton.__version__
    except Exception:
        pass
    print(json.dumps({k: report[k] for k in ("torch", "python", "gpu", "triton")}), flush=True)
    chunk = args.chunk_mb * 2 ** 20
    index = {}
    if args.features or args.clip_features:
        from knnmetrics.embedding_grid import build_index, find_roots
        index = build_index(find_roots(args.features), find_roots(args.clip_features))
    if "synthetic" not in args.skip:
        print("synthetic sets", flush=True)
        check_synthetic(args.device, chunk, report)
    if "real" not in args.skip and index:
        print("real features", flush=True)
        check_real(index, args.real, args.real_preprocessings, args.device, chunk, report, args.real_max_queries)
    if "tiles" not in args.skip and report["triton"] and args.device.startswith("cuda"):
        print("triton tiles", flush=True)
        tile_sweep(args.device, report)
    if "bench" not in args.skip and index:
        print("throughput", flush=True)
        benchmark(index, args.bench, args.device, chunk, report, *args.bench_queries)
    checks = report["synthetic"] + report["real"]
    bad = [c for c in checks if not c["identical"]]
    report["all_identical"] = not bad
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"{len(checks)} comparisons, {len(bad)} not identical; report in {args.out}", flush=True)
    for c in bad:
        print("  NOT IDENTICAL:", {k: c[k] for k in c if k in ("set", "dataset", "representation", "preprocessing",
                                                               "metric", "row_agreement")}, flush=True)


if __name__ == "__main__":
    main()
