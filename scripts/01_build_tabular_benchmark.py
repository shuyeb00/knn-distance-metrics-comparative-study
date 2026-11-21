import argparse
import time
from pathlib import Path

import pandas as pd

from knnmetrics.data_tabular import CC18_SUITE_ID, MAX_ROWS, configure_openml_cache, suite_task_ids, task_metadata

REPO = Path(__file__).resolve().parents[1]


def run_metadata(out_path):
    task_ids = suite_task_ids(CC18_SUITE_ID)
    print(f"suite {CC18_SUITE_ID}: {len(task_ids)} tasks")
    rows, failures = [], []
    for i, tid in enumerate(task_ids, 1):
        t0 = time.perf_counter()
        try:
            row = task_metadata(tid)
            rows.append(row)
            print(f"[{i:2d}/{len(task_ids)}] task {tid:>6} {row['name'][:28]:<28} "
                  f"N={row['n_instances']:>6} D={row['n_features']:>5} C={row['n_classes']:>3} "
                  f"num={row['n_numeric']:>5} nom={row['n_nominal']:>4} "
                  f"miss={row['missing_rate'] or 0:.3f} ({time.perf_counter() - t0:.1f}s)")
        except Exception as exc:
            failures.append({"task_id": tid, "error": repr(exc)})
            print(f"[{i:2d}/{len(task_ids)}] task {tid:>6} FAILED: {exc!r}")

    df = pd.DataFrame(rows).sort_values("task_id")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(df)} tasks)")
    if failures:
        fail_path = out_path.with_name(out_path.stem + "_failures.csv")
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f"{len(failures)} failures written to {fail_path}; rerun to retry")

    numeric = df[df["all_numeric"]]
    print(f"\nall-numeric tasks: {len(numeric)} of {len(df)}")
    print(f"  with missing values: {int((numeric['missing_cells'] > 0).sum())}")
    print(f"  above the {MAX_ROWS}-row cap: {int(numeric['exceeds_row_cap'].sum())}")
    print(f"  with row-id or ignore columns: "
          f"{int(((numeric['row_id_attribute'] != '') | (numeric['ignore_attributes'] != '')).sum())}")
    print(f"mixed-type tasks (secondary set): {len(df) - len(numeric)}")


def main():
    parser = argparse.ArgumentParser(description="Build the OpenML-CC18 tabular benchmark.")
    parser.add_argument("stage", choices=["metadata"])
    parser.add_argument("--cache", default=str(REPO / "cache" / "openml"))
    parser.add_argument("--out", default=str(REPO / "results" / "benchmark" / "cc18_task_metadata.csv"))
    args = parser.parse_args()

    configure_openml_cache(args.cache)
    if args.stage == "metadata":
        run_metadata(Path(args.out).resolve())


if __name__ == "__main__":
    main()
