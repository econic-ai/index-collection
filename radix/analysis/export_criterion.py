#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Criterion summaries to analysis/data/<impl>.csv"
    )
    parser.add_argument("--impl", required=True, help="Implementation key (e.g. m0_hashbrown)")
    parser.add_argument("--tag", default="", help="Optional run tag appended to each row")
    parser.add_argument(
        "--criterion-dir",
        default="target/criterion",
        help="Criterion output directory (default: target/criterion)",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis/data",
        help="CSV output directory (default: analysis/data)",
    )
    return parser.parse_args()


def parse_op_and_lf_from_benchmark(benchmark_path: Path, impl: str) -> tuple[str, str] | None:
    if not benchmark_path.exists():
        return None
    data = json.loads(benchmark_path.read_text())
    group_id = data.get("group_id")
    load_factor = data.get("value_str")
    if not isinstance(group_id, str) or not isinstance(load_factor, str):
        return None

    prefix = f"impl={impl}/op="
    if not group_id.startswith(prefix):
        return None

    op = group_id.removeprefix(prefix)
    try:
        float(load_factor)
    except ValueError:
        return None
    return op, load_factor


def load_point_estimates(estimates_path: Path) -> dict[str, float]:
    data = json.loads(estimates_path.read_text())
    out: dict[str, float] = {}
    for metric in ("mean", "median", "std_dev"):
        value = data.get(metric, {}).get("point_estimate")
        if isinstance(value, (int, float)):
            out[metric] = float(value)
    return out


def load_elements_per_iteration(benchmark_path: Path) -> float | None:
    if not benchmark_path.exists():
        return None
    data = json.loads(benchmark_path.read_text())
    throughput = data.get("throughput")
    if not isinstance(throughput, dict):
        return None
    value = throughput.get("Elements")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def collect_rows(criterion_dir: Path, impl: str, tag: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for estimates_path in sorted(criterion_dir.rglob("new/estimates.json")):
        benchmark_path = estimates_path.parent / "benchmark.json"
        parsed = parse_op_and_lf_from_benchmark(benchmark_path, impl)
        if parsed is None:
            continue

        op, load_factor = parsed
        estimates = load_point_estimates(estimates_path)
        if not estimates:
            continue

        elems_per_iter = load_elements_per_iteration(benchmark_path)
        mean_ns = estimates.get("mean")

        for metric, value in estimates.items():
            rows.append(
                {
                    "timestamp": timestamp,
                    "impl": impl,
                    "op": op,
                    "load_factor": load_factor,
                    "metric": metric,
                    "value": f"{value:.6f}",
                    "unit": "ns",
                    "tag": tag,
                }
            )

        if elems_per_iter is not None and mean_ns is not None and mean_ns > 0:
            ops_per_sec = elems_per_iter * 1_000_000_000.0 / mean_ns
            rows.append(
                {
                    "timestamp": timestamp,
                    "impl": impl,
                    "op": op,
                    "load_factor": load_factor,
                    "metric": "throughput_ops_per_sec",
                    "value": f"{ops_per_sec:.6f}",
                    "unit": "ops/s",
                    "tag": tag,
                }
            )

    return rows


def append_rows(out_csv: Path, rows: list[dict[str, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    fields = ["timestamp", "impl", "op", "load_factor", "metric", "value", "unit", "tag"]
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    criterion_dir = Path(args.criterion_dir)
    if not criterion_dir.exists():
        print(f"Criterion directory not found: {criterion_dir}")
        return 1

    rows = collect_rows(criterion_dir, args.impl, args.tag)
    if not rows:
        print(f"No matching Criterion summaries found for impl={args.impl}")
        return 1

    out_csv = Path(args.out_dir) / f"{args.impl}.csv"
    append_rows(out_csv, rows)
    print(f"Appended {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
