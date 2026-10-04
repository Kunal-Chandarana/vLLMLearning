#!/usr/bin/env python3
"""
Compare Benchmark Runs

Prints two serving_benchmark.py result files side by side, per load level,
with the ratio between them.

    python benchmarks/compare_results.py benchmarks/results/vllm-*.json benchmarks/results/nanoserve-*.json
"""

import argparse
import json


def load(path):
    with open(path) as f:
        data = json.load(f)
    label = data["config"].get("tag") or path
    return label, {r["load_value"]: r["summary"] for r in data["runs"]}


def main():
    parser = argparse.ArgumentParser(description="Compare two benchmark result files.")
    parser.add_argument("baseline", help="Result JSON to compare against (e.g. vllm)")
    parser.add_argument("candidate", help="Result JSON being compared (e.g. nanoserve)")
    args = parser.parse_args()

    (a_name, a), (b_name, b) = load(args.baseline), load(args.candidate)
    metrics = [
        ("output tok/s", lambda s: s["output_token_throughput"], "{:.0f}"),
        ("TTFT p50 ms", lambda s: s["ttft_ms"]["p50"], "{:.0f}"),
        ("TPOT p50 ms", lambda s: s["tpot_ms"]["p50"], "{:.1f}"),
    ]
    header = f"{'load':>6} | " + " | ".join(f"{m:^28}" for m, _, _ in metrics)
    sub = f"{'':>6} | " + " | ".join(f"{a_name[:9]:>9} {b_name[:9]:>9} {'ratio':>8}" for _ in metrics)
    print(header)
    print(sub)
    print("-" * len(sub))
    for load_value in sorted(set(a) & set(b)):
        cells = []
        for _, get, fmt in metrics:
            va, vb = get(a[load_value]), get(b[load_value])
            ratio = vb / va if va else float("nan")
            cells.append(f"{fmt.format(va):>9} {fmt.format(vb):>9} {ratio:>7.2f}x")
        print(f"{load_value:>6g} | " + " | ".join(cells))
    print(f"\nratio = {b_name} / {a_name}. For tok/s higher is better; for latency lower is better.")


if __name__ == "__main__":
    main()
