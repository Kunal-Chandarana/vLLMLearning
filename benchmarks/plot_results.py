#!/usr/bin/env python3
"""
Plot Serving Benchmark Results

Turns one or more serving_benchmark.py JSON files into a 2x2 figure. Pass
several files to compare configurations (e.g. baseline vs prefix caching vs
FP8) on the same axes; each file becomes one colored series.

    python benchmarks/plot_results.py results/baseline-*.json results/fp8-*.json \\
        --out benchmarks/results/baseline_vs_fp8.png

Panels:
1. Output token throughput vs load        -- where does the server saturate?
2. TTFT p50 (solid) / p99 (dashed) vs load -- queueing and prefill cost
3. TPOT p50 (solid) / p99 (dashed) vs load -- decode slowdown as batches grow
4. Per-user speed vs total throughput     -- the trade-off curve; up-and-right is better
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Fixed categorical order (validated for color-vision deficiency); series keep
# their slot no matter how many files are passed.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"


def load(path: str) -> Dict:
    with open(path) as f:
        data = json.load(f)
    data["label"] = data["config"].get("tag") or os.path.basename(path)
    return data


def style_axis(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot(datasets: List[Dict], out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), facecolor=SURFACE)
    (ax_tput, ax_ttft), (ax_tpot, ax_pareto) = axes

    load_type = datasets[0]["runs"][0]["load_type"]
    xlabel = "Concurrent requests" if load_type == "concurrency" else "Request rate (req/s)"
    use_log_x = load_type == "concurrency"

    for i, data in enumerate(datasets):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        label = data["label"]
        runs = sorted(data["runs"], key=lambda r: r["load_value"])
        x = [r["load_value"] for r in runs]
        s = [r["summary"] for r in runs]
        line = dict(color=color, linewidth=2, marker="o", markersize=5)

        ax_tput.plot(x, [v["output_token_throughput"] for v in s], label=label, **line)
        ax_ttft.plot(x, [v["ttft_ms"]["p50"] for v in s], label=f"{label} p50", **line)
        ax_ttft.plot(x, [v["ttft_ms"]["p99"] for v in s], label=f"{label} p99", linestyle="--", **line)
        ax_tpot.plot(x, [v["tpot_ms"]["p50"] for v in s], label=f"{label} p50", **line)
        ax_tpot.plot(x, [v["tpot_ms"]["p99"] for v in s], label=f"{label} p99", linestyle="--", **line)

        # Per-user speed: how fast one stream reads to a user (tokens/s).
        user_speed = [1000.0 / v["tpot_ms"]["p50"] if v["tpot_ms"]["p50"] > 0 else 0 for v in s]
        total = [v["output_token_throughput"] for v in s]
        ax_pareto.plot(total, user_speed, label=label, **line)
        for xv, yv, lv in zip(total, user_speed, x):
            ax_pareto.annotate(f"{lv:g}", (xv, yv), textcoords="offset points", xytext=(5, 5),
                               fontsize=7, color=TEXT_SECONDARY)

    style_axis(ax_tput, "Output throughput", xlabel, "Output tokens/s")
    style_axis(ax_ttft, "Time to first token", xlabel, "ms")
    style_axis(ax_tpot, "Time per output token", xlabel, "ms")
    style_axis(ax_pareto, "Throughput vs per-user speed (labels = load)",
               "Total output tokens/s", "Tokens/s per request (1000 / TPOT p50)")
    if use_log_x:
        for ax in (ax_tput, ax_ttft, ax_tpot):
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    for ax in (ax_tput, ax_ttft, ax_tpot, ax_pareto):
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, frameon=False, labelcolor=TEXT_PRIMARY)

    cfg = datasets[0]["config"]
    fig.suptitle(f"{cfg['model']}  ·  input ~{cfg['input_len']} tok, output {cfg['output_len']} tok",
                 x=0.06, ha="left", fontsize=13, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot serving benchmark results.")
    parser.add_argument("files", nargs="+", help="Result JSON files from serving_benchmark.py")
    parser.add_argument("--out", help="Output PNG path (default: next to the first file)")
    args = parser.parse_args()
    datasets = [load(p) for p in args.files]
    out = args.out or os.path.splitext(args.files[0])[0] + ".png"
    plot(datasets, out)


if __name__ == "__main__":
    main()
