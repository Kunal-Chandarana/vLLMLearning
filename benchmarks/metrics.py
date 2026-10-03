"""
Serving Benchmark Metrics

Per-request timing records and the latency / throughput summary computed from
them. Kept free of networking code so it can be unit tested on its own.

Metric definitions (the same ones vLLM's own benchmark_serving.py reports):
- TTFT  (time to first token): request sent -> first generated token received.
        Dominated by queueing + prefill.
- TPOT  (time per output token): (E2E - TTFT) / (output_tokens - 1), per request.
        The average decode step time that request experienced.
- ITL   (inter-token latency): gap between consecutive streamed chunks, pooled
        over all requests. Shows decode jitter (e.g. stalls caused by other
        requests' prefills being scheduled in the same batch).
- E2E   (end-to-end latency): request sent -> last token received.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np


PERCENTILES = (50, 90, 95, 99)


@dataclass
class RequestResult:
    """Timing record for a single streamed request. Times are in seconds."""
    success: bool
    prompt_tokens: int = 0
    output_tokens: int = 0
    start_time: float = 0.0
    ttft: float = 0.0
    e2e_latency: float = 0.0
    itls: List[float] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def tpot(self) -> Optional[float]:
        if self.output_tokens <= 1:
            return None
        return (self.e2e_latency - self.ttft) / (self.output_tokens - 1)


def _distribution(values_s: List[float]) -> Dict[str, float]:
    """Mean / median / percentiles of a list of seconds, reported in milliseconds."""
    if not values_s:
        return {"mean": float("nan"), **{f"p{p}": float("nan") for p in PERCENTILES}}
    arr = np.asarray(values_s) * 1000.0
    stats = {"mean": float(arr.mean()), "std": float(arr.std())}
    for p in PERCENTILES:
        stats[f"p{p}"] = float(np.percentile(arr, p))
    return stats


def summarize(results: List[RequestResult], duration_s: float,
              slo_ttft_ms: Optional[float] = None,
              slo_tpot_ms: Optional[float] = None) -> Dict:
    """Aggregate per-request results from one benchmark run.

    duration_s is wall-clock time from the first request sent to the last
    response finished, so throughput numbers include idle gaps in the load.

    Goodput counts only requests that met every SLO given, which is a more
    honest capacity number than raw throughput: a server can push throughput
    up by letting latency explode.
    """
    ok = [r for r in results if r.success]
    output_tokens = sum(r.output_tokens for r in ok)
    prompt_tokens = sum(r.prompt_tokens for r in ok)
    tpots = [r.tpot for r in ok if r.tpot is not None]

    good = ok
    if slo_ttft_ms is not None:
        good = [r for r in good if r.ttft * 1000 <= slo_ttft_ms]
    if slo_tpot_ms is not None:
        good = [r for r in good if r.tpot is None or r.tpot * 1000 <= slo_tpot_ms]

    return {
        "num_requests": len(results),
        "completed": len(ok),
        "failed": len(results) - len(ok),
        "duration_s": duration_s,
        "total_prompt_tokens": prompt_tokens,
        "total_output_tokens": output_tokens,
        "request_throughput": len(ok) / duration_s if duration_s > 0 else 0.0,
        "output_token_throughput": output_tokens / duration_s if duration_s > 0 else 0.0,
        "total_token_throughput": (prompt_tokens + output_tokens) / duration_s if duration_s > 0 else 0.0,
        "goodput": len(good) / duration_s if duration_s > 0 else 0.0,
        "ttft_ms": _distribution([r.ttft for r in ok]),
        "tpot_ms": _distribution(tpots),
        "itl_ms": _distribution([x for r in ok for x in r.itls]),
        "e2e_ms": _distribution([r.e2e_latency for r in ok]),
        "errors": sorted({r.error for r in results if r.error})[:5],
    }


def results_to_dicts(results: List[RequestResult]) -> List[Dict]:
    """Raw per-request records for saving alongside the summary."""
    rows = []
    for r in results:
        row = asdict(r)
        row["tpot"] = r.tpot
        rows.append(row)
    return rows
