import math
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
from metrics import RequestResult, summarize


def make(ttft, e2e, out_tokens, itls=None, success=True):
    return RequestResult(success=success, prompt_tokens=100, output_tokens=out_tokens,
                         ttft=ttft, e2e_latency=e2e, itls=itls or [])


def test_tpot_excludes_first_token():
    # 0.1s to first token, then 9 more tokens over 0.9s -> 100ms per token
    r = make(ttft=0.1, e2e=1.0, out_tokens=10)
    assert r.tpot == pytest.approx(0.1)


def test_tpot_undefined_for_single_token():
    assert make(ttft=0.1, e2e=0.1, out_tokens=1).tpot is None


def test_summary_throughput_and_failures():
    results = [make(0.1, 1.0, 10), make(0.2, 1.0, 10), make(0, 0, 0, success=False)]
    s = summarize(results, duration_s=2.0)
    assert s["completed"] == 2
    assert s["failed"] == 1
    assert s["request_throughput"] == pytest.approx(1.0)
    assert s["output_token_throughput"] == pytest.approx(10.0)
    assert s["total_token_throughput"] == pytest.approx(110.0)
    assert s["ttft_ms"]["mean"] == pytest.approx(150.0)


def test_itl_pools_all_requests():
    results = [make(0.1, 0.3, 3, itls=[0.1, 0.1]), make(0.1, 0.5, 3, itls=[0.2, 0.2])]
    s = summarize(results, duration_s=1.0)
    assert s["itl_ms"]["p50"] == pytest.approx(150.0)
    assert s["itl_ms"]["p99"] == pytest.approx(200.0)


def test_goodput_applies_slos():
    fast = make(0.05, 0.5, 10)   # TTFT 50ms, TPOT 50ms
    slow_start = make(0.5, 1.0, 10)  # TTFT 500ms
    slow_decode = make(0.05, 2.0, 10)  # TPOT ~217ms
    s = summarize([fast, slow_start, slow_decode], duration_s=1.0,
                  slo_ttft_ms=100, slo_tpot_ms=100)
    assert s["goodput"] == pytest.approx(1.0)
    assert s["request_throughput"] == pytest.approx(3.0)


def test_empty_results_are_nan_not_crash():
    s = summarize([], duration_s=1.0)
    assert s["completed"] == 0
    assert math.isnan(s["ttft_ms"]["p99"])
