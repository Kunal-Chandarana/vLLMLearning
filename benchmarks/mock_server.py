#!/usr/bin/env python3
"""
Mock OpenAI-compatible Streaming Server

A fake LLM server for testing the benchmark harness without a GPU. It mimics
the shape of a continuous-batching engine, not its numbers:

- Prefill costs time proportional to prompt length, and prefills are
  serialized (one at a time), so TTFT grows with queueing under load.
- Every decode step emits one token for each running request. A step gets
  slower as the batch grows, so TPOT rises with concurrency while total
  throughput still goes up.

    python benchmarks/mock_server.py --port 8001
"""

import argparse
import asyncio
import json
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


class FakeEngine:
    def __init__(self, prefill_ms_per_1k: float, step_ms: float, step_ms_per_seq: float):
        self.prefill_ms_per_1k = prefill_ms_per_1k
        self.step_ms = step_ms
        self.step_ms_per_seq = step_ms_per_seq
        self.prefill_lock = None  # created on first use, inside the server's event loop
        self.running = 0

    async def prefill(self, prompt_tokens: int):
        if self.prefill_lock is None:
            self.prefill_lock = asyncio.Lock()
        async with self.prefill_lock:
            await asyncio.sleep(self.prefill_ms_per_1k * prompt_tokens / 1000 / 1000)

    async def decode_step(self):
        await asyncio.sleep((self.step_ms + self.step_ms_per_seq * self.running) / 1000)


def build_app(engine: FakeEngine, model_name: str) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        prompt_tokens = len(prompt.split())
        max_tokens = int(body.get("max_tokens", 16))
        include_usage = (body.get("stream_options") or {}).get("include_usage", False)
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"

        def sse(obj) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        async def stream():
            await engine.prefill(prompt_tokens)
            engine.running += 1
            try:
                for i in range(max_tokens):
                    if i > 0:
                        await engine.decode_step()
                    yield sse({"id": req_id, "object": "text_completion", "created": int(time.time()),
                               "model": model_name,
                               "choices": [{"index": 0, "text": " tok", "finish_reason": None}]})
            finally:
                engine.running -= 1
            if include_usage:
                yield sse({"id": req_id, "object": "text_completion", "model": model_name, "choices": [],
                           "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens,
                                     "total_tokens": prompt_tokens + max_tokens}})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main():
    parser = argparse.ArgumentParser(description="Mock streaming LLM server for benchmark testing.")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--prefill-ms-per-1k", type=float, default=40.0)
    parser.add_argument("--step-ms", type=float, default=10.0)
    parser.add_argument("--step-ms-per-seq", type=float, default=0.3)
    args = parser.parse_args()
    engine = FakeEngine(args.prefill_ms_per_1k, args.step_ms, args.step_ms_per_seq)
    uvicorn.run(build_app(engine, args.model), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
