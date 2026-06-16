#!/usr/bin/env python3
"""Concurrency / aggregate-throughput test for the vLLM server.

Fires N requests at the server *at the same time* and reports how many the
engine served concurrently and the aggregate decode throughput. vLLM does
continuous batching, so there is no hard "8 request" cap — concurrency is
bounded by KV-cache size (raise GPU_UTIL for more) and --max-num-seqs.

Usage:
    uv run python test_parallel.py            # default 16 concurrent
    uv run python test_parallel.py 32         # 32 concurrent
    uv run python test_parallel.py 16 512     # 16 concurrent, 512 tokens each
"""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("SERVED_NAME", "qwen3.6-27b")
API_KEY = os.environ.get("VLLM_API_KEY", "")

CONCURRENCY = int(sys.argv[1]) if len(sys.argv) > 1 else 16
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
PROMPT = "Write a detailed essay about the history of computing."


def one_request(i: int) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "ignore_eos": True,  # force full MAX_TOKENS so per-request work is uniform
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}),
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    dt = time.perf_counter() - t0
    toks = data.get("usage", {}).get("completion_tokens", 0)
    return {"i": i, "dt": dt, "toks": toks, "ok": True}


def main() -> None:
    print(f"Firing {CONCURRENCY} concurrent requests "
          f"({MAX_TOKENS} tokens each) at {BASE} ...")
    wall0 = time.perf_counter()
    results, errors = [], []
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(one_request, i) for i in range(CONCURRENCY)]
        for f in cf.as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
    wall = time.perf_counter() - wall0

    ok = len(results)
    total_toks = sum(r["toks"] for r in results)
    print(f"\n  succeeded : {ok}/{CONCURRENCY}")
    if errors:
        print(f"  failed    : {len(errors)}")
        for e in errors[:5]:
            print(f"    - {e}")
    if not results:
        sys.exit(1)
    per_req = [r["toks"] / r["dt"] for r in results if r["dt"] > 0]
    print(f"  wall time : {wall:.1f}s")
    print(f"  agg tput  : {total_toks / wall:.1f} tok/s "
          f"(all {ok} requests / wall time)")
    print(f"  per-req   : avg {sum(per_req)/len(per_req):.1f} tok/s, "
          f"min {min(per_req):.1f}, max {max(per_req):.1f}")
    print(f"  latency   : min {min(r['dt'] for r in results):.1f}s, "
          f"max {max(r['dt'] for r in results):.1f}s")


if __name__ == "__main__":
    main()
