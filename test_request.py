#!/usr/bin/env python3
"""Minimal POST-request client for the vLLM server.

Usage:
    uv run python test_request.py "Explain what a GPU is in one sentence."
"""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("SERVED_NAME", "qwen3.6-27b")
API_KEY = os.environ.get("VLLM_API_KEY", "")


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Say hello in one short sentence."
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}),
        },
    )
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    print(data["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
