#!/usr/bin/env python3
"""CD's GPU test: what the clients of this server rely on, against the live container.

    python3 ci/live_test.py            (VLLM_BASE_URL, SERVED_NAME to point it elsewhere)

Stdlib only, runs on the host. At most 2 requests at once: the server runs 2 sequences (MAX_NUM_SEQS).
"""
import base64
import json
import os
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("SERVED_NAME", "qwen3.8-27b")
CONTEXT = 40960                     # docker-compose.yml MAX_LEN; the SEICA agent needs ~32k prompts
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def call(path, body=None):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode() if body else None,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def chat(content, **kw):
    return call("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": content}],
                                         "temperature": 0, **kw})


def red_png(n=64):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    raw = b"".join(b"\0" + b"\xff\x00\x00" * n for _ in range(n))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def t_models():
    m = call("/v1/models")["data"][0]
    assert m["id"] == MODEL, m["id"]
    assert m["max_model_len"] == CONTEXT, m["max_model_len"]


def t_reasoning():
    msg = chat("What is 17*23? Reply with just the number.", max_tokens=6000)["choices"][0]["message"]
    assert "391" in (msg["content"] or ""), msg["content"]
    assert "<think>" not in msg["content"], "thinking leaked into content (reasoning parser)"
    assert msg.get("reasoning"), "no message.reasoning (reasoning parser)"


def t_tool_call():
    tools = [{"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
    msg = chat("Use the calculator to compute 1234*5678.", tools=tools, max_tokens=500, **NO_THINK)["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    assert calls and calls[0]["function"]["name"] == "calculator", msg
    assert "expression" in json.loads(calls[0]["function"]["arguments"])


def t_image():
    url = "data:image/png;base64," + base64.b64encode(red_png()).decode()
    content = [{"type": "text", "text": "What colour is this image? One word."},
               {"type": "image_url", "image_url": {"url": url}}]
    text = chat(content, max_tokens=20, **NO_THINK)["choices"][0]["message"]["content"]
    assert "red" in text.lower(), text


def t_long_context():
    lines = [f"Line {i:05d}: the quick brown fox jumps over the lazy dog." for i in range(2000)]
    lines[1234] = "Line 01234: the secret code is 4721."
    r = chat("\n".join(lines) + "\n\nWhat is the secret code? Reply with just the number.", max_tokens=20, **NO_THINK)
    assert r["usage"]["prompt_tokens"] > 25000, r["usage"]
    assert "4721" in r["choices"][0]["message"]["content"], r["choices"][0]["message"]["content"]


def t_two_at_once():
    with ThreadPoolExecutor(2) as pool:
        out = list(pool.map(lambda _: chat("Say OK.", max_tokens=10, **NO_THINK), range(2)))
    assert all(o["choices"][0]["message"]["content"] for o in out)


def main():
    failed = 0
    for t in (t_models, t_reasoning, t_tool_call, t_image, t_long_context, t_two_at_once):
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}"[:600], flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
