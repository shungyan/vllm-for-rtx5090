# vLLM for RTX 5090 — Qwen3.8-27B (NVFP4)

Run **`Qwen/Qwen3.8-27B`** as an OpenAI-compatible API server with **vLLM** on an
**NVIDIA GeForce RTX 5090** (Blackwell, `sm_120`), quantized to **NVFP4** for the
Blackwell tensor cores. Ships with a one-command Docker setup.

The previous generation is kept alongside it — see [Rolling back to Qwen3.6](#rolling-back-to-qwen36).

- **Model:** `unsloth/Qwen3.8-27B-NVFP4` — multimodal (vision+text), hybrid
  linear/full-attention arch (`model_type: qwen3_5`, unchanged from Qwen3.6). The
  checkpoint is **~23.4 GB** (22.6 GB weights + 0.85 GB MTP head), and takes
  **~23 GB of VRAM** once `lm_head` is de-quantized to BF16 — note: *not* ~14 GB.
- **Hardware:** RTX 5090 — **32 GB dedicated VRAM**, on a host with **62 GB RAM**.
  This is very different from the DGX Spark / GB10 (121 GB *unified* memory) the
  upstream config assumed, and the difference drives every tuning choice below.
- **Engine:** vLLM 0.22.1 with the FlashInfer NVFP4 GEMM kernel, CUDA 13.
- **Packages:** managed by [`uv`](https://docs.astral.sh/uv/).

> The 32 GB VRAM and 62 GB system RAM are *hard, separate* limits. The defaults
> here are tuned so vLLM fits and, critically, so it can **never OOM-crash the
> host** (see *Memory & stability* below).

## Prerequisites

1. **NVIDIA proprietary driver** (`nvidia-driver-595-open` or newer) — and
   **`nouveau` must be blacklisted**. On this Blackwell card, nouveau loading
   hard-crashes the machine. Verify with `lsmod | grep nouveau` (must be empty).
   See *Avoiding host crashes* below.
2. **Docker** + **NVIDIA Container Toolkit** for GPU passthrough:
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   # sanity check:
   docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi -L
   ```

## Quick start (Docker — recommended)

```bash
# Point at your Hugging Face cache so weights aren't re-downloaded (~23 GB):
export HF_CACHE_DIR=$HOME/.cache/huggingface

docker compose build              # build the image
```

**One-time, before the first start:** this checkpoint quantizes `lm_head` to FP8,
which vLLM 0.22.1 cannot load (it builds an *unquantized* `ParallelLMHead`, then
dies with `no module or parameter named 'lm_head.weight_scale'`). Dequantize it in
the HF cache first — nothing else on the box may be holding the GPU:

```bash
docker run --rm --gpus all \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -v "$PWD/patch_lmhead_bf16.py:/patch.py:ro" \
  --entrypoint /app/.venv/bin/python vllm-qwen38-nvfp4:latest /patch.py
```

It is idempotent, keeps the original FP8 blob as a backup, and **must be re-run
after any re-download** of the model. Then:

```bash
docker compose up -d              # start in background
docker compose logs -f            # watch startup
docker compose ps                 # STATUS shows "healthy" when ready
```

First start is slow (~7–9 min cold): heavy imports + 23 GB weight load + a
one-time FlashInfer/GDN kernel JIT compile (cached afterward under
`~/.cache/vllm-jit`). Then call it:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Explain a GPU in one sentence."}],
    "max_tokens": 256
  }'
```

## What changed moving from Qwen3.6 to Qwen3.8

The architecture is identical (`qwen3_5`, 64 layers, hidden 5120, vocab 248320),
so vLLM loads it with no engine changes. The chat template and the extras did
change, and the compose file reflects that:

| Change | Setting | Why |
|---|---|---|
| **Thinking is parsed out server-side** | `REASONING_PARSER=qwen3` | Qwen3.8's template no longer splits `<think>…</think>` back out of an assistant message's `content` when replaying history — it reads `reasoning_content` only. Without the parser, multi-turn tool loops end up with nested think blocks. **API shape changes:** `content` now holds just the final answer, the chain-of-thought moves to the `reasoning` field on the message (vLLM 0.22.1 names it `reasoning`, NOT `reasoning_content` — same key in streaming deltas). Verified 2026-08-26. |
| **Thinking isn't re-fed every turn** | `DEFAULT_CHAT_TEMPLATE_KWARGS={"preserve_thinking": false}` | Its template defaults `preserve_thinking=true`, which replays every past `<think>` block into each prompt. |
| **Reasoning depth is tunable** | `chat_template_kwargs={"reasoning_effort": …}` | **New in 3.8 — Qwen3.6 had no such mechanism at all.** Nothing sets it here, so the template's own `default('xhigh')` applies: the MOST verbose level, injecting *"think carefully… validate key assumptions…"* into every request. Measured cost: on a 5-task coding benchmark `xhigh` spent 10,000 tokens and returned EMPTY `content` on 2 of 5; `medium` solved all 5 and was 3–5x cheaper (semver 4,898→1,447 tokens, bucket 4,800→853). `medium` is also the only level that injects no instruction text, so it is the closest match to how 3.6 behaved. Settable per request or server-wide via `DEFAULT_CHAT_TEMPLATE_KWARGS`. |
| **MTP speculative decoding** | `SPEC_DECODE=1`, `SPEC_NUM_TOKENS=1` | Qwen3.8 ships trained MTP weights (`model_mtp.safetensors`, in the safetensors index so vLLM loads them with the rest). vLLM **shares** the target's `embed_tokens` + `lm_head` with the draft head, so it costs only the extra decoder layer (total model load 23.12 GiB incl. MTP). Measured 76–96 % draft acceptance. Set to `"0"` to disable. |
| **`lm_head` is still FP8** | `patch_lmhead_bf16.py` | Same problem as 3.6, same fix. The script now takes a repo id and streams the (single, 22.5 GB) shard instead of loading it into RAM. |

Unchanged: the `qwen3_xml` tool-call format, vision support, and every memory knob.

## Memory & stability (why the defaults look the way they do)

The upstream GB10 config OOM-crashed this 32 GB/62 GB box. The `docker-compose.yml`
defaults here fix that:

| Setting | Value | Why |
|---|---|---|
| `mem_limit` / `memswap_limit` | `52g` | **Host-crash safety net.** With no cap, a vLLM memory spike triggers the *host* OOM-killer and locks up the machine. Capping makes the *container* OOM itself instead; ~10 GB stays free for the OS. |
| `ENFORCE_EAGER` | `0` | CUDA graphs **on**. Eager mode was launch-overhead bound (GPU 27 % util, 148 W): 10.7 → 61.3 tok/s (~5.7×) single-stream decode. Safe because `MAX_NUM_SEQS=8` keeps the capture set tiny (~0.11 GiB graph pool). |
| `MAX_NUM_SEQS` | `8` | Small graph-capture set → low graph memory → preserves KV cache for the 40 k context. |
| `LIMIT_MM` | `{"image":8,"video":0}` | Vision on, capped at 8 images/request. Startup profiles the encoder with a max-resolution dummy image (~50 GB RAM) — `MM_PROCESSOR_KWARGS` is what makes that affordable. |
| `MM_PROCESSOR_KWARGS` | `{"max_pixels": 1605632}` | Caps each image at ~2048 vision tokens, which tames the encoder profiling and per-request memory. |
| `MAX_JOBS` / `FLASHINFER_NVCC_THREADS` | `2` / `1` | The GDN linear-attention kernels JIT-compile on first run; ninja's default one-job-per-core spawns many multi-GB compilers and OOMs. One-time cost, then cached. |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | Bounds prefill/profiling batch memory. |
| `MAX_LEN` | `40960` | Context length. The SEICA agent reads whole manuals into the prompt (~32 k tokens/query), so 8192 overflowed → HTTP 400. |
| `GPU_UTIL` | `0.86` | The fraction of the card vLLM may touch; the rest is left for the desktop. Was 0.93 (crash loop, see the warning below), then 0.90, then 0.86 once OBS/NVENC needed room — this GPU also drives the display. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Counters allocator fragmentation; the OOM traces showed 574–665 MB reserved-but-unallocated. |
| `REVISION` | pinned sha | The `lm_head` patch lives in ONE HF snapshot dir. An upstream commit creates a new snapshot of the same unpatched blobs and the crash returns. Pinned in two places — see the table above. |
| `SPEC_DECODE` / `SPEC_NUM_TOKENS` | `1` / `1` | MTP speculative decoding (see above). |
| `REASONING_PARSER` | `qwen3` | Thinking → the `reasoning` field (see above). |

`restart: always` is safe **only** because nouveau is permanently blacklisted and
`mem_limit` prevents a host OOM, so it can't crash-loop the box. All of the above
are env-overridable (see `serve.sh`).

### Avoiding host crashes (nouveau)

If the machine hard-reboots when the GPU comes under load, check
`journalctl -k -b -1` — a boot that ends abruptly at `fbcon: Taking over console`
with a `nouveau` NULL-pointer Oops means nouveau grabbed the card. Permanently
disable it:

```bash
sudo tee /etc/modprobe.d/blacklist-nouveau.conf >/dev/null <<'EOF'
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
install nouveau /bin/false
EOF
# also disable on the kernel cmdline (covers early boot):
sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\([^"]*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 modprobe.blacklist=nouveau nouveau.modeset=0"/' /etc/default/grub
sudo update-initramfs -u && sudo update-grub
sudo reboot
```

## Rolling back to Qwen3.6

The previous stack is pinned in `docker-compose.qwen3.6.yml` + `Dockerfile.qwen3.6`.
**Only one generation can run at a time** — each needs ~23 GB of the 32 GB VRAM,
and both bind port 8000:

```bash
docker compose down                                        # stop Qwen3.8 first
docker compose -f docker-compose.qwen3.6.yml up -d --build
```

Its `restart` policy defaults to `no` (not `always`) so the fallback can't quietly
come back at boot and fight Qwen3.8 for the GPU. If you make 3.6 primary again,
start it with `RESTART_POLICY=always`. Qwen3.6 is served **without** a reasoning
parser, so `<think>` tags stay inline in `content` there — its own template strips
them back out on replay.

## Run on the host (without Docker)

```bash
uv sync           # creates .venv and installs vLLM + deps
MAX_JOBS=2 MAX_LEN=40960 GPU_UTIL=0.86 MAX_NUM_SEQS=8 \
  REASONING_PARSER=qwen3 TOOL_CALL_PARSER=qwen3_xml ./serve.sh
```

`serve.sh` auto-detects a missing system CUDA toolkit and falls back to the
complete toolkit (nvcc/headers/libs) bundled in the torch `cu13` wheel inside the
venv, so FlashInfer's JIT works without a system CUDA install. (It needs a C++
compiler — `sudo apt-get install -y build-essential`.)

## Using the API

OpenAI-compatible — works with the `openai` SDK (`base_url="http://localhost:8000/v1"`).
Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`.

- **Single request:** `uv run python test_request.py "your prompt"`
- **Concurrency test:** `python3 test_parallel.py 32 64` (32 concurrent, 64 tokens each)
- **Streaming:** add `"stream": true` and use `curl -N`.
- **Vision is enabled** (8 images/request, `max_pixels` ~2048 vision tokens each;
  video is off). Send OpenAI-style `image_url` parts with a `data:` URI.
  **Budget tokens generously for image requests** — at the default `xhigh`
  reasoning effort a 400-token cap returned EMPTY `content` with no error, because
  all of it went to thinking. Use >=1500 `max_tokens`, or pass
  `chat_template_kwargs={"reasoning_effort": "medium"}`.
- It's a reasoning model — the chain-of-thought comes back in `message.reasoning`
  (not `content`, and NOT `reasoning_content`; streaming deltas use `reasoning` too). Pass `chat_template_kwargs={"enable_thinking": false}` to skip
  thinking entirely, or `{"reasoning_effort": "low"}` to keep it brief.
- **Recommended sampling** (from the model card): thinking mode `temperature=1.0,
  top_p=0.95, top_k=20`; non-thinking `temperature=0.7, top_p=0.8, top_k=20,
  presence_penalty=1.5`.

## Concurrency

vLLM does continuous batching — just fire N requests in parallel at the one
server and they're batched together. There is **no hard 8-request limit** in the
protocol; concurrency is bounded by `--max-num-seqs` (8 here, to protect the KV
cache) and KV-cache size.

**Measured on Qwen3.8 (2026-09-24)**, RTX 5090, CUDA graphs on, MTP on,
`MAX_LEN=40960`, `GPU_UTIL=0.86`:

| Metric | Value |
|---|---|
| GPU KV cache | 2.66 GiB = 65,280 tokens (1.59x concurrency at a full 40,960-token request) |
| Single-stream decode | ~73 tok/s (3-run mean; vs ~61 tok/s for Qwen3.6 with CUDA graphs, no MTP) |
| MTP draft acceptance | 68–70 % recently, 76–96 % when first measured (`num_speculative_tokens: 1`) |
| Model weights on GPU | 23.12 GiB (includes MTP — vLLM shares the target's embedding + lm_head with the draft head) |
| VRAM in use | ~30.3 GB of 32.6 GB, leaving ~1.8–3 GB for the desktop |
| Engine init (warm) | ~55 s (17 s compile) |
| Cold start | up to ~7 min — the model is read from spinning disks, see *Storage* below |

The KV cache is **already FP8**: the checkpoint carries a `kv_cache_scheme`
(`num_bits: 8, type: float`) that vLLM applies automatically, even though the
flag reports `kv_cache_dtype=auto`. The arithmetic confirms it — 16 full-attention
layers x 2 (K+V) x 4 kv_heads x 256 head_dim = 32,768 elements/token, so BF16
would be 64.0 KiB/token and FP8 32.0 KiB/token; the measured 42.7 KiB/token
(which also amortises the 48 GDN layers' mamba state) is below the BF16 figure
and therefore cannot be BF16. **Passing `--kv-cache-dtype fp8` explicitly is a
no-op here** and buys no extra context.

> **Sizing warning (learned the hard way, 2026-09-01).** `GPU_UTIL` was 0.93, which
> gave the KV cache 4.7 GiB / 116,480 tokens and left only ~50-250 MB free. Ordinary
> prefill scratch buffers then failed to allocate mid-request — `torch.OutOfMemoryError`
> -> `EngineDeadError` -> the entire engine dies and the container restarts. 7 restarts
> in 5 days. Every crash logged `GPU KV cache usage: 0.0%`: the KV pool was **empty**
> while the GPU was **full**. A big KV cache number is not free headroom — it is
> memory taken away from activations. Do not go above 0.90; do not go below ~0.84
> (the KV floor for a single 40,960-token request is 1.65 GiB).

### Storage

`/` is ext4 on a RAID1 of two **spinning** Toshiba DT02ACA200 disks, while a 1.9 TB
NVMe (`nvme0n1`) sits idle with its array `md126` degraded `[_U]`. The 23.8 GB
checkpoint, the HF cache and all Docker images live on the HDDs, so a cold restart
can sit in `D` (uninterruptible disk sleep) for minutes — observed 112 ms write /
147 ms flush latency at 85 % utilisation. The fast "Model loading took ~10 s" case
is warm page cache. Recording video into `~/Videos` hits the same array and
contends directly with model loads. Moving `HF_CACHE_DIR` (and ideally Docker's
data-root) to the NVMe is the real fix.

### Sharing the GPU with desktop apps

This card also drives the display. vLLM at `GPU_UTIL 0.93` left 644 MB free, which
was not enough for OBS: NVENC hard-crashed the moment recording started (the OBS
log ends exactly at `==== Recording Start ====` with no shutdown). 0.86 leaves
~3 GB, enough for a 1080p NVENC session. DaVinci Resolve needs far more than that —
stop vLLM before using it. Note this only works because vLLM starts **first** and
profiles against a free card.

Older Qwen3.6 numbers (text-only, eager, `MAX_LEN=8192`, `MAX_NUM_SEQS=64`), kept
for reference — they predate CUDA graphs, the 40 k context and Qwen3.8:

| Concurrency | Result | Aggregate throughput |
|---|---|---|
| 16 | 16/16 ✅ | ~107 tok/s |
| 32 | 32/32 ✅ | ~145 tok/s |
| 48 | 48/48 ✅ | ~146 tok/s |

## Repository layout

| File | Purpose |
|---|---|
| `serve.sh` | Launches `vllm serve` with RTX 5090-tuned, env-overridable flags (serves either generation) |
| `docker-compose.yml` | **Live stack — Qwen3.8.** GPU passthrough, HF-cache mount, memory caps |
| `Dockerfile` | CUDA 13 devel base + uv-managed Python + vLLM |
| `docker-compose.qwen3.6.yml` | Pinned Qwen3.6 fallback |
| `Dockerfile.qwen3.6` | Pinned Qwen3.6 fallback image (identical body; separate tag) |
| `patch_lmhead_bf16.py` | One-time FP8 → BF16 `lm_head` dequant in the HF cache |
| `test_request.py` | Minimal single-request client |
| `test_parallel.py` | Concurrency / aggregate-throughput test |
| `pyproject.toml` / `uv.lock` | Pinned dependencies |
| `BENCHMARKS.md` | Throughput measurements (original GB10 numbers) |

## License

Code here is provided as-is. The model is under its own license (Apache-2.0 for
Qwen3.8-27B); review the model card before use.
