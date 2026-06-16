# vLLM for RTX 5090 — Qwen3.6-27B (NVFP4)

Run **`Qwen/Qwen3.6-27B`** as an OpenAI-compatible API server with **vLLM** on an
**NVIDIA GeForce RTX 5090** (Blackwell, `sm_120`), quantized to **NVFP4** for the
Blackwell tensor cores. Ships with a one-command Docker setup.

- **Model:** `unsloth/Qwen3.6-27B-NVFP4` — multimodal (vision+text), hybrid
  linear/full-attention arch (`model_type: qwen3_5`). The checkpoint is
  **~24.6 GB** (the NVFP4 weights take ~24 GB of VRAM — note: *not* ~14 GB).
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
# Point at your Hugging Face cache so weights aren't re-downloaded (~24 GB):
export HF_CACHE_DIR=$HOME/.cache/huggingface

docker compose up -d --build      # build + start in background
docker compose logs -f            # watch startup (downloads ~24 GB on first run)
docker compose ps                 # STATUS shows "healthy" when ready
```

First start is slow (~7–9 min cold): heavy imports + 24 GB weight load + a
one-time FlashInfer/GDN kernel JIT compile (cached afterward under
`~/.cache/flashinfer`). Then call it:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Explain a GPU in one sentence."}],
    "max_tokens": 256
  }'
```

## Memory & stability (why the defaults look the way they do)

The upstream GB10 config OOM-crashed this 32 GB/62 GB box. The `docker-compose.yml`
defaults here fix that:

| Setting | Value | Why |
|---|---|---|
| `mem_limit` / `memswap_limit` | `52g` | **Host-crash safety net.** With no cap, a vLLM memory spike triggers the *host* OOM-killer and locks up the machine. Capping makes the *container* OOM itself instead; ~10 GB stays free for the OS. |
| `ENFORCE_EAGER` | `1` | Skip CUDA-graph capture — big RAM saving + much faster startup (slightly lower throughput). |
| `LIMIT_MM` | `{"image":0,"video":0}` | **Text-only.** Vision-encoder profiling at startup needs ~50 GB RAM and OOMs. Drop this (or raise images) only if you have headroom. |
| `MAX_JOBS` / `FLASHINFER_NVCC_THREADS` | `2` / `1` | The GDN linear-attention kernels JIT-compile on first run; ninja's default one-job-per-core spawns many multi-GB compilers and OOMs. This caps it (one-time cost, then cached). |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | Bounds prefill/profiling batch memory. |
| `MAX_NUM_SEQS` | `64` | Max concurrent sequences (bounds memory; plenty for high concurrency). |
| `MAX_LEN` | `8192` | Context length. KV-cache room is tight after ~24 GB of weights on a 32 GB card; raise cautiously. |
| `GPU_UTIL` | `0.9` | ~28.8 GB of the 32 GB VRAM (~24 GB weights + ~4 GB KV). |

`restart` is set to `"no"` while validating — switch to `always` once stable
**and** only after nouveau is permanently blacklisted (otherwise a crash-boot can
loop). All of the above are env-overridable (see `serve.sh`).

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

## Run on the host (without Docker)

```bash
uv sync           # creates .venv and installs vLLM + deps
ENFORCE_EAGER=1 LIMIT_MM='{"image":0,"video":0}' MAX_JOBS=2 MAX_LEN=8192 ./serve.sh
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
- It's a reasoning model — pass `chat_template_kwargs={"enable_thinking": false}`
  to suppress chain-of-thought in responses.

## Concurrency

vLLM does continuous batching — just fire N requests in parallel at the one
server and they're batched together. There is **no hard 8-request limit**;
concurrency is bounded by `--max-num-seqs` (64 here) and KV-cache size. Measured
on the RTX 5090 (text-only, eager, `MAX_LEN=8192`):

| Concurrency | Result | Aggregate throughput |
|---|---|---|
| 16 | 16/16 ✅ | ~107 tok/s |
| 32 | 32/32 ✅ | ~145 tok/s |
| 48 | 48/48 ✅ | ~146 tok/s |

Throughput plateaus near 48 in eager mode; all requests still succeed.

## Repository layout

| File | Purpose |
|---|---|
| `serve.sh` | Launches `vllm serve` with RTX 5090-tuned, env-overridable flags |
| `docker-compose.yml` | Service with GPU passthrough, HF-cache mount, memory caps |
| `Dockerfile` | CUDA 13 devel base + uv-managed Python + vLLM |
| `test_request.py` | Minimal single-request client |
| `test_parallel.py` | Concurrency / aggregate-throughput test |
| `pyproject.toml` / `uv.lock` | Pinned dependencies |
| `BENCHMARKS.md` | Throughput measurements (original GB10 numbers) |

## License

Code here is provided as-is. The model is under its own license (Apache-2.0 for
Qwen3.6-27B); review the model card before use.
