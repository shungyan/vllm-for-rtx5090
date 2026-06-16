#!/usr/bin/env bash
# Launch the vLLM OpenAI-compatible API server for Qwen/Qwen3.6-27B.
#
# Once running, POST to:  http://localhost:8000/v1/chat/completions
# (OpenAI-compatible).  See test_request.py / the curl example in README.md.
set -euo pipefail

cd "$(dirname "$0")"

# --- tunables (override via env) ---------------------------------------------
# NVFP4 4-bit (compressed-tensors) — ~14GB, Blackwell tensor-core accelerated.
# vLLM auto-detects the quant from the checkpoint's config.json.
MODEL="${MODEL:-unsloth/Qwen3.6-27B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-qwen3.6-27b}"   # the "model" string you pass in POST bodies
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Native ctx is 262144; that needs a huge KV cache. Start at 32768; raise if needed.
MAX_LEN="${MAX_LEN:-32768}"
# GB10 has UNIFIED memory shared with the OS, so don't grab everything.
# (On a dedicated-VRAM card like the RTX 5090's 32GB, 0.9 reserves ~29GB; note the
#  NVFP4 weights are actually ~24GB here, leaving only ~4GB of that for KV cache.)
GPU_UTIL="${GPU_UTIL:-0.9}"
# CUDA-graph capture (default) costs a lot of extra host+device RAM and a long,
# silent startup phase. On a memory-constrained host (e.g. a 62GB box, vs the
# GB10's 121GB) set ENFORCE_EAGER=1 to skip it — slightly lower throughput, far
# less memory and a much faster, safer startup.
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
# Optional cap on concurrent sequences (vLLM default is 256).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
API_KEY="${VLLM_API_KEY:-}"                  # optional bearer token; empty = open

# HF hub 1.x uses hf-xet for fast parallel downloads automatically.

# FlashInfer JIT-compiles the NVFP4 GEMM kernels at runtime and needs `nvcc`.
# In the Docker image that comes from the CUDA *devel* base ($CUDA_HOME=/usr/local/cuda).
# On a bare host with no system CUDA toolkit, fall back to the complete toolkit
# (nvcc/ptxas/headers/libs) that the torch cu13 wheel bundles inside the venv.
if [ -z "${CUDA_HOME:-}" ] && ! command -v nvcc >/dev/null 2>&1 && [ ! -d /usr/local/cuda ]; then
  bundled_cuda="$(ls -d "$PWD"/.venv/lib/python*/site-packages/nvidia/cu13 2>/dev/null | head -1)"
  if [ -n "$bundled_cuda" ] && [ -x "$bundled_cuda/bin/nvcc" ]; then
    export CUDA_HOME="$bundled_cuda"
    export PATH="$bundled_cuda/bin:$PATH"
    # The torch wheel bundles a newer nvcc (13.3) than its cudart headers (13.0).
    # CCCL's strict equality check rejects that, so disable it — same CUDA major,
    # compiler newer than headers, ABI-compatible. (Not needed with the Docker
    # devel image, whose nvcc + headers are a matched set.)
    export FLASHINFER_EXTRA_CUDAFLAGS="${FLASHINFER_EXTRA_CUDAFLAGS:-} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
    echo "Using torch-bundled CUDA toolkit for FlashInfer JIT: $CUDA_HOME"
  fi
fi

args=(
  --model "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --dtype bfloat16
)

# Cap images per request so the multimodal preprocessor doesn't over-allocate.
# Startup profiles the vision tower with a max-resolution dummy image, which can
# need 50GB+ host RAM. Set LIMIT_MM image:0 on a memory-constrained host to run
# text-only and skip that profiling entirely.
# (Assign the default separately — a brace-containing default inside ${VAR:-...}
#  breaks bash brace matching and duplicates the trailing '}'.)
LIMIT_MM="${LIMIT_MM:-}"
[ -z "$LIMIT_MM" ] && LIMIT_MM='{"image":4,"video":0}'
args+=(--limit-mm-per-prompt "$LIMIT_MM")

# Memory-saving knobs (opt-in via env) for hosts smaller than the GB10.
case "${ENFORCE_EAGER,,}" in 1|true|yes) args+=(--enforce-eager) ;; esac
[ -n "$MAX_NUM_SEQS" ] && args+=(--max-num-seqs "$MAX_NUM_SEQS")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")

[ -n "$API_KEY" ] && args+=(--api-key "$API_KEY")

echo "Starting vLLM: model=$MODEL served-as=$SERVED_NAME  http://$HOST:$PORT"
# Host: `uv run vllm`. In the container the venv is on PATH, so VLLM_LAUNCHER=vllm.
exec ${VLLM_LAUNCHER:-uv run vllm} serve "${args[@]}" "$@"
