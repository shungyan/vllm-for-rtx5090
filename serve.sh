#!/usr/bin/env bash
# Launch the vLLM OpenAI-compatible API server for Qwen/Qwen3.8-27B.
#
# Also serves Qwen3.6-27B unchanged — every knob below is env-driven, so the
# pinned Qwen3.6 stack (docker-compose.qwen3.6.yml) just overrides MODEL et al.
#
# Once running, POST to:  http://localhost:8000/v1/chat/completions
# (OpenAI-compatible).  See test_request.py / the curl example in README.md.
set -euo pipefail

cd "$(dirname "$0")"

# --- tunables (override via env) ---------------------------------------------
# NVFP4 4-bit (compressed-tensors) — ~22.6GB on disk, Blackwell tensor-core
# accelerated. vLLM auto-detects the quant from the checkpoint's config.json.
# NOTE: this checkpoint quantizes lm_head to FP8, which vLLM 0.22.1 cannot load
# (it builds an UNquantized ParallelLMHead). Run patch_lmhead_bf16.py once
# against the HF cache before first boot — see that script's docstring.
MODEL="${MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
# PIN THE REVISION. The lm_head fix below is applied to ONE cached snapshot dir.
# When upstream pushes a new commit, HF creates a NEW snapshot dir that symlinks
# the same (unpatched) blobs, vLLM resolves to it, and the crash comes straight
# back. Pinning keeps it on the snapshot that was actually patched. To move to a
# newer revision: bump this, then re-run patch_lmhead_bf16.py against it.
REVISION="${REVISION:-}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b}"   # the "model" string you pass in POST bodies
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Native ctx is 262144; that needs a huge KV cache. Start at 32768; raise if needed.
MAX_LEN="${MAX_LEN:-32768}"
# GB10 has UNIFIED memory shared with the OS, so don't grab everything.
# (On a dedicated-VRAM card like the RTX 5090's 32GB, 0.9 reserves ~29GB; note the
#  NVFP4 weights are actually ~22GB here — ~23.8GB once lm_head is de-quantized to
#  BF16 — leaving only ~5GB of that for the KV cache.)
GPU_UTIL="${GPU_UTIL:-0.9}"
# CUDA-graph capture (default) costs a lot of extra host+device RAM and a long,
# silent startup phase. On a memory-constrained host (e.g. a 62GB box, vs the
# GB10's 121GB) set ENFORCE_EAGER=1 to skip it — slightly lower throughput, far
# less memory and a much faster, safer startup.
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
# Optional cap on concurrent sequences (vLLM default is 256).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
# MTP (multi-token prediction) speculative decoding — see the block below.
SPEC_DECODE="${SPEC_DECODE:-}"
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

# Pin to a specific HF revision (commit sha) when set — see REVISION above.
if [ -n "$REVISION" ]; then
  args+=(--revision "$REVISION" --tokenizer-revision "$REVISION")
fi

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
# Cap vision input size. Startup profiles the encoder at the model's MAX image
# feature size (~16384 tokens, ~50GB RAM); capping max_pixels shrinks that
# profiling (and per-request memory) dramatically — needed to run vision (OCR)
# on a memory-constrained host. e.g. '{"max_pixels": 1605632}'.
[ -n "${MM_PROCESSOR_KWARGS:-}" ] && args+=(--mm-processor-kwargs "$MM_PROCESSOR_KWARGS")

# Tool / function calling (OpenAI-compatible). Set TOOL_CALL_PARSER to enable it.
# Qwen3.6 and Qwen3.8 both emit the XML <tool_call><function=name><parameter=...>
# format, parsed by 'qwen3_xml' (NOT 'hermes', which expects JSON).
[ -n "${TOOL_CALL_PARSER:-}" ] && args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")

# Reasoning ("thinking") output. Qwen3.8 thinks by default, and unlike Qwen3.6 its
# chat template no longer splits <think>...</think> back out of an assistant
# message's `content` when replaying history — it only reads `reasoning_content`.
# So serve it with the qwen3 reasoning parser: thinking lands in `reasoning_content`
# and `content` carries just the final answer. Leave empty for the old Qwen3.6
# behaviour (raw <think> tags inline in `content`).
[ -n "${REASONING_PARSER:-}" ] && args+=(--reasoning-parser "$REASONING_PARSER")

# Server-side defaults for chat-template variables, merged with (and overridable
# by) each request's `chat_template_kwargs`. Two that matter on Qwen3.8:
#   preserve_thinking  — defaults to TRUE in its template, which re-feeds every
#                        past <think> block into each prompt; false keeps
#                        multi-turn / tool-loop prompts small.
#   reasoning_effort   — 'xhigh' (template default), 'medium' or 'low'. Drop to
#                        medium/low if the extra thinking costs too much latency.
[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ] && args+=(--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS")

# MTP (multi-token prediction) speculative decoding. Qwen3.8 ships trained MTP
# weights (model_mtp.safetensors, ~0.85GB, listed in the safetensors index so
# vLLM loads them with the rest). The draft model is the target model itself, so
# only the method + token count are needed. Costs VRAM that would otherwise be
# KV cache — raise SPEC_NUM_TOKENS only if there's headroom.
# The draft model is the target model, but vLLM does NOT inherit --revision into the
# speculative config (SpeculativeConfig.revision defaults to None and is passed
# straight to the draft's ModelConfig). Unpinned, the draft resolves to the newest
# snapshot — the unpatched one — and dies on 'lm_head.weight_scale' in Qwen3_5MTP
# AFTER the target model has already loaded. So pin the draft to the same revision.
case "${SPEC_DECODE,,}" in
  1|true|yes)
    spec="{\"method\": \"${SPEC_METHOD:-mtp}\", \"num_speculative_tokens\": ${SPEC_NUM_TOKENS:-1}"
    [ -n "$REVISION" ] && spec="$spec, \"revision\": \"$REVISION\""
    spec="$spec}"
    args+=(--speculative-config "$spec")
    ;;
esac

# Prefix caching. vLLM leaves it OFF by default for hybrid (GDN/Mamba) models like
# Qwen3.8, so every agent round re-prefilled the whole conversation. "align" mode
# caches the GDN state at block boundaries. Uses the existing KV pool, no extra VRAM.
case "${PREFIX_CACHING,,}" in
  1|true|yes) args+=(--enable-prefix-caching --mamba-cache-mode "${MAMBA_CACHE_MODE:-align}") ;;
esac

[ -n "$API_KEY" ] && args+=(--api-key "$API_KEY")

echo "Starting vLLM: model=$MODEL served-as=$SERVED_NAME  http://$HOST:$PORT"
# Host: `uv run vllm`. In the container the venv is on PATH, so VLLM_LAUNCHER=vllm.
exec ${VLLM_LAUNCHER:-uv run vllm} serve "${args[@]}" "$@"
