# vLLM OpenAI-compatible server for Qwen/Qwen3.8-27B (NVFP4) on NVIDIA GB10.
#
# Identical to Dockerfile.qwen3.6 — the image is model-agnostic, the model is
# chosen by $MODEL at runtime. The two exist only so each compose file pins its
# own image tag and neither generation's build can clobber the other's.
#
# We need the CUDA *devel* toolkit (nvcc + headers) in the image because
# FlashInfer JIT-compiles the NVFP4 GEMM kernels at runtime. The GPU driver
# (libcuda.so.1) is injected at runtime by the NVIDIA Container Toolkit
# (`--gpus all`). CUDA 13.0.1 matches the host's nvcc 13.0 / torch cu130 wheels.
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# FlashInfer's JIT looks here for nvcc; the base image sets this, but be explicit.
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

# Pinned uv via the official installer (avoids a hard dep on ghcr.io).
RUN curl -LsSf https://astral.sh/uv/0.11.19/install.sh | \
    env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh

ENV UV_PYTHON_PREFERENCE=only-managed \
    UV_FROZEN=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Dependency layer (cached): managed CPython 3.12 + vLLM. Headers included.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project   # UV_FROZEN=1 already pins to uv.lock

# App scripts.
COPY serve.sh test_request.py ./
RUN chmod +x serve.sh

# Put the venv on PATH so serve.sh launches `vllm` directly (no uv at runtime).
ENV PATH="/app/.venv/bin:${PATH}" \
    VLLM_LAUNCHER=vllm

EXPOSE 8000
CMD ["./serve.sh"]
