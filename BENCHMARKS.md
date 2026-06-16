# vLLM benchmarks — Qwen/Qwen3.6-27B (NVFP4) on NVIDIA GB10

- **Model:** `unsloth/Qwen3.6-27B-NVFP4` (4-bit NVFP4, ~14 GB), served as `qwen3.6-27b`
- **Engine:** vLLM 0.22.1, FlashInfer NVFP4 GEMM kernel
- **Hardware:** NVIDIA GB10 (Blackwell sm_121), 121 GB unified memory
- **Server config:** `max_model_len=32768`, `gpu_memory_utilization=0.85`, `dtype=bfloat16`
- **Bench method:** `max_tokens=256`, `ignore_eos=true`, `temperature=0.7`, prompt = "Write a detailed essay about the history of computing." (warmup run first)
- **Measured:** 2026-06-09/10

## Single request (interactive)

| Metric | bf16 (baseline) | **NVFP4 (host)** | NVFP4 (in Docker) |
|---|---|---|---|
| Decode throughput | 4.4 tok/s | **9.5 tok/s** | 8.8 tok/s |
| Time-to-first-token | 304 ms | **133 ms** | 154 ms |
| Weights size | ~54 GB | ~14 GB | ~14 GB |

## Aggregate throughput (concurrent requests, NVFP4)

| Concurrency | Aggregate (host) | Per-request (host) | Aggregate (Docker) |
|---|---|---|---|
| 1 | 9.5 tok/s | 9.5 tok/s | 8.8 tok/s |
| 4 | 36.6 tok/s | 9.1 tok/s | — |
| 8 | **69.9 tok/s** | 8.7 tok/s | 62.9 tok/s (7.9/req) |

## Notes

- NVFP4 ≈ **2.2× faster single-stream** than bf16 (9.5 vs 4.4 tok/s); smaller 4-bit
  weights = less memory traffic per token (decode is bandwidth-bound here).
- Scales **near-linearly** with concurrency (per-request barely drops 9.1 → 8.7
  going 4 → 8). ~22× concurrency headroom reported by the engine, so aggregate
  keeps climbing past 8.
- Docker costs ~7–10% vs bare-metal (normal container overhead).
- TensorRT-LLM was evaluated as an alternative but could not serve this model on
  this box (see memory note `trtllm-qwen36-port`); vLLM is the chosen engine.
