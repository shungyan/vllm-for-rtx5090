#!/bin/sh
# Wait up to 20 min for vllm-qwen38 to turn healthy (model load + CUDA graphs; start_period is 600 s).
for _ in $(seq 1 120); do
  s=$(docker inspect -f '{{.State.Health.Status}}' vllm-qwen38 2>/dev/null || echo missing)
  [ "$s" = healthy ] && { echo "vllm-qwen38 healthy"; exit 0; }
  [ "$s" = unhealthy ] && break
  sleep 10
done
echo "vllm-qwen38 not healthy ($s)"; docker logs --tail 80 vllm-qwen38 2>&1; exit 1
