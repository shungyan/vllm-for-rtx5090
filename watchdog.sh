#!/bin/sh
# Restart vLLM when it is STALLED: a request is running but no token has been generated for
# STALL_CHECKS checks in a row. /health stays 200 through this hang (see 2026-11-19), so
# neither the healthcheck nor `restart: always` ever fires. Runs in the vllm-watchdog sidecar.
URL="${METRICS_URL:-http://vllm:8000/metrics}"
INTERVAL="${INTERVAL:-30}"; STALL_CHECKS="${STALL_CHECKS:-3}"; TARGET="${TARGET:-vllm-qwen38}"
prev=""; same=0
while sleep "$INTERVAL"; do
  m=$(wget -qO- -T 10 "$URL") || { prev=""; same=0; continue; }      # loading / restarting
  run=$(echo "$m" | awk '/^vllm:num_requests_running\{/{print int($2)}')
  tok=$(echo "$m" | awk '/^vllm:generation_tokens_total\{/{print $2}')
  if [ "${run:-0}" -gt 0 ] && [ "$tok" = "$prev" ]; then same=$((same + 1)); else same=0; fi
  prev=$tok
  if [ "$same" -ge "$STALL_CHECKS" ]; then
    echo "$(date -Iseconds) STALL: $run running, generation_tokens_total stuck at $tok for $((same * INTERVAL)) s -> docker restart $TARGET"
    docker restart "$TARGET"; prev=""; same=0
  fi
done
