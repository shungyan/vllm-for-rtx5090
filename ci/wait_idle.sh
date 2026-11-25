#!/bin/sh
# Block until nothing is using vLLM: no request running or queued, and no AMC tuning run. A down server counts as idle.
# ponytail: check-then-act — a request that arrives in the seconds before the swap is cut off.
vllm_busy() { curl -sf -m 10 http://localhost:8000/metrics | awk '/^vllm:num_requests_(running|waiting)\{/{s+=$2} END{exit !(s>0)}'; }
amc_busy() { curl -sf -m 10 http://localhost:8096/api/batch/status | grep -q '"status": *"running"'; }
while vllm_busy || amc_busy; do
  echo "vLLM in use (a request or an AMC tuning run) — waiting"; sleep 30
done
echo "idle"
