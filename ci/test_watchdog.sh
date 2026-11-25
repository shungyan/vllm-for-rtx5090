#!/bin/sh
# watchdog.sh must restart vLLM when tokens stop with a request running, and leave it alone while tokens flow.
# Runs it in its real image with fake wget (the metrics) and fake docker (records the restart).
set -eu
cd "$(dirname "$0")/.."
run() {
  docker run --rm --init -e MODE="$1" -e INTERVAL=1 -e STALL_CHECKS=3 \
    -e PATH=/fake:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -v "$PWD/watchdog.sh:/watchdog.sh:ro" -v "$PWD/ci/fake:/fake:ro" \
    --entrypoint timeout docker:27-cli 8 sh /watchdog.sh 2>&1 || true
}
run stuck | grep -q "FAKE docker restart vllm-qwen38" || { echo "FAIL: no restart on a stall"; exit 1; }
! run flowing | grep -q "FAKE docker" || { echo "FAIL: restarted while tokens were flowing"; exit 1; }
echo "watchdog: restarts on a stall, not while generating"
