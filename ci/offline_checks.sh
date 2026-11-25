#!/bin/sh
# CI functional checks that need no GPU:  sh ci/offline_checks.sh <image>
set -eu
cd "$(dirname "$0")/.."
IMAGE=$1

want=$(grep -A1 '^name = "vllm"$' uv.lock | sed -n 's/^version = "\(.*\)"/\1/p')
echo "== the image has vLLM $want (uv.lock) and its launcher"
docker run --rm --entrypoint python "$IMAGE" -c "
import vllm, transformers
assert vllm.__version__ == '$want', vllm.__version__
print('vllm', vllm.__version__, 'transformers', transformers.__version__)"
docker run --rm --entrypoint bash "$IMAGE" -n /app/serve.sh

echo "== compose files are valid"
docker compose -f docker-compose.yml config -q
docker compose -f docker-compose.qwen3.6.yml config -q

echo "== monitoring config"
docker run --rm --entrypoint promtool -v "$PWD/monitoring:/m:ro" prom/prometheus:v3.5.0 check config /m/prometheus.yml
for f in monitoring/grafana/dashboards/*.json; do python3 -m json.tool "$f" >/dev/null || { echo "bad JSON: $f"; exit 1; }; done

echo "== the stall watchdog"
sh ci/test_watchdog.sh
echo "all offline checks passed"
