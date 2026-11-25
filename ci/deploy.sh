#!/bin/sh
# CD: move the live stack to commit $1, rebuild, test on the GPU (ci/live_test.py), roll back on failure.
#   sh ci/deploy.sh <sha>
# Runs from LIVE_DIR (default ~/vllm-for-dgx-spark) with project name vllm-for-dgx-spark: the watchdog and the
# monitoring bind-mount files from that folder, and AMC reaches vLLM over the vllm-for-dgx-spark_default network.
set -eu
SHA=$1
LIVE=${LIVE_DIR:-$HOME/vllm-for-dgx-spark}
IMG=vllm-qwen38-nvfp4
dc() { docker compose -p vllm-for-dgx-spark "$@"; }

cd "$LIVE"
[ "$(git branch --show-current)" = main ] || { echo "$LIVE is not on main — not deploying"; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "$LIVE has uncommitted changes — not deploying"; git status --short; exit 1; }
PREV=$(git rev-parse HEAD)
git fetch -q origin
git merge -q --ff-only "$SHA"

docker tag "$IMG:latest" "$IMG:previous"
dc build vllm                  # CI built this commit already, so the layers come from the cache
sh ci/wait_idle.sh
dc up -d
if sh ci/wait_healthy.sh && python3 ci/live_test.py; then
  echo "deployed $SHA"
  exit 0
fi

echo "live test FAILED — rolling back to $PREV and the previous image"
docker tag "$IMG:previous" "$IMG:latest"
git reset -q --hard "$PREV"    # the compose file goes back too; the next deploy fast-forwards again
dc up -d --no-build
sh ci/wait_healthy.sh && echo "rolled back; vLLM is serving the previous version" || echo "ROLLBACK ALSO FAILED — check docker logs vllm-qwen38"
exit 1
