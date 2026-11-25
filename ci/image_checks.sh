#!/bin/sh
# CD gate before the swap: the built image has the vLLM in uv.lock and a valid launcher.  sh ci/image_checks.sh <image>
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
