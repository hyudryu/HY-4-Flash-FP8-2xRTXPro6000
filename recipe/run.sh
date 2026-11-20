#!/usr/bin/env bash
# Serve tencent/Hy4-flash-FP8 on 2x RTX PRO 6000 Blackwell (SM120) with vLLM.
# The stock vllm/vllm-openai:hy4-flash image only supports SM90/SM100 for this
# model; the three files in ../patch are bind-mounted read-only over the image
# to add an SM120 path (FlashInfer SM120 sparse MLA with attention sinks).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
P="$REPO/patch/vllm/models"
D=/usr/local/lib/python3.12/dist-packages/vllm/models
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:6b645bfbf4b89602d38f6e93c1c418516ec0bfbf54b64cd661b8075bfec01da1}  # tag hy4-flash
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
PORT=${PORT:-8000}

docker rm -f hy4-flash-fp8 >/dev/null 2>&1 || true
docker run -d --name hy4-flash-fp8 --gpus all --ipc=host --shm-size 16g -p "$PORT:8000" \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_RUST_FRONTEND=1 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$P/hy_v4_flash/nvidia/sm120.py:$D/hy_v4_flash/nvidia/sm120.py:ro" \
  -v "$P/hy_v4_flash/nvidia/attention.py:$D/hy_v4_flash/nvidia/attention.py:ro" \
  -v "$P/hy_v4/nvidia/attention.py:$D/hy_v4/nvidia/attention.py:ro" \
  --entrypoint vllm "$IMAGE" \
  serve tencent/Hy4-flash-FP8 --revision bd0e136e89a9ecd5ea17668cd89cf54fa127e795 \
    --host 0.0.0.0 --port 8000 --served-model-name hy4-flash \
    --tensor-parallel-size 2 --dtype bfloat16 --gpu-memory-utilization 0.95 \
    --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --speculative-config '{"num_speculative_tokens":5,"method":"dfly"}' \
    --enable-auto-tool-choice --reasoning-parser hy_v4 --tool-call-parser hy_v4 \
    --enable-prompt-tokens-details "$@"

echo "Started. Follow logs with: docker logs -f hy4-flash-fp8   (weights load ~3 min, ready in ~8-9 min)"
