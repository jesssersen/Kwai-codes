
#!/bin/bash
set -euo pipefail
set -x

# Disable torch compile to avoid startup overhead.
export TORCH_COMPILE_DISABLE=1

# Resolve libcuda lookup issues on some nodes.
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# Force system toolchain for extension builds.
unset CC
unset CXX
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Keep tokenizer threads from oversubscribing CPU.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export SKIP_ERR="${SKIP_ERR:-1}"

WORK_DIR="${WORK_DIR:-/m2v_intern/xuboshen/zgw/eval_2gpu}"
REUSE="${REUSE:-0}"

# ---------------------------------------------------------------------------
# Offline mode: use local HF cache, disable all network access.
# ---------------------------------------------------------------------------
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-/m2v_intern/xuboshen/zgw/hf_cache_temp}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# ---------------------------------------------------------------------------
# Multi-GPU data-parallel launch via torchrun.
#
# Each rank loads a full copy of the model on its own GPU and processes
# 1/NGPU of the dataset (VLMEvalKit shards by rank automatically).
# This is faster than vLLM tensor-parallel for small models (< 8B) because
# there is zero inter-GPU communication during inference.
#
# NGPU: how many GPUs to use (default: all visible).
# ---------------------------------------------------------------------------
NGPU="${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

CMD=(
  torchrun
  --nproc-per-node="${NGPU}"
  --master-port="${MASTER_PORT:-29500}"
  run.py
  --data
  LongVideoBench_1fps
  MVBench_MP4_1fps
  --model
  Qwen3-VL-4B-Instruct
  --work-dir "${WORK_DIR}"
)

if [ "${REUSE}" = "1" ]; then
  CMD+=(--reuse)
fi

"${CMD[@]}"
