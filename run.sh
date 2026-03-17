
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

# Keep tokenizer threads from oversubscribing CPU when using torchrun.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export SKIP_ERR="${SKIP_ERR:-1}"

WORK_DIR="${WORK_DIR:-/m2v_intern/xuboshen/zgw/eval}"
CONFIG_PATH="$(dirname "$0")/configs/qwen3_vl_4b_aot_fast.json"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l | tr -d ' ')}"

# Fast AoT path for Qwen3-VL-4B:
# 1. torchrun for sample-level parallelism across GPUs
# 2. transformers backend instead of vLLM TP
# 3. short decoding budget for MCQ ("option letter only")
torchrun --nproc-per-node="${NPROC_PER_NODE}" \
  run.py \
  --config "${CONFIG_PATH}" \
  --work-dir "${WORK_DIR}" \
  --reuse
