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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SKIP_ERR="${SKIP_ERR:-1}"

WORK_DIR="${WORK_DIR:-/m2v_intern/xuboshen/zgw/VideoProxyMixed/evaluation}"
REUSE="${REUSE:-0}"

# ---------------------------------------------------------------------------
# Offline mode: use local HF cache, disable all network access.
# ---------------------------------------------------------------------------
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=0
export HF_HOME="${HF_HOME:-/m2v_intern/xuboshen/zgw/hf_cache_temp}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# ---------------------------------------------------------------------------
# Local dataset overrides (skip HuggingFace download).
# ---------------------------------------------------------------------------
export VIDEO_HOLMES_DIR="${VIDEO_HOLMES_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/Video-Holmes}"
export TIMELENS_DIR="${TIMELENS_DIR:-/m2v_intern/xuboshen/zgw/hf_cache_temp/TimeLens-Bench}"
export PERCEPTION_TEST_DIR="${PERCEPTION_TEST_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/PerceptionTest}"
export MLVU_DIR="${MLVU_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/MLVU_Test}"

# ---------------------------------------------------------------------------
# Multi-GPU data-parallel launch via torchrun.
# ---------------------------------------------------------------------------
NGPU="${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

# ---------------------------------------------------------------------------
# CoT mode: chain-of-thought reasoning with <think>/<answer> tags.
# Strips "answer directly" instructions, appends CoT prompt,
# temperature=0.7, max_new_tokens=2048, extracts <answer> content.
#
# ── Acceleration for 80GB GPUs ──
# Qwen3-VL-4B only uses ~10GB VRAM (tp=1). With 80GB available, we
# significantly increase vLLM concurrency to saturate GPU utilization:
#   VLLM_MAX_NUM_SEQS:   32 (default 8) — 4x more concurrent sequences
#   VLLM_BATCH_CHUNK_SIZE: 64 (default 32) — larger batch chunks
# This lets vLLM overlap KV cache computation across many sequences,
# which is the main lever since CoT generates 2048 tokens per sample.
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen3-VL-4B-Instruct}"
export USE_COT=1
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-64}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"

CMD=(
  torchrun
  --nproc-per-node="${NGPU}"
  --master-addr="${MASTER_ADDR:-127.0.0.1}"
  --master-port="${MASTER_PORT:-29500}"
  run.py
  --use-vllm
  --data
  AoTBench_ReverseFilm_16frame
  AoTBench_UCF101_16frame
  AoTBench_Rtime_t2v_16frame
  AoTBench_Rtime_v2t_16frame
  AoTBench_QA_16frame
  FutureOmni_64frame
  CharadesTimeLens_1fps
  MVBench_MP4_1fps
  PerceptionTest_val_16frame
  --model
  Qwen3-VL-4B-Instruct_aot_ablation_exp1_v2t_binary
  Qwen3-VL-4B-Instruct_aot_ablation_exp2_v2t_3way
  Qwen3-VL-4B-Instruct_aot_ablation_exp3_t2v_binary
  Qwen3-VL-4B-Instruct_aot_ablation_exp4_t2v_3way
  --work-dir "${WORK_DIR}"
)

if [ "${REUSE}" = "1" ]; then
  CMD+=(--reuse)
fi

"${CMD[@]}"
