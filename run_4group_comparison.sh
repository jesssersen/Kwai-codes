#!/bin/bash
# ==========================================================================
# run_4group_comparison.sh — Compare Qwen3-VL-4B across 4 configurations:
#
#   Group 1: No CoT + temperature=0   (greedy, direct answer)
#   Group 2: No CoT + temperature=0.7 (sampling, direct answer)
#   Group 3: CoT    + temperature=0   (greedy, chain-of-thought)
#   Group 4: CoT    + temperature=0.7 (sampling, chain-of-thought)
#
# All groups run serially. Logs are written to $LOG_DIR.
#
# Usage:
#   bash run_4group_comparison.sh
# ==========================================================================
set -uo pipefail
set -x

# ---------------------------------------------------------------------------
# Environment setup (same as run_vllm.sh)
# ---------------------------------------------------------------------------
export TORCH_COMPILE_DISABLE=1
export VLLM_HOST_IP=127.0.0.1

export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

unset CC
unset CXX
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SKIP_ERR="${SKIP_ERR:-1}"

REUSE="${REUSE:-0}"
NGPU="${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
DELAY="${DELAY:-15}"

# ---------------------------------------------------------------------------
# HF / offline settings
# ---------------------------------------------------------------------------
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=0
export HF_HOME="${HF_HOME:-/m2v_intern/xuboshen/zgw/hf_cache_temp}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# ---------------------------------------------------------------------------
# Local dataset overrides
# ---------------------------------------------------------------------------
export VIDEO_HOLMES_DIR="${VIDEO_HOLMES_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/Video-Holmes}"
export TIMELENS_DIR="${TIMELENS_DIR:-/m2v_intern/xuboshen/zgw/hf_cache_temp/TimeLens-Bench}"
export PERCEPTION_TEST_DIR="${PERCEPTION_TEST_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/PerceptionTest}"
export MLVU_DIR="${MLVU_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/MLVU_Test}"

# ---------------------------------------------------------------------------
# vLLM settings
# ---------------------------------------------------------------------------
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-64}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

# ---------------------------------------------------------------------------
# Datasets & Model
# ---------------------------------------------------------------------------
DATASETS=(
  MLVU_64frame
  Video_Holmes_64frame
  AoTBench_ReverseFilm_16frame
  AoTBench_UCF101_16frame
  AoTBench_Rtime_t2v_16frame
  AoTBench_Rtime_v2t_16frame
  AoTBench_QA_16frame
  FutureOmni_64frame
  CharadesTimeLens_1fps
  MVBench_MP4_1fps
  PerceptionTest_val_16frame
  Video-MME_64frame
)

MODEL=Qwen3-VL-4B-Instruct

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
WORK_DIR_BASE="${WORK_DIR_BASE:-/m2v_intern/xuboshen/zgw/VideoProxyMixed}"
LOG_DIR="${LOG_DIR:-${WORK_DIR_BASE}/logs_4group}"
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Helper: run one evaluation group
# ---------------------------------------------------------------------------
run_group() {
  local group_name="$1"
  local use_cot="$2"
  local temperature="$3"
  local work_dir="$4"
  local log_file="${LOG_DIR}/${group_name}.log"

  echo ""
  echo "=================================================================="
  echo " [$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${group_name}"
  echo "   USE_COT=${use_cot}  TEMPERATURE=${temperature}"
  echo "   work_dir=${work_dir}"
  echo "   log_file=${log_file}"
  echo "=================================================================="

  export USE_COT="${use_cot}"
  export TEMPERATURE="${temperature}"

  local CMD=(
    python launch_workers.py
    --ngpu "${NGPU}"
    --delay "${DELAY}"
    --
    run.py
    --use-vllm
    --data "${DATASETS[@]}"
    --model "${MODEL}"
    --work-dir "${work_dir}"
  )

  if [ "${REUSE}" = "1" ]; then
    CMD+=(--reuse)
  fi

  # Run and tee to both console and log file; capture exit code
  "${CMD[@]}" 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}

  if [ ${rc} -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ${group_name} (exit code ${rc})" | tee -a "${log_file}"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: ${group_name}" | tee -a "${log_file}"
  fi

  return ${rc}
}

# ---------------------------------------------------------------------------
# Main: run 4 groups serially
# ---------------------------------------------------------------------------
FAILED_GROUPS=()

echo "===== 4-Group Comparison: Qwen3-VL-4B-Instruct ====="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Logs directory: ${LOG_DIR}"
echo ""

# Group 1: No CoT + temperature=0
run_group "noCoT_temp0" 0 0 "${WORK_DIR_BASE}/eval_noCoT_temp0" || \
  FAILED_GROUPS+=("noCoT_temp0")

# Group 2: No CoT + temperature=0.7
run_group "noCoT_temp0.7" 0 0.7 "${WORK_DIR_BASE}/eval_noCoT_temp0.7" || \
  FAILED_GROUPS+=("noCoT_temp0.7")

# Group 3: CoT + temperature=0
run_group "CoT_temp0" 1 0 "${WORK_DIR_BASE}/eval_CoT_temp0" || \
  FAILED_GROUPS+=("CoT_temp0")

# Group 4: CoT + temperature=0.7
run_group "CoT_temp0.7" 1 0.7 "${WORK_DIR_BASE}/eval_CoT_temp0.7" || \
  FAILED_GROUPS+=("CoT_temp0.7")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "===== Summary ====="
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"

if [ ${#FAILED_GROUPS[@]} -eq 0 ]; then
  echo "All 4 groups completed successfully!"
else
  echo "FAILED groups: ${FAILED_GROUPS[*]}"
  echo "Check logs in: ${LOG_DIR}"
  exit 1
fi
