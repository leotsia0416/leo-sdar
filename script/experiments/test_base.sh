#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_eval_base_hf
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/test_base_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/test_base_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

REPO_ROOT="/work/leotsia0416/projects/SDAR"
EVAL_ENV_PREFIX="${EVAL_ENV_PREFIX:-/work/leotsia0416/sdar_eval}"

export PATH="${EVAL_ENV_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
cd "${REPO_ROOT}/evaluation/opencompass"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

resolve_latest_training_root() {
  if [[ ! -d "${REPO_ROOT}/checkpoint" ]]; then
    return 0
  fi
  find "${REPO_ROOT}/checkpoint" -mindepth 1 -maxdepth 1 -type d -name 'training_*' | sort -V | tail -n 1
}

resolve_latest_checkpoint_dir() {
  local checkpoint_root="${1:-}"
  if [[ -z "${checkpoint_root}" || ! -d "${checkpoint_root}" ]]; then
    return 0
  fi
  find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

has_model_weights() {
  local model_dir="${1:?missing model dir}"
  [[ -e "${model_dir}/model.safetensors" ]] && return 0
  [[ -e "${model_dir}/model.safetensors.index.json" ]] && return 0
  [[ -e "${model_dir}/pytorch_model.bin" ]] && return 0
  [[ -e "${model_dir}/pytorch_model.bin.index.json" ]] && return 0
  compgen -G "${model_dir}/model-*.safetensors" > /dev/null && return 0
  return 1
}

validate_hf_model_dir() {
  local model_dir="${1:?missing model dir}"
  if [[ ! -d "${model_dir}" ]]; then
    echo "Resolved model path does not exist: ${model_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${model_dir}/config.json" ]]; then
    echo "Resolved model path is missing config.json: ${model_dir}" >&2
    exit 1
  fi
  if ! has_model_weights "${model_dir}"; then
    echo "Resolved model path is missing model weights: ${model_dir}" >&2
    exit 1
  fi
}

DEFAULT_CHECKPOINT_ROOT="$(resolve_latest_training_root)"
DEFAULT_MODEL_PATH="$(resolve_latest_checkpoint_dir "${DEFAULT_CHECKPOINT_ROOT}")"
if [[ -z "${DEFAULT_MODEL_PATH}" ]]; then
  DEFAULT_MODEL_PATH="${REPO_ROOT}/Models/SDAR-1.7B-Chat-"
fi

export SDAR_MODEL_PATH="${SDAR_MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
validate_hf_model_dir "${SDAR_MODEL_PATH}"
export SDAR_MODEL_ROOT="${SDAR_MODEL_ROOT:-$(dirname "${SDAR_MODEL_PATH}")}"
export SDAR_MODEL_NAME="${SDAR_MODEL_NAME:-$(basename "${SDAR_MODEL_PATH}")}"
export SDAR_EVAL_GPUS="${SDAR_EVAL_GPUS:-1}"
export SDAR_INFER_BATCH_SIZE="${SDAR_INFER_BATCH_SIZE:-1}"
export SDAR_CONFIDENCE_THRESHOLD="${SDAR_CONFIDENCE_THRESHOLD:-1.0}"
export SDAR_BLOCK_LENGTH="${SDAR_BLOCK_LENGTH:-4}"
export SDAR_MAX_NEW_TOKENS="${SDAR_MAX_NEW_TOKENS:-1024}"
export SDAR_TEMPERATURE="${SDAR_TEMPERATURE:-0.0}"
export SDAR_TOP_K="${SDAR_TOP_K:-1}"
export SDAR_TOP_P="${SDAR_TOP_P:-1.0}"
export SDAR_TORCH_DTYPE="${SDAR_TORCH_DTYPE:-bfloat16}"
export SDAR_WORK_DIR="${SDAR_WORK_DIR:-./outputs/eval-chat-sdar}"

export HF_HOME="${SDAR_MODEL_ROOT}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${SDAR_MODEL_ROOT}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1200
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

echo "Using base model path: ${SDAR_MODEL_PATH}" >&2
echo "Base decode threshold: ${SDAR_CONFIDENCE_THRESHOLD}" >&2
echo "Infer batch size: ${SDAR_INFER_BATCH_SIZE}" >&2

"${EVAL_ENV_PREFIX}/bin/python" run.py configs/eval_sdar_hf.py
