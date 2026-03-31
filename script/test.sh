#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_eval_lmdeploy
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/test_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

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

export PATH="/work/leotsia0416/sdar_eval/bin:${PATH}"
export PYTHONNOUSERSITE=1
cd /work/leotsia0416/projects/SDAR/evaluation/opencompass
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export SDAR_MODEL_PATH="/work/leotsia0416/projects/SDAR/Models/SDAR-1.7B-Chat-"
validate_hf_model_dir "${SDAR_MODEL_PATH}"
export SDAR_MODEL_ROOT="$(dirname "${SDAR_MODEL_PATH}")"
export SDAR_MODEL_NAME="$(basename "${SDAR_MODEL_PATH}")"
export SDAR_EVAL_SCOPE="gsm8k"
export SDAR_EVAL_GPUS="2"
export SDAR_INFER_BATCH_SIZE="1"
export SDAR_CONFIDENCE_THRESHOLD="0.95"
export SDAR_BLOCK_LENGTH="4"
export SDAR_MAX_NEW_TOKENS="1536"
export SDAR_WORK_DIR="./outputs/eval-chat-sdar"

export HF_HOME="${SDAR_MODEL_ROOT}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$SDAR_MODEL_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1200
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

/work/leotsia0416/sdar_eval/bin/python run.py configs/eval_sdar_lmdeploy.py
