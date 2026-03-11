#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_eval_lmdeploy
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH -o SDAR_eval.out
#SBATCH -e SDAR_eval.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tom.ee13@nycu.edu.tw

set -euo pipefail

export PATH="/work/tom900908/sdar_eval/bin:${PATH}"
export PYTHONNOUSERSITE=1
cd /work/tom900908/SDAR/evaluation/opencompass

export SDAR_MODEL_ROOT="/work/tom900908/SDAR/Models"
export SDAR_MODEL_NAME="SDAR-1.7B-Chat"
export SDAR_EVAL_SCOPE="gsm8k"
export SDAR_EVAL_GPUS="2"
export SDAR_INFER_BATCH_SIZE="1"
export SDAR_CONFIDENCE_THRESHOLD="0.95"
export SDAR_BLOCK_LENGTH="4"
export SDAR_MAX_NEW_TOKENS="1024"
export SDAR_WORK_DIR="./outputs/eval-chat-sdar"

export HF_HOME="${SDAR_MODEL_ROOT}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$SDAR_MODEL_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1200
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

/work/tom900908/sdar_eval/bin/python run.py configs/eval_sdar_lmdeploy.py
