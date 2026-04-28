#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_hfsweep
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/test_gap_hf_remask_sweep_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/test_gap_hf_remask_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

REPO_ROOT="/work/leotsia0416/projects/SDAR"
CHECKPOINT_PATH="${SDAR_SWEEP_MODEL_PATH:-${REPO_ROOT}/checkpoint/training_154151/checkpoint-150}"
WORK_ROOT="${SDAR_SWEEP_WORK_ROOT:-./outputs/eval-chat-sdar-gap-training_154151-checkpoint-150-hf-remask-sweep}"

run_case() {
  local label="$1"
  local remask_threshold="$2"
  local remask_start_ratio="$3"
  local remask_interval_blocks="$4"
  local remask_window_blocks="$5"
  local work_dir="${WORK_ROOT%/}/${label}"

  echo "[${label}] model=${CHECKPOINT_PATH}" >&2
  echo "[${label}] remask_threshold=${remask_threshold}" >&2
  echo "[${label}] remask_start_ratio=${remask_start_ratio}" >&2
  echo "[${label}] remask_interval_blocks=${remask_interval_blocks}" >&2
  echo "[${label}] remask_window_blocks=${remask_window_blocks}" >&2
  echo "[${label}] work_dir=${work_dir}" >&2

  cd "${REPO_ROOT}"
  export SDAR_MODEL_PATH="${CHECKPOINT_PATH}"
  export SDAR_EVAL_SCOPE="gsm8k"
  export SDAR_USE_REMASK="true"
  export SDAR_EVAL_CONFIG="configs/eval_sdar_gap_hf.py"
  export SDAR_REMASK_THRESHOLD="${remask_threshold}"
  export SDAR_REMASK_START_RATIO="${remask_start_ratio}"
  export SDAR_REMASK_INTERVAL_BLOCKS="${remask_interval_blocks}"
  export SDAR_REMASK_WINDOW_BLOCKS="${remask_window_blocks}"
  export SDAR_WORK_DIR="${work_dir}"
  bash "${REPO_ROOT}/script/test_gap.sh"
}

run_case "rt0_30-rs0_50-ri4-rw2" "0.30" "0.50" "4" "2"
run_case "rt0_30-rs0_70-ri4-rw2" "0.30" "0.70" "4" "2"
run_case "rt0_35-rs0_70-ri4-rw2" "0.35" "0.70" "4" "2"
