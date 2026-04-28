#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_hfcmp
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/test_gap_hf_compare_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/test_gap_hf_compare_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

REPO_ROOT="/work/leotsia0416/projects/SDAR"
CHECKPOINT_PATH="${SDAR_COMPARE_MODEL_PATH:-${REPO_ROOT}/checkpoint/training_154151/checkpoint-150}"
REMASK_THRESHOLD="${SDAR_COMPARE_REMASK_THRESHOLD:-0.3}"
NOREMASK_THRESHOLD="${SDAR_COMPARE_NOREMASK_THRESHOLD:-1.0}"
REMASK_WORK_DIR="${SDAR_COMPARE_REMASK_WORK_DIR:-./outputs/eval-chat-sdar-gap-training_154151-checkpoint-150-hf-remask}"
NOREMASK_WORK_DIR="${SDAR_COMPARE_NOREMASK_WORK_DIR:-./outputs/eval-chat-sdar-gap-training_154151-checkpoint-150-hf-noremask}"

run_case() {
  local label="$1"
  local remask_threshold="$2"
  local work_dir="$3"

  echo "[${label}] model=${CHECKPOINT_PATH}" >&2
  echo "[${label}] remask_threshold=${remask_threshold}" >&2
  echo "[${label}] work_dir=${work_dir}" >&2

  cd "${REPO_ROOT}"
  export SDAR_MODEL_PATH="${CHECKPOINT_PATH}"
  export SDAR_EVAL_SCOPE="gsm8k"
  export SDAR_USE_REMASK="true"
  export SDAR_EVAL_CONFIG="configs/eval_sdar_gap_hf.py"
  export SDAR_REMASK_THRESHOLD="${remask_threshold}"
  export SDAR_WORK_DIR="${work_dir}"
  bash "${REPO_ROOT}/script/test_gap.sh"
}

run_case "hf-remask" "${REMASK_THRESHOLD}" "${REMASK_WORK_DIR}"
run_case "hf-noremask" "${NOREMASK_THRESHOLD}" "${NOREMASK_WORK_DIR}"
