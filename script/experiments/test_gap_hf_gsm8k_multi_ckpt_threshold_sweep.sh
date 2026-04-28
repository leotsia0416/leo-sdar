#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_gsmmulti
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/test_gap_hf_gsm8k_multi_ckpt_threshold_sweep_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/test_gap_hf_gsm8k_multi_ckpt_threshold_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

REPO_ROOT="/work/leotsia0416/projects/SDAR"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

CHECKPOINT_ROOT="${SDAR_GSM8K_MULTI_CKPT_ROOT:-${REPO_ROOT}/checkpoint/training_173120}"
SELECTED_CKPTS="${SDAR_GSM8K_MULTI_CKPTS:-checkpoint-250 checkpoint-300 checkpoint-350 checkpoint-375}"
THRESHOLDS="${SDAR_GSM8K_MULTI_THRESHOLDS:-0.85 0.90 0.95 1.00}"
WORK_ROOT="${SDAR_GSM8K_MULTI_WORK_ROOT:-${REPO_ROOT}/outputs/gsm8k_threshold_sweep_training_173120_ckpt250300350375_rt085090095100_${RUN_STAMP}}"

echo "checkpoint_root=${CHECKPOINT_ROOT}" >&2
echo "selected_ckpts=${SELECTED_CKPTS}" >&2
echo "thresholds=${THRESHOLDS}" >&2
echo "work_root=${WORK_ROOT}" >&2

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "Missing checkpoint root: ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

mkdir -p "${WORK_ROOT}"

for ckpt_name in ${SELECTED_CKPTS}; do
  checkpoint_path="${CHECKPOINT_ROOT}/${ckpt_name}"
  checkpoint_work_root="${WORK_ROOT}/${ckpt_name}"

  if [[ ! -d "${checkpoint_path}" ]]; then
    echo "Missing checkpoint path: ${checkpoint_path}" >&2
    exit 1
  fi

  echo "[${ckpt_name}] start GSM8K threshold sweep" >&2
  echo "[${ckpt_name}] checkpoint_path=${checkpoint_path}" >&2
  echo "[${ckpt_name}] work_root=${checkpoint_work_root}" >&2

  SDAR_GSM8K_SWEEP_MODEL_PATH="${checkpoint_path}" \
  SDAR_GSM8K_SWEEP_THRESHOLDS="${THRESHOLDS}" \
  SDAR_GSM8K_SWEEP_WORK_ROOT="${checkpoint_work_root}" \
  bash "${REPO_ROOT}/script/experiments/test_gap_hf_gsm8k_full_threshold_sweep.sh"

  echo "[${ckpt_name}] finished GSM8K threshold sweep" >&2
done

echo "Completed multi-checkpoint GSM8K threshold sweep." >&2
echo "Output root: ${WORK_ROOT}" >&2
