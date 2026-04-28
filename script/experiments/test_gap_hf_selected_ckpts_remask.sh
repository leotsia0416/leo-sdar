#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_hfselrm
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/test_gap_hf_selected_ckpts_remask_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/test_gap_hf_selected_ckpts_remask_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

REPO_ROOT="/work/leotsia0416/projects/SDAR"
CHECKPOINT_ROOT="${SDAR_SELECTED_CKPTS_ROOT:-${REPO_ROOT}/checkpoint/training_157757}"
WORK_ROOT="${SDAR_SELECTED_CKPTS_WORK_ROOT:-./outputs/eval-chat-sdar-gap-training_157757-hf-remask-picked-lite}"
SELECTED_CKPTS="${SDAR_SELECTED_CKPTS:-checkpoint-100 checkpoint-300 checkpoint-375}"

REMASK_THRESHOLD="${SDAR_SELECTED_REMASK_THRESHOLD:-0.3}"
REMASK_START_RATIO="${SDAR_SELECTED_REMASK_START_RATIO:-0.0}"
REMASK_START_TOKENS="${SDAR_SELECTED_REMASK_START_TOKENS:-192}"
REMASK_PREFIX_GUARD_TOKENS="${SDAR_SELECTED_REMASK_PREFIX_GUARD_TOKENS:-192}"
REMASK_TAIL_GUARD_BLOCKS="${SDAR_SELECTED_REMASK_TAIL_GUARD_BLOCKS:-1}"
REMASK_INTERVAL_BLOCKS="${SDAR_SELECTED_REMASK_INTERVAL_BLOCKS:-2}"
REMASK_WINDOW_BLOCKS="${SDAR_SELECTED_REMASK_WINDOW_BLOCKS:-3}"

echo "checkpoint_root=${CHECKPOINT_ROOT}" >&2
echo "work_root=${WORK_ROOT}" >&2
echo "selected_ckpts=${SELECTED_CKPTS}" >&2
echo "remask_threshold=${REMASK_THRESHOLD}" >&2
echo "remask_start_ratio=${REMASK_START_RATIO}" >&2
echo "remask_start_tokens=${REMASK_START_TOKENS}" >&2
echo "remask_prefix_guard_tokens=${REMASK_PREFIX_GUARD_TOKENS}" >&2
echo "remask_tail_guard_blocks=${REMASK_TAIL_GUARD_BLOCKS}" >&2
echo "remask_interval_blocks=${REMASK_INTERVAL_BLOCKS}" >&2
echo "remask_window_blocks=${REMASK_WINDOW_BLOCKS}" >&2

cd "${REPO_ROOT}"

for ckpt_name in ${SELECTED_CKPTS}; do
  checkpoint_path="${CHECKPOINT_ROOT}/${ckpt_name}"
  work_dir="${WORK_ROOT%/}/$(basename "${CHECKPOINT_ROOT}")/${ckpt_name}"

  if [[ ! -d "${checkpoint_path}" ]]; then
    echo "Missing checkpoint: ${checkpoint_path}" >&2
    exit 1
  fi

  echo "[run] ${ckpt_name} -> ${work_dir}" >&2
  export SDAR_MODEL_PATH="${checkpoint_path}"
  export SDAR_EVAL_SCOPE="gsm8k"
  export SDAR_EVAL_CONFIG="configs/eval_sdar_gap_hf.py"
  export SDAR_USE_REMASK="true"
  export SDAR_REMASK_THRESHOLD="${REMASK_THRESHOLD}"
  export SDAR_REMASK_START_RATIO="${REMASK_START_RATIO}"
  export SDAR_REMASK_START_TOKENS="${REMASK_START_TOKENS}"
  export SDAR_REMASK_PREFIX_GUARD_TOKENS="${REMASK_PREFIX_GUARD_TOKENS}"
  export SDAR_REMASK_TAIL_GUARD_BLOCKS="${REMASK_TAIL_GUARD_BLOCKS}"
  export SDAR_REMASK_INTERVAL_BLOCKS="${REMASK_INTERVAL_BLOCKS}"
  export SDAR_REMASK_WINDOW_BLOCKS="${REMASK_WINDOW_BLOCKS}"
  export SDAR_WORK_DIR="${work_dir}"
  bash "${REPO_ROOT}/script/test_gap.sh"
done
