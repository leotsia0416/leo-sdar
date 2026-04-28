#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_hfallnr
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/test_gap_hf_all_ckpts_noremask_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/test_gap_hf_all_ckpts_noremask_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

REPO_ROOT="/work/leotsia0416/projects/SDAR"
CHECKPOINT_ROOT="${SDAR_ALL_CKPTS_ROOT:-${REPO_ROOT}/checkpoint/training_157757}"
WORK_ROOT="${SDAR_ALL_CKPTS_WORK_ROOT:-./outputs/eval-chat-sdar-gap-training_157757-hf-noremask-all}"
CHECKPOINT_ORDER="${SDAR_ALL_CKPTS_ORDER:-asc}"
REMASK_THRESHOLD="${SDAR_ALL_CKPTS_NOREMASK_THRESHOLD:-1.0}"

echo "checkpoint_root=${CHECKPOINT_ROOT}" >&2
echo "work_root=${WORK_ROOT}" >&2
echo "checkpoint_order=${CHECKPOINT_ORDER}" >&2
echo "remask_threshold=${REMASK_THRESHOLD}" >&2

cd "${REPO_ROOT}"
export SDAR_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}"
export SDAR_WORK_DIR="${WORK_ROOT}"
export SDAR_CHECKPOINT_ORDER="${CHECKPOINT_ORDER}"
export SDAR_EVAL_ALL_CHECKPOINTS="true"
export SDAR_EVAL_SCOPE="gsm8k"
export SDAR_EVAL_CONFIG="configs/eval_sdar_gap_hf.py"
export SDAR_USE_REMASK="true"
export SDAR_REMASK_THRESHOLD="${REMASK_THRESHOLD}"

bash "${REPO_ROOT}/script/test_gap.sh"
