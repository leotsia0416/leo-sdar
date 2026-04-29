#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J lmdeploy_gap_sweep
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/lmdeploy_gap_sweep_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/lmdeploy_gap_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

REPO_ROOT="/work/leotsia0416/projects/SDAR"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoint/training_173120/checkpoint-250}"
MODEL_NAME="${MODEL_NAME:-training_173120-checkpoint-250}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/eval-lmdeploy-gap-remask-training_173120-checkpoint-250-start-threshold-sweep}"

START_TOKENS_LIST="${START_TOKENS_LIST:-32 64 96 128}"
REMASK_THRESHOLDS="${REMASK_THRESHOLDS:-0.80 0.85 0.90 0.95}"

mkdir -p "${OUT_ROOT}"

echo "model_path=${MODEL_PATH}" >&2
echo "out_root=${OUT_ROOT}" >&2
echo "start_tokens=${START_TOKENS_LIST}" >&2
echo "remask_thresholds=${REMASK_THRESHOLDS}" >&2
echo "slurm_job_id=${SLURM_JOB_ID:-local}" >&2

for start_tokens in ${START_TOKENS_LIST}; do
  for threshold in ${REMASK_THRESHOLDS}; do
    threshold_tag="${threshold/./}"
    case_root="${OUT_ROOT}/start${start_tokens}_rt${threshold_tag}"
    trace_path="${case_root}/remask_trace.jsonl"
    mkdir -p "${case_root}"
    rm -f "${trace_path}"

    echo "[$(date '+%F %T')] start_tokens=${start_tokens} remask_threshold=${threshold}" >&2

    SDAR_MODEL_PATH="${MODEL_PATH}" \
    SDAR_MODEL_NAME="${MODEL_NAME}" \
    SDAR_EVAL_CONFIG="configs/eval_sdar_lmdeploy_gap_remask.py" \
    SDAR_EVAL_GPUS=2 \
    SDAR_INFER_BATCH_SIZE=1 \
    SDAR_MAX_NEW_TOKENS=1536 \
    SDAR_WORK_DIR="${case_root}" \
    SDAR_RUN_NAME="lmdeploy_gap_${MODEL_NAME}_start${start_tokens}_rt${threshold_tag}" \
    SDAR_CONFIDENCE_THRESHOLD=0.95 \
    SDAR_REMASK_THRESHOLD="${threshold}" \
    SDAR_REMASK_INTERVAL_BLOCKS=2 \
    SDAR_REMASK_START_TOKENS="${start_tokens}" \
    SDAR_LMDEPLOY_REMASK_TRACE_PATH="${trace_path}" \
    bash "${REPO_ROOT}/script/test.sh"
  done
done

echo "[$(date '+%F %T')] sweep done" >&2
