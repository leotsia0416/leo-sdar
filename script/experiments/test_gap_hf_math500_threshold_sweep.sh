#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_m5thr
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/test_gap_hf_math500_threshold_sweep_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/test_gap_hf_math500_threshold_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

REPO_ROOT="/work/leotsia0416/projects/SDAR"

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

DEFAULT_CHECKPOINT_PATH="$(resolve_latest_checkpoint_dir "$(resolve_latest_training_root)")"
if [[ -z "${DEFAULT_CHECKPOINT_PATH}" ]]; then
  DEFAULT_CHECKPOINT_PATH="${REPO_ROOT}/Models/SDAR-1.7B-Chat-"
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_PATH="${SDAR_MATH500_SWEEP_MODEL_PATH:-${DEFAULT_CHECKPOINT_PATH}}"
THRESHOLDS="${SDAR_MATH500_SWEEP_THRESHOLDS:-0.80 0.85 0.95}"
WORK_ROOT="${SDAR_MATH500_SWEEP_WORK_ROOT:-./outputs/math500_threshold_sweep_${RUN_STAMP}}"
CONFIDENCE_THRESHOLD="${SDAR_MATH500_SWEEP_CONFIDENCE_THRESHOLD:-0.95}"
REMASK_START_RATIO="${SDAR_MATH500_SWEEP_REMASK_START_RATIO:-0.0}"
REMASK_START_TOKENS="${SDAR_MATH500_SWEEP_REMASK_START_TOKENS:-192}"
REMASK_PREFIX_GUARD_TOKENS="${SDAR_MATH500_SWEEP_REMASK_PREFIX_GUARD_TOKENS:-192}"
REMASK_TAIL_GUARD_BLOCKS="${SDAR_MATH500_SWEEP_REMASK_TAIL_GUARD_BLOCKS:-1}"
REMASK_INTERVAL_BLOCKS="${SDAR_MATH500_SWEEP_REMASK_INTERVAL_BLOCKS:-2}"
REMASK_WINDOW_BLOCKS="${SDAR_MATH500_SWEEP_REMASK_WINDOW_BLOCKS:-3}"
MAX_NEW_TOKENS="${SDAR_MATH500_SWEEP_MAX_NEW_TOKENS:-1536}"
EVAL_GPUS="${SDAR_MATH500_SWEEP_EVAL_GPUS:-2}"
INFER_BATCH_SIZE="${SDAR_MATH500_SWEEP_INFER_BATCH_SIZE:-1}"
FULL_DATASET_SIZE="${SDAR_MATH500_SWEEP_FULL_DATASET_SIZE:-500}"
DATASET_JSON_NAME="${SDAR_MATH500_SWEEP_RESULT_JSON:-math-500.json}"

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Missing checkpoint path: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

case "${WORK_ROOT}" in
  /*) WORK_ROOT_ABS="${WORK_ROOT}" ;;
  *) WORK_ROOT_ABS="${REPO_ROOT}/${WORK_ROOT#./}" ;;
esac
mkdir -p "${WORK_ROOT_ABS}"
SUMMARY_TSV="${WORK_ROOT_ABS}/summary.tsv"

python - <<'PY' > "${SUMMARY_TSV}"
columns = [
    'label',
    'subset_size',
    'accuracy',
    'event_checks',
    'active_checks',
    'active_candidate_checks',
    'triggered_checks',
    'trigger_rate_active_candidates',
    'best_score_p10',
    'best_score_p25',
    'best_score_p50',
    'best_score_p75',
    'best_score_p90',
    'best_score_p95',
    'best_score_min',
    'best_score_max',
    'triggered_best_score_p50',
    'triggered_best_score_p90',
    'score_margin_p50',
    'score_margin_p90',
    'would_trigger_rate_at_0.30',
    'would_trigger_rate_at_0.40',
    'would_trigger_rate_at_0.50',
    'would_trigger_rate_at_0.60',
    'would_trigger_rate_at_0.70',
    'would_trigger_rate_at_0.80',
    'would_trigger_rate_at_0.90',
]
print('\t'.join(columns))
PY

echo "checkpoint_path=${CHECKPOINT_PATH}" >&2
echo "work_root=${WORK_ROOT_ABS}" >&2
echo "thresholds=${THRESHOLDS}" >&2
echo "confidence_threshold=${CONFIDENCE_THRESHOLD}" >&2
echo "remask_start_tokens=${REMASK_START_TOKENS}" >&2
echo "remask_prefix_guard_tokens=${REMASK_PREFIX_GUARD_TOKENS}" >&2
echo "remask_interval_blocks=${REMASK_INTERVAL_BLOCKS}" >&2
echo "remask_window_blocks=${REMASK_WINDOW_BLOCKS}" >&2
echo "eval_gpus=${EVAL_GPUS}" >&2

cd "${REPO_ROOT}"

for threshold in ${THRESHOLDS}; do
  label="rt${threshold/./_}"
  case_root="${WORK_ROOT_ABS}/${label}"
  trace_path="${case_root}/remask_trace.jsonl"
  event_trace_path="${case_root}/remask_event_trace.jsonl"
  remask_summary_path="${case_root}/remask_summary.txt"
  event_summary_path="${case_root}/event_summary.txt"
  score_summary_path="${case_root}/score_summary.txt"

  mkdir -p "${case_root}"
  rm -f "${trace_path}" "${event_trace_path}" "${remask_summary_path}" "${event_summary_path}" "${score_summary_path}"

  echo "[${label}] threshold=${threshold}" >&2
  echo "[${label}] case_root=${case_root}" >&2

  export SDAR_MODEL_PATH="${CHECKPOINT_PATH}"
  export SDAR_MODEL_NAME="$(basename "${CHECKPOINT_PATH}")"
  export SDAR_EVAL_SCOPE="math500"
  export SDAR_EVAL_CONFIG="configs/eval_sdar_gap_hf_math500.py"
  export SDAR_USE_REMASK="true"
  export SDAR_EVAL_GPUS="${EVAL_GPUS}"
  export SDAR_INFER_BATCH_SIZE="${INFER_BATCH_SIZE}"
  export SDAR_FULL_DATASET="true"
  export SDAR_MATH500_ABBR="math-500"
  unset SDAR_TEST_RANGE
  export SDAR_CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD}"
  export SDAR_REMASK_THRESHOLD="${threshold}"
  export SDAR_REMASK_START_RATIO="${REMASK_START_RATIO}"
  export SDAR_REMASK_START_TOKENS="${REMASK_START_TOKENS}"
  export SDAR_REMASK_PREFIX_GUARD_TOKENS="${REMASK_PREFIX_GUARD_TOKENS}"
  export SDAR_REMASK_TAIL_GUARD_BLOCKS="${REMASK_TAIL_GUARD_BLOCKS}"
  export SDAR_REMASK_INTERVAL_BLOCKS="${REMASK_INTERVAL_BLOCKS}"
  export SDAR_REMASK_WINDOW_BLOCKS="${REMASK_WINDOW_BLOCKS}"
  export SDAR_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"
  export SDAR_RECORD_REMASK="true"
  export SDAR_RECORD_REMASK_EVENTS="true"
  export SDAR_WORK_DIR="${case_root}"
  export SDAR_REMASK_TRACE_PATH="${trace_path}"
  export SDAR_REMASK_EVENT_TRACE_PATH="${event_trace_path}"
  export SDAR_REMASK_SUMMARY_PATH="${remask_summary_path}"

  bash "${REPO_ROOT}/script/test_gap.sh"

  result_path="$(find "${case_root}" -type f -path "*/results/*/${DATASET_JSON_NAME}" | sort | tail -n 1)"
  if [[ -z "${result_path}" ]]; then
    echo "[${label}] Missing ${DATASET_JSON_NAME} under ${case_root}" >&2
    exit 1
  fi

  python "${REPO_ROOT}/script/helpers/analyze_remask_event_trace.py" "${event_trace_path}" > "${event_summary_path}"
  python "${REPO_ROOT}/script/helpers/summarize_gsm8k_remask_score_probe.py" \
    --label "${label}" \
    --subset-size "${FULL_DATASET_SIZE}" \
    --event-trace "${event_trace_path}" \
    --results "${result_path}" > "${score_summary_path}"
  python "${REPO_ROOT}/script/helpers/summarize_gsm8k_remask_score_probe.py" \
    --label "${label}" \
    --subset-size "${FULL_DATASET_SIZE}" \
    --event-trace "${event_trace_path}" \
    --results "${result_path}" \
    --tsv | tail -n 1 >> "${SUMMARY_TSV}"
done

echo "Completed MATH-500 threshold sweep. Summary written to ${SUMMARY_TSV}" >&2
cat "${SUMMARY_TSV}" >&2
