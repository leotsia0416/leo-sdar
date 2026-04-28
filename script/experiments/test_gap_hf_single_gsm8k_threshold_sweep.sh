#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_gsm1sweep
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/test_gap_hf_single_gsm8k_threshold_sweep_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/test_gap_hf_single_gsm8k_threshold_sweep_%j.err
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

format_threshold_tag() {
  python - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print(f"{value:.2f}".replace('.', '_'))
PY
}

extract_question_preview() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

idx = int(sys.argv[1])
path = Path('/work/leotsia0416/datasets/gsm8k/test.jsonl')
with path.open('r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        if line_idx == idx:
            payload = json.loads(line)
            print(payload['question'].splitlines()[0])
            break
PY
}

DEFAULT_CHECKPOINT_PATH="$(resolve_latest_checkpoint_dir "$(resolve_latest_training_root)")"
if [[ -z "${DEFAULT_CHECKPOINT_PATH}" ]]; then
  DEFAULT_CHECKPOINT_PATH="${REPO_ROOT}/Models/SDAR-1.7B-Chat-"
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_PATH="${SDAR_SWEEP_MODEL_PATH:-${DEFAULT_CHECKPOINT_PATH}}"
EXAMPLE_INDEX="${SDAR_SWEEP_EXAMPLE_INDEX:-13}"
THRESHOLDS="${SDAR_SWEEP_THRESHOLDS:-0.30 0.50 0.65 0.80 0.90 1.00}"
WORK_ROOT="${SDAR_SWEEP_WORK_ROOT:-./outputs/gsm8k_single_remask_threshold_sweep_${RUN_STAMP}}"
CONFIDENCE_THRESHOLD="${SDAR_SWEEP_CONFIDENCE_THRESHOLD:-0.95}"
REMASK_START_RATIO="${SDAR_SWEEP_REMASK_START_RATIO:-0.0}"
REMASK_START_TOKENS="${SDAR_SWEEP_REMASK_START_TOKENS:-192}"
REMASK_PREFIX_GUARD_TOKENS="${SDAR_SWEEP_REMASK_PREFIX_GUARD_TOKENS:-192}"
REMASK_TAIL_GUARD_BLOCKS="${SDAR_SWEEP_REMASK_TAIL_GUARD_BLOCKS:-1}"
REMASK_INTERVAL_BLOCKS="${SDAR_SWEEP_REMASK_INTERVAL_BLOCKS:-2}"
REMASK_WINDOW_BLOCKS="${SDAR_SWEEP_REMASK_WINDOW_BLOCKS:-3}"
MAX_NEW_TOKENS="${SDAR_SWEEP_MAX_NEW_TOKENS:-1536}"
SUMMARY_TSV=""

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Missing checkpoint path: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

QUESTION_PREVIEW="$(extract_question_preview "${EXAMPLE_INDEX}")"
TEST_RANGE="[${EXAMPLE_INDEX}:$((EXAMPLE_INDEX + 1))]"

case "${WORK_ROOT}" in
  /*) WORK_ROOT_ABS="${WORK_ROOT}" ;;
  *) WORK_ROOT_ABS="${REPO_ROOT}/${WORK_ROOT#./}" ;;
esac
mkdir -p "${WORK_ROOT_ABS}"
SUMMARY_TSV="${WORK_ROOT_ABS}/summary.tsv"

python "${REPO_ROOT}/script/helpers/summarize_single_gsm8k_remask_sweep.py" --header > "${SUMMARY_TSV}"

echo "checkpoint_path=${CHECKPOINT_PATH}" >&2
echo "work_root=${WORK_ROOT_ABS}" >&2
echo "example_index=${EXAMPLE_INDEX}" >&2
echo "test_range=${TEST_RANGE}" >&2
echo "question_preview=${QUESTION_PREVIEW}" >&2
echo "thresholds=${THRESHOLDS}" >&2
echo "confidence_threshold=${CONFIDENCE_THRESHOLD}" >&2
echo "remask_start_ratio=${REMASK_START_RATIO}" >&2
echo "remask_start_tokens=${REMASK_START_TOKENS}" >&2
echo "remask_prefix_guard_tokens=${REMASK_PREFIX_GUARD_TOKENS}" >&2
echo "remask_tail_guard_blocks=${REMASK_TAIL_GUARD_BLOCKS}" >&2
echo "remask_interval_blocks=${REMASK_INTERVAL_BLOCKS}" >&2
echo "remask_window_blocks=${REMASK_WINDOW_BLOCKS}" >&2
echo "summary_tsv=${SUMMARY_TSV}" >&2

cd "${REPO_ROOT}"

for threshold in ${THRESHOLDS}; do
  threshold_tag="$(format_threshold_tag "${threshold}")"
  label="thr${threshold_tag}"
  case_root="${WORK_ROOT_ABS}/${label}"
  trace_path="${case_root}/remask_trace.jsonl"
  event_trace_path="${case_root}/remask_event_trace.jsonl"
  remask_summary_path="${case_root}/remask_summary.txt"
  event_summary_path="${case_root}/event_summary.txt"

  mkdir -p "${case_root}"
  rm -f "${trace_path}" "${event_trace_path}" "${remask_summary_path}" "${event_summary_path}"

  echo "[${label}] threshold=${threshold}" >&2
  echo "[${label}] case_root=${case_root}" >&2

  export SDAR_MODEL_PATH="${CHECKPOINT_PATH}"
  export SDAR_EVAL_SCOPE="gsm8k"
  export SDAR_EVAL_CONFIG="configs/eval_sdar_gap_hf.py"
  export SDAR_USE_REMASK="true"
  export SDAR_EVAL_GPUS="1"
  export SDAR_INFER_BATCH_SIZE="1"
  export SDAR_FULL_DATASET="false"
  export SDAR_TEST_RANGE="${TEST_RANGE}"
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

  result_path="$(find "${case_root}" -type f -path '*/results/*/gsm8k.json' | sort | tail -n 1)"
  if [[ -z "${result_path}" ]]; then
    echo "[${label}] Missing gsm8k result json under ${case_root}" >&2
    exit 1
  fi

  python "${REPO_ROOT}/script/helpers/analyze_remask_event_trace.py" "${event_trace_path}" > "${event_summary_path}"
  python "${REPO_ROOT}/script/helpers/summarize_single_gsm8k_remask_sweep.py" \
    --label "${label}" \
    --threshold "${threshold}" \
    --trace "${trace_path}" \
    --event-trace "${event_trace_path}" \
    --results "${result_path}" >> "${SUMMARY_TSV}"
done

echo "Completed sweep. Summary written to ${SUMMARY_TSV}" >&2
cat "${SUMMARY_TSV}" >&2
