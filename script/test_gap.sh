#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_eval_gap_hf
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=3:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/test_gap_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/test_gap_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

REPO_ROOT="/work/leotsia0416/projects/SDAR"
EVAL_ENV_PREFIX="${EVAL_ENV_PREFIX:-/work/leotsia0416/sdar_eval}"

normalize_bool() {
  local value="${1:-}"
  value="${value,,}"
  case "${value}" in
    1|true|yes|on)
      echo "true"
      ;;
    *)
      echo "false"
      ;;
  esac
}

export PATH="${EVAL_ENV_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
cd "${REPO_ROOT}/evaluation/opencompass"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DEFAULT_MODEL_PATH="${REPO_ROOT}/checkpoint/training_140274/checkpoint-450"  # change this to your checkpoint path
if [[ -n "${SDAR_MODEL_PATH:-}" ]]; then
  if [[ ! -d "${SDAR_MODEL_PATH}" ]]; then
    echo "SDAR_MODEL_PATH does not exist: ${SDAR_MODEL_PATH}" >&2
    exit 1
  fi
  export SDAR_MODEL_PATH="$(realpath "${SDAR_MODEL_PATH}")"
  export SDAR_MODEL_ROOT="$(dirname "${SDAR_MODEL_PATH}")"
  export SDAR_MODEL_NAME="$(basename "${SDAR_MODEL_PATH}")"
else
  export SDAR_MODEL_PATH="${SDAR_MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
  export SDAR_MODEL_ROOT="${SDAR_MODEL_ROOT:-$(dirname "${SDAR_MODEL_PATH}")}"
  export SDAR_MODEL_NAME="${SDAR_MODEL_NAME:-$(basename "${SDAR_MODEL_PATH}")}"
  export SDAR_MODEL_PATH="${SDAR_MODEL_ROOT}/${SDAR_MODEL_NAME}"
fi

if [[ ! -d "${SDAR_MODEL_PATH}" ]]; then
  echo "Resolved model path does not exist: ${SDAR_MODEL_PATH}" >&2
  exit 1
fi

export SDAR_EVAL_GPUS="${SDAR_EVAL_GPUS:-1}"
export SDAR_INFER_BATCH_SIZE="${SDAR_INFER_BATCH_SIZE:-16}"
export SDAR_CONFIDENCE_THRESHOLD="${SDAR_CONFIDENCE_THRESHOLD:-0.95}"
export SDAR_REMASK_THRESHOLD="${SDAR_REMASK_THRESHOLD:-0.5}"
export SDAR_BLOCK_LENGTH="${SDAR_BLOCK_LENGTH:-4}"
export SDAR_MAX_NEW_TOKENS="${SDAR_MAX_NEW_TOKENS:-1024}"
export SDAR_TEMPERATURE="${SDAR_TEMPERATURE:-1.0}"
export SDAR_TORCH_DTYPE="${SDAR_TORCH_DTYPE:-bfloat16}"
export SDAR_WORK_DIR="${SDAR_WORK_DIR:-./outputs/eval-chat-sdar-gap}"

export HF_HOME="${SDAR_MODEL_ROOT}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$SDAR_MODEL_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1200
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

case "$SDAR_WORK_DIR" in
  /*) SDAR_WORK_DIR_ABS="$SDAR_WORK_DIR" ;;
  *) SDAR_WORK_DIR_ABS="$(pwd)/${SDAR_WORK_DIR#./}" ;;
esac
mkdir -p "$SDAR_WORK_DIR_ABS"

RUN_FULL_DATASET="$(normalize_bool "${SDAR_FULL_DATASET:-true}")"
if [[ "${RUN_FULL_DATASET}" == "true" ]]; then
  unset SDAR_TEST_RANGE
fi

RECORD_REMASK="$(normalize_bool "${SDAR_RECORD_REMASK:-true}")"
LAUNCH_TS="$(date +%s)"
DEFAULT_REMASK_TRACE_PATH="${SDAR_WORK_DIR_ABS}/remask_trace_${LAUNCH_TS}.jsonl"
DEFAULT_REMASK_SUMMARY_PATH="${SDAR_WORK_DIR_ABS}/remask_summary_${LAUNCH_TS}.txt"
CUSTOM_REMASK_TRACE_PATH="false"
CUSTOM_REMASK_SUMMARY_PATH="false"
if [[ -n "${SDAR_REMASK_TRACE_PATH:-}" ]]; then
  CUSTOM_REMASK_TRACE_PATH="true"
fi
if [[ -n "${SDAR_REMASK_SUMMARY_PATH:-}" ]]; then
  CUSTOM_REMASK_SUMMARY_PATH="true"
fi
if [[ "${RECORD_REMASK}" == "true" ]]; then
  export SDAR_REMASK_TRACE_PATH="${SDAR_REMASK_TRACE_PATH:-${DEFAULT_REMASK_TRACE_PATH}}"
  SDAR_REMASK_SUMMARY_PATH="${SDAR_REMASK_SUMMARY_PATH:-${DEFAULT_REMASK_SUMMARY_PATH}}"
  rm -f "${SDAR_REMASK_TRACE_PATH}" "${SDAR_REMASK_SUMMARY_PATH}"
else
  export SDAR_REMASK_TRACE_PATH=""
  SDAR_REMASK_SUMMARY_PATH="${SDAR_REMASK_SUMMARY_PATH:-${DEFAULT_REMASK_SUMMARY_PATH}}"
fi

TAIL_PID=""
EXP_DIR=""
cleanup() {
  if [[ -n "${TAIL_PID:-}" ]]; then
    kill "${TAIL_PID}" 2>/dev/null || true
    wait "${TAIL_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

locate_exp_dir() {
  local work_dir="$1"
  local launch_ts="$2"
  local candidate=""
  local candidate_ts=""
  local best_dir=""
  local best_ts=0

  while IFS= read -r candidate; do
    candidate_ts="$(stat -c %Y "$candidate" 2>/dev/null || echo 0)"
    if [[ "$candidate_ts" -ge "$launch_ts" && "$candidate_ts" -ge "$best_ts" ]]; then
      best_dir="$candidate"
      best_ts="$candidate_ts"
    fi
  done < <(find "$work_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)

  EXP_DIR="$best_dir"
}

mirror_infer_progress() {
  local run_pid="$1"
  local work_dir="$2"
  local launch_ts="$3"
  local infer_log=""

  while kill -0 "$run_pid" 2>/dev/null; do
    if [[ -z "$EXP_DIR" ]]; then
      locate_exp_dir "$work_dir" "$launch_ts"
    fi

    if [[ -n "$EXP_DIR" && -z "$infer_log" ]]; then
      infer_log="$(find "$EXP_DIR/logs/infer" -type f -name '*.out' 2>/dev/null | sort | head -n 1 || true)"
    fi

    if [[ -n "$infer_log" ]]; then
      echo "Mirroring infer progress from $infer_log to stderr" >&2
      tail --pid="$run_pid" -n 0 -f "$infer_log" >&2 &
      TAIL_PID="$!"
      break
    fi

    sleep 2
  done
}

write_remask_summary() {
  local trace_path="$1"
  local results_path="$2"
  local summary_path="$3"

  mkdir -p "$(dirname "$summary_path")"
  "${EVAL_ENV_PREFIX}/bin/python" - "$trace_path" "$results_path" "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

trace_path = Path(sys.argv[1])
results_path = Path(sys.argv[2]) if sys.argv[2] else None
summary_path = Path(sys.argv[3])

records = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines() if line.strip()]
count = len(records)
with_remask = sum(1 for record in records if record["total_remasked_tokens"] > 0)
without_remask = count - with_remask
total_remasked_tokens = sum(record["total_remasked_tokens"] for record in records)
total_remask_steps = sum(record["steps_with_remask"] for record in records)

lines = [
    f"trace_path={trace_path}",
    f"num_examples={count}",
    f"examples_with_remask={with_remask}",
    f"examples_without_remask={without_remask}",
    f"total_remasked_tokens={total_remasked_tokens}",
    f"avg_remasked_tokens_per_example={(total_remasked_tokens / count):.4f}" if count else "avg_remasked_tokens_per_example=0.0000",
    f"avg_remask_steps_per_example={(total_remask_steps / count):.4f}" if count else "avg_remask_steps_per_example=0.0000",
    f"avg_remasked_tokens_per_hit={(total_remasked_tokens / with_remask):.4f}" if with_remask else "avg_remasked_tokens_per_hit=0.0000",
]

details = None
if results_path is not None and results_path.is_file():
    result_payload = json.loads(results_path.read_text(encoding="utf-8"))
    details = result_payload.get("details")
    lines.append(f"results_path={results_path}")
    lines.append(f"accuracy={result_payload.get('accuracy')}")

if details is not None and len(details) == count:
    with_remask_correct = 0
    without_remask_correct = 0
    lines.append("per_example=index correct steps_with_remask total_remasked_tokens prompt")
    for idx, (record, detail) in enumerate(zip(records, details)):
        correct = bool(detail["correct"][0])
        if record["total_remasked_tokens"] > 0:
            with_remask_correct += int(correct)
        else:
            without_remask_correct += int(correct)
        prompt = record["input"][0]["prompt"].splitlines()[0] if record.get("input") else ""
        lines.append(
            f"{idx} {correct} {record['steps_with_remask']} {record['total_remasked_tokens']} {prompt}"
        )

    lines.append(
        f"with_remask_correct={with_remask_correct}/{with_remask}"
        if with_remask else "with_remask_correct=0/0"
    )
    lines.append(
        f"without_remask_correct={without_remask_correct}/{without_remask}"
        if without_remask else "without_remask_correct=0/0"
    )
    lines.append(
        f"with_remask_accuracy={(with_remask_correct / with_remask):.4f}"
        if with_remask else "with_remask_accuracy=0.0000"
    )
    lines.append(
        f"without_remask_accuracy={(without_remask_correct / without_remask):.4f}"
        if without_remask else "without_remask_accuracy=0.0000"
    )
elif details is not None:
    lines.append(f"results_detail_count_mismatch trace={count} results={len(details)}")

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_path)
PY
}

echo "Using model path: ${SDAR_MODEL_PATH}" >&2
if [[ "${RUN_FULL_DATASET}" == "true" ]]; then
  echo "Evaluating full GSM8K split." >&2
else
  echo "Evaluating subset with SDAR_TEST_RANGE=${SDAR_TEST_RANGE:-unset}" >&2
fi
if [[ "${RECORD_REMASK}" == "true" ]]; then
  echo "Recording remask trace to ${SDAR_REMASK_TRACE_PATH}" >&2
fi

"${EVAL_ENV_PREFIX}/bin/python" run.py configs/eval_sdar_gap_hf.py &
RUN_PID=$!
mirror_infer_progress "$RUN_PID" "$SDAR_WORK_DIR_ABS" "$LAUNCH_TS"
wait "$RUN_PID"

locate_exp_dir "$SDAR_WORK_DIR_ABS" "$LAUNCH_TS"

if [[ "${RECORD_REMASK}" == "true" && -f "${SDAR_REMASK_TRACE_PATH}" ]]; then
  RESULTS_PATH=""
  if [[ -n "${EXP_DIR}" ]]; then
    RESULTS_PATH="$(find "${EXP_DIR}/results" -type f -name 'gsm8k.json' 2>/dev/null | sort | head -n 1 || true)"
    if [[ "${CUSTOM_REMASK_SUMMARY_PATH}" != "true" ]]; then
      SDAR_REMASK_SUMMARY_PATH="${EXP_DIR}/summary/remask_summary.txt"
    fi
    if [[ "${CUSTOM_REMASK_TRACE_PATH}" != "true" ]]; then
      cp "${SDAR_REMASK_TRACE_PATH}" "${EXP_DIR}/summary/remask_trace.jsonl"
    fi
  fi
  write_remask_summary "${SDAR_REMASK_TRACE_PATH}" "${RESULTS_PATH}" "${SDAR_REMASK_SUMMARY_PATH}" >/tmp/sdar_remask_summary_path.txt
  REMASK_SUMMARY_WRITTEN_PATH="$(cat /tmp/sdar_remask_summary_path.txt)"
  rm -f /tmp/sdar_remask_summary_path.txt
  echo "Remask summary written to ${REMASK_SUMMARY_WRITTEN_PATH}" >&2
  sed -n '1,80p' "${REMASK_SUMMARY_WRITTEN_PATH}" >&2
fi
