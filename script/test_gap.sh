#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_eval_gap_hf
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
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

has_model_weights() {
  local model_dir="${1:?missing model dir}"
  [[ -e "${model_dir}/model.safetensors" ]] && return 0
  [[ -e "${model_dir}/model.safetensors.index.json" ]] && return 0
  [[ -e "${model_dir}/pytorch_model.bin" ]] && return 0
  [[ -e "${model_dir}/pytorch_model.bin.index.json" ]] && return 0
  compgen -G "${model_dir}/model-*.safetensors" > /dev/null && return 0
  return 1
}

validate_hf_model_dir() {
  local model_dir="${1:?missing model dir}"
  if [[ ! -d "${model_dir}" ]]; then
    echo "Resolved model path does not exist: ${model_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${model_dir}/config.json" ]]; then
    echo "Resolved model path is missing config.json: ${model_dir}" >&2
    exit 1
  fi
  if ! has_model_weights "${model_dir}"; then
    echo "Resolved model path is missing model weights: ${model_dir}" >&2
    exit 1
  fi
}

DEFAULT_CHECKPOINT_ROOT="$(resolve_latest_training_root)"
DEFAULT_MODEL_PATH="$(resolve_latest_checkpoint_dir "${DEFAULT_CHECKPOINT_ROOT}")"
if [[ -z "${DEFAULT_MODEL_PATH}" ]]; then
  DEFAULT_MODEL_PATH="${REPO_ROOT}/Models/SDAR-1.7B-Chat-"
fi
DEFAULT_BLOCK_LENGTH="${SDAR_BLOCK_LENGTH:-4}"
DEFAULT_USE_REMASK="${SDAR_USE_REMASK:-true}"
DEFAULT_REMASK_THRESHOLD="${SDAR_REMASK_THRESHOLD:-0.15}"
DEFAULT_REMASK_START_RATIO="${SDAR_REMASK_START_RATIO:-0.5}"
DEFAULT_REMASK_INTERVAL_BLOCKS="${SDAR_REMASK_INTERVAL_BLOCKS:-2}"
DEFAULT_REMASK_WINDOW_BLOCKS="${SDAR_REMASK_WINDOW_BLOCKS:-3}"
DEFAULT_PROMPT_BUCKET_SIZE="${SDAR_PROMPT_BUCKET_SIZE:-32}"
DEFAULT_TEMPERATURE="${SDAR_TEMPERATURE:-0.0}"
DEFAULT_ACC_MONITOR_INTERVAL="${SDAR_ACC_MONITOR_INTERVAL:-30}"
DEFAULT_CONFIDENCE_THRESHOLD_REMASK="0.95"
DEFAULT_CONFIDENCE_THRESHOLD_BASE="0.95"
DEFAULT_EVAL_GPUS_REMASK="2"
DEFAULT_EVAL_GPUS_BASE="2"
DEFAULT_INFER_BATCH_SIZE_REMASK="1"
DEFAULT_INFER_BATCH_SIZE_NOREMASK="32"
DEFAULT_INFER_BATCH_SIZE_BASE="1"
DEFAULT_MAX_NEW_TOKENS_REMASK="1536"
DEFAULT_MAX_NEW_TOKENS_BASE="1536"

use_remask_mode() {
  normalize_bool "${SDAR_USE_REMASK:-${DEFAULT_USE_REMASK}}"
}

remask_disabled_by_threshold() {
  local threshold="${SDAR_REMASK_THRESHOLD:-${DEFAULT_REMASK_THRESHOLD}}"
  awk -v threshold="${threshold}" 'BEGIN { exit !(threshold >= 1.0) }'
}

default_single_work_dir() {
  if [[ "$(use_remask_mode)" == "true" ]]; then
    printf './outputs/eval-chat-sdar-gap\n'
  else
    printf './outputs/eval-chat-sdar-lmdeploy\n'
  fi
}

default_eval_gpus() {
  if [[ "$(use_remask_mode)" == "true" ]]; then
    printf '%s\n' "${DEFAULT_EVAL_GPUS_REMASK}"
  else
    printf '%s\n' "${DEFAULT_EVAL_GPUS_BASE}"
  fi
}

default_infer_batch_size() {
  if [[ "$(use_remask_mode)" == "true" ]]; then
    if remask_disabled_by_threshold; then
      printf '%s\n' "${DEFAULT_INFER_BATCH_SIZE_NOREMASK}"
    else
      printf '%s\n' "${DEFAULT_INFER_BATCH_SIZE_REMASK}"
    fi
  else
    printf '%s\n' "${DEFAULT_INFER_BATCH_SIZE_BASE}"
  fi
}

default_decode_backend() {
  printf 'gap\n'
}

default_confidence_threshold() {
  if [[ "$(use_remask_mode)" == "true" ]]; then
    printf '%s\n' "${DEFAULT_CONFIDENCE_THRESHOLD_REMASK}"
  else
    printf '%s\n' "${DEFAULT_CONFIDENCE_THRESHOLD_BASE}"
  fi
}

default_max_new_tokens() {
  if [[ "$(use_remask_mode)" == "true" ]]; then
    printf '%s\n' "${DEFAULT_MAX_NEW_TOKENS_REMASK}"
  else
    printf '%s\n' "${DEFAULT_MAX_NEW_TOKENS_BASE}"
  fi
}

default_eval_config() {
  if [[ "$(use_remask_mode)" == "true" ]]; then
    printf 'configs/eval_sdar_gap_hf.py\n'
  else
    printf 'configs/eval_sdar_lmdeploy.py\n'
  fi
}

list_checkpoint_dirs() {
  local checkpoint_root="$1"
  local checkpoint_order="${2:-desc}"
  local -a checkpoint_dirs=()
  local -a ordered_dirs=()
  local idx=0

  if [[ ! -d "${checkpoint_root}" ]]; then
    echo "Checkpoint root does not exist: ${checkpoint_root}" >&2
    return 1
  fi

  mapfile -t checkpoint_dirs < <(find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  if [[ "${#checkpoint_dirs[@]}" -eq 0 ]]; then
    echo "No checkpoint-* directories found under ${checkpoint_root}" >&2
    return 1
  fi

  case "${checkpoint_order,,}" in
    desc|reverse|backward)
      for (( idx=${#checkpoint_dirs[@]}-1; idx>=0; idx-- )); do
        ordered_dirs+=("${checkpoint_dirs[$idx]}")
      done
      ;;
    asc|forward)
      ordered_dirs=("${checkpoint_dirs[@]}")
      ;;
    *)
      echo "Unsupported SDAR_CHECKPOINT_ORDER: ${checkpoint_order}" >&2
      return 1
      ;;
  esac

  printf '%s\n' "${ordered_dirs[@]}"
}

format_threshold() {
  local value="${1:?missing threshold value}"
  printf '%.2f' "${value}" | tr '.' '_'
}

resolve_work_dir_abs() {
  local work_dir="${1:?missing work dir}"
  case "${work_dir}" in
    /*) printf '%s\n' "${work_dir}" ;;
    *) printf '%s\n' "$(pwd)/${work_dir#./}" ;;
  esac
}

default_all_work_dir() {
  local checkpoint_root="${1:?missing checkpoint root}"
  if [[ "$(use_remask_mode)" == "true" ]]; then
    printf './outputs/eval-gap-%s-all-ckpts\n' "$(basename "${checkpoint_root}")"
  else
    printf './outputs/eval-lmdeploy-%s-all-ckpts\n' "$(basename "${checkpoint_root}")"
  fi
}

build_model_abbr() {
  local model_name="${1:?missing model name}"
  local confidence_threshold=""
  confidence_threshold="${SDAR_CONFIDENCE_THRESHOLD:-}"
  if [[ -z "${confidence_threshold}" ]]; then
    if [[ "$(use_remask_mode)" == "true" ]]; then
      confidence_threshold="${DEFAULT_CONFIDENCE_THRESHOLD_REMASK}"
    else
      confidence_threshold="${DEFAULT_CONFIDENCE_THRESHOLD_BASE}"
    fi
  fi

  if [[ "$(use_remask_mode)" == "true" ]]; then
    printf '%s-gap-b%s-thr%s-rt%s-t%s-rs%s\n' \
      "${model_name}" \
      "${DEFAULT_BLOCK_LENGTH}" \
      "$(format_threshold "${confidence_threshold}")" \
      "$(format_threshold "${DEFAULT_REMASK_THRESHOLD}")" \
      "$(format_threshold "${DEFAULT_TEMPERATURE}")" \
      "$(format_threshold "${DEFAULT_REMASK_START_RATIO}")-ri${DEFAULT_REMASK_INTERVAL_BLOCKS}-rw${DEFAULT_REMASK_WINDOW_BLOCKS}"
  else
    printf '%s-b%s-thr%s\n' \
      "${model_name}" \
      "${DEFAULT_BLOCK_LENGTH}" \
      "$(format_threshold "${confidence_threshold}")"
  fi
}

find_existing_result() {
  local checkpoint_path="${1:?missing checkpoint path}"
  local work_dir_base_abs="${2:?missing work dir base}"
  local checkpoint_root="${3:?missing checkpoint root}"
  local model_name=""
  local model_abbr=""
  local legacy_work_dir_abs=""
  local search_root=""
  local existing_result=""
  local -a search_roots=()

  model_name="$(basename "${checkpoint_path}")"
  model_abbr="$(build_model_abbr "${model_name}")"
  legacy_work_dir_abs="$(resolve_work_dir_abs "$(default_single_work_dir)")"
  search_roots+=("${work_dir_base_abs}")
  if [[ "${legacy_work_dir_abs}" != "${work_dir_base_abs}" ]]; then
    search_roots+=("${legacy_work_dir_abs}")
  fi

  for search_root in "${search_roots[@]}"; do
    if [[ ! -d "${search_root}" ]]; then
      continue
    fi
    existing_result="$(find "${search_root}" -type f -path "*/results/${model_abbr}/gsm8k.json" -print -quit 2>/dev/null || true)"
    if [[ -n "${existing_result}" ]]; then
      printf '%s\n' "${existing_result}"
      return 0
    fi
  done

  return 1
}

run_all_checkpoints() {
  local requested_root="${SDAR_CHECKPOINT_ROOT:-${DEFAULT_CHECKPOINT_ROOT:-${SDAR_MODEL_PATH:-${DEFAULT_MODEL_PATH}}}}"
  local checkpoint_root="${requested_root}"
  local checkpoint_order="${SDAR_CHECKPOINT_ORDER:-desc}"
  local checkpoint_limit="${SDAR_CHECKPOINT_LIMIT:-}"
  local work_dir_base=""
  local work_dir_base_abs=""
  local script_path
  local -a checkpoint_dirs=()
  local dispatched=0
  local skipped=0
  local checkpoint_path=""
  local model_root=""
  local model_name=""
  local child_work_dir=""
  local total_checkpoints=0
  local existing_result=""

  if [[ "${checkpoint_root##*/}" == checkpoint-* ]]; then
    checkpoint_root="$(dirname "${checkpoint_root}")"
  fi
  checkpoint_root="$(realpath "${checkpoint_root}")"
  script_path="$(realpath "$0")"

  if [[ -n "${checkpoint_limit}" && ! "${checkpoint_limit}" =~ ^[0-9]+$ ]]; then
    echo "SDAR_CHECKPOINT_LIMIT must be an integer, got: ${checkpoint_limit}" >&2
    return 1
  fi

  mapfile -t checkpoint_dirs < <(list_checkpoint_dirs "${checkpoint_root}" "${checkpoint_order}")
  if [[ "${#checkpoint_dirs[@]}" -eq 0 ]]; then
    echo "No checkpoints to run under ${checkpoint_root}" >&2
    return 1
  fi
  total_checkpoints="${#checkpoint_dirs[@]}"
  work_dir_base="${SDAR_WORK_DIR:-$(default_all_work_dir "${checkpoint_root}")}"
  work_dir_base_abs="$(resolve_work_dir_abs "${work_dir_base}")"

  echo "Running ${total_checkpoints} checkpoints sequentially from ${checkpoint_root} (${checkpoint_order})" >&2
  echo "Batch work dir base: ${work_dir_base_abs}" >&2
  for checkpoint_path in "${checkpoint_dirs[@]}"; do
    if [[ -n "${checkpoint_limit}" && "${dispatched}" -ge "${checkpoint_limit}" ]]; then
      break
    fi

    model_root="$(dirname "${checkpoint_path}")"
    model_name="$(basename "${checkpoint_path}")"
    child_work_dir="${work_dir_base%/}/$(basename "${checkpoint_root}")/${model_name}"
    existing_result="$(find_existing_result "${checkpoint_path}" "${work_dir_base_abs}" "${checkpoint_root}" || true)"
    if [[ -n "${existing_result}" ]]; then
      skipped=$((skipped + 1))
      echo "[skip ${skipped}] ${model_name} already has results at ${existing_result}" >&2
      continue
    fi
    echo "[$((dispatched + 1))/${total_checkpoints}] ${model_name} -> ${child_work_dir}" >&2
    SDAR_EVAL_ALL_CHECKPOINTS=false \
      SDAR_USE_REMASK="$(use_remask_mode)" \
      SDAR_MODEL_PATH="${checkpoint_path}" \
      SDAR_MODEL_ROOT="${model_root}" \
      SDAR_MODEL_NAME="${model_name}" \
      SDAR_WORK_DIR="${child_work_dir}" \
      bash "${script_path}"
    dispatched=$((dispatched + 1))
  done

  echo "Completed ${dispatched} checkpoint evals. Skipped ${skipped} completed checkpoints." >&2
}

DEFAULT_RUN_ALL_CHECKPOINTS="false"
if [[ -z "${SDAR_EVAL_ALL_CHECKPOINTS:-}" && -z "${SDAR_MODEL_PATH:-}" && -n "${DEFAULT_CHECKPOINT_ROOT:-}" ]]; then
  DEFAULT_RUN_ALL_CHECKPOINTS="true"
fi
RUN_ALL_CHECKPOINTS="$(normalize_bool "${SDAR_EVAL_ALL_CHECKPOINTS:-${DEFAULT_RUN_ALL_CHECKPOINTS}}")"
if [[ "${RUN_ALL_CHECKPOINTS}" == "true" ]]; then
  run_all_checkpoints
  exit 0
fi

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
validate_hf_model_dir "${SDAR_MODEL_PATH}"

export SDAR_USE_REMASK="$(use_remask_mode)"
export SDAR_EVAL_GPUS="${SDAR_EVAL_GPUS:-$(default_eval_gpus)}"
export SDAR_INFER_BATCH_SIZE="${SDAR_INFER_BATCH_SIZE:-$(default_infer_batch_size)}"
export SDAR_CONFIDENCE_THRESHOLD="${SDAR_CONFIDENCE_THRESHOLD:-$(default_confidence_threshold)}"
export SDAR_REMASK_THRESHOLD="${SDAR_REMASK_THRESHOLD:-0.15}"
export SDAR_REMASK_START_RATIO="${SDAR_REMASK_START_RATIO:-0.5}"
export SDAR_REMASK_INTERVAL_BLOCKS="${SDAR_REMASK_INTERVAL_BLOCKS:-2}"
export SDAR_REMASK_WINDOW_BLOCKS="${SDAR_REMASK_WINDOW_BLOCKS:-3}"
export SDAR_PROMPT_BUCKET_SIZE="${SDAR_PROMPT_BUCKET_SIZE:-32}"
export SDAR_BLOCK_LENGTH="${SDAR_BLOCK_LENGTH:-4}"
export SDAR_MAX_NEW_TOKENS="${SDAR_MAX_NEW_TOKENS:-$(default_max_new_tokens)}"
export SDAR_TEMPERATURE="${SDAR_TEMPERATURE:-0.0}"
export SDAR_TORCH_DTYPE="${SDAR_TORCH_DTYPE:-bfloat16}"
export SDAR_DECODE_BACKEND="${SDAR_DECODE_BACKEND:-$(default_decode_backend)}"
export SDAR_WORK_DIR="${SDAR_WORK_DIR:-$(default_single_work_dir)}"
export SDAR_ACC_MONITOR_INTERVAL="${SDAR_ACC_MONITOR_INTERVAL:-${DEFAULT_ACC_MONITOR_INTERVAL}}"
export SDAR_EVAL_CONFIG="${SDAR_EVAL_CONFIG:-$(default_eval_config)}"

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

DEFAULT_RECORD_REMASK="true"
if [[ "${SDAR_USE_REMASK}" != "true" ]]; then
  DEFAULT_RECORD_REMASK="false"
fi
RECORD_REMASK="$(normalize_bool "${SDAR_RECORD_REMASK:-${DEFAULT_RECORD_REMASK}}")"
RECORD_REMASK_EVENTS="$(normalize_bool "${SDAR_RECORD_REMASK_EVENTS:-false}")"
LAUNCH_TS="$(date +%s)"
DEFAULT_REMASK_TRACE_PATH="${SDAR_WORK_DIR_ABS}/remask_trace_${LAUNCH_TS}.jsonl"
DEFAULT_REMASK_EVENT_TRACE_PATH="${SDAR_WORK_DIR_ABS}/remask_event_trace_${LAUNCH_TS}.jsonl"
DEFAULT_REMASK_SUMMARY_PATH="${SDAR_WORK_DIR_ABS}/remask_summary_${LAUNCH_TS}.txt"
CUSTOM_REMASK_TRACE_PATH="false"
CUSTOM_REMASK_EVENT_TRACE_PATH="false"
CUSTOM_REMASK_SUMMARY_PATH="false"
if [[ -n "${SDAR_REMASK_TRACE_PATH:-}" ]]; then
  CUSTOM_REMASK_TRACE_PATH="true"
fi
if [[ -n "${SDAR_REMASK_EVENT_TRACE_PATH:-}" ]]; then
  CUSTOM_REMASK_EVENT_TRACE_PATH="true"
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
if [[ "${RECORD_REMASK_EVENTS}" == "true" ]]; then
  export SDAR_REMASK_EVENT_TRACE_PATH="${SDAR_REMASK_EVENT_TRACE_PATH:-${DEFAULT_REMASK_EVENT_TRACE_PATH}}"
  rm -f "${SDAR_REMASK_EVENT_TRACE_PATH}"
else
  export SDAR_REMASK_EVENT_TRACE_PATH=""
fi

TAIL_PID=""
ACC_MONITOR_PID=""
EXP_DIR=""
cleanup() {
  if [[ -n "${TAIL_PID:-}" ]]; then
    kill "${TAIL_PID}" 2>/dev/null || true
    wait "${TAIL_PID}" 2>/dev/null || true
  fi
  if [[ -n "${ACC_MONITOR_PID:-}" ]]; then
    kill "${ACC_MONITOR_PID}" 2>/dev/null || true
    wait "${ACC_MONITOR_PID}" 2>/dev/null || true
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

locate_prediction_progress_path() {
  local exp_dir="$1"
  local tmp_path=""
  local final_path=""

  if [[ -z "${exp_dir}" || ! -d "${exp_dir}" ]]; then
    return 0
  fi

  tmp_path="$(find "${exp_dir}/predictions" -type f -name 'tmp_gsm8k.json' 2>/dev/null | sort | head -n 1 || true)"
  if [[ -n "${tmp_path}" ]]; then
    printf '%s\n' "${tmp_path}"
    return 0
  fi

  final_path="$(find "${exp_dir}/predictions" -type f -name 'gsm8k.json' 2>/dev/null | sort | head -n 1 || true)"
  if [[ -n "${final_path}" ]]; then
    printf '%s\n' "${final_path}"
  fi
}

emit_running_accuracy() {
  local progress_path="$1"

  "${EVAL_ENV_PREFIX}/bin/python" - "$progress_path" <<'PY'
import json
import sys
from pathlib import Path

from opencompass.datasets import MATHEvaluator, gsm8k_dataset_postprocess, math_postprocess_sdar

progress_path = Path(sys.argv[1])
if not progress_path.is_file():
    sys.exit(1)

try:
    payload = json.loads(progress_path.read_text(encoding='utf-8'))
except Exception:
    sys.exit(2)

if not isinstance(payload, dict) or not payload:
    sys.exit(3)

def sort_key(item):
    key = item[0]
    try:
        return (0, int(key))
    except Exception:
        return (1, str(key))

predictions = []
references = []
for _, record in sorted(payload.items(), key=sort_key):
    if not isinstance(record, dict):
        continue
    pred = record.get('prediction')
    gold = record.get('gold')
    if pred is None or gold is None:
        continue
    try:
        predictions.append(math_postprocess_sdar(pred))
        references.append(gsm8k_dataset_postprocess(gold))
    except Exception:
        continue

if not predictions:
    sys.exit(4)

result = MATHEvaluator(version='v2').score(predictions, references)
accuracy = float(result.get('accuracy', 0.0))
details = result.get('details') or []
correct = sum(1 for detail in details if detail.get('correct'))
print(f"running accuracy: {accuracy:.2f}% ({correct}/{len(predictions)} complete) [{progress_path.name}]")
PY
}

monitor_running_accuracy() {
  local run_pid="$1"
  local work_dir="$2"
  local launch_ts="$3"
  local interval_s="${SDAR_ACC_MONITOR_INTERVAL:-${DEFAULT_ACC_MONITOR_INTERVAL}}"
  local progress_path=""
  local report=""
  local last_report=""

  if [[ ! "${interval_s}" =~ ^[0-9]+$ ]] || [[ "${interval_s}" -lt 1 ]]; then
    interval_s="${DEFAULT_ACC_MONITOR_INTERVAL}"
  fi

  while kill -0 "$run_pid" 2>/dev/null; do
    if [[ -z "$EXP_DIR" ]]; then
      locate_exp_dir "$work_dir" "$launch_ts"
    fi

    progress_path="$(locate_prediction_progress_path "${EXP_DIR:-}")"
    if [[ -n "${progress_path}" ]]; then
      report="$(emit_running_accuracy "${progress_path}" 2>/dev/null || true)"
      if [[ -n "${report}" && "${report}" != "${last_report}" ]]; then
        echo "${report}" >&2
        last_report="${report}"
      fi
    fi

    sleep "${interval_s}"
  done

  progress_path="$(locate_prediction_progress_path "${EXP_DIR:-}")"
  if [[ -n "${progress_path}" ]]; then
    report="$(emit_running_accuracy "${progress_path}" 2>/dev/null || true)"
    if [[ -n "${report}" && "${report}" != "${last_report}" ]]; then
      echo "${report}" >&2
    fi
  fi
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
echo "Using eval config: ${SDAR_EVAL_CONFIG}" >&2
echo "Use remask: ${SDAR_USE_REMASK}" >&2
echo "Decode backend: ${SDAR_DECODE_BACKEND}" >&2
echo "Infer batch size: ${SDAR_INFER_BATCH_SIZE}" >&2
echo "Eval workers: ${SDAR_EVAL_GPUS}" >&2
echo "Prompt bucket size: ${SDAR_PROMPT_BUCKET_SIZE}" >&2
echo "Remask cadence: start_ratio=${SDAR_REMASK_START_RATIO}, interval_blocks=${SDAR_REMASK_INTERVAL_BLOCKS}" >&2
echo "Running accuracy monitor interval: ${SDAR_ACC_MONITOR_INTERVAL}s" >&2
if [[ -n "${SDAR_GSM8K_PATH:-}" ]]; then
  echo "GSM8K dataset path: ${SDAR_GSM8K_PATH}" >&2
fi
if [[ -n "${SDAR_GSM8K_ABBR:-}" ]]; then
  echo "GSM8K dataset abbr: ${SDAR_GSM8K_ABBR}" >&2
fi
if [[ "${RUN_FULL_DATASET}" == "true" ]]; then
  if [[ -n "${SDAR_GSM8K_PATH:-}" && "${SDAR_GSM8K_PATH}" != "/work/leotsia0416/datasets/gsm8k" ]]; then
    echo "Evaluating custom GSM8K dataset directory." >&2
  else
    echo "Evaluating full GSM8K split." >&2
  fi
else
  echo "Evaluating subset with SDAR_TEST_RANGE=${SDAR_TEST_RANGE:-unset}" >&2
fi
if [[ "${RECORD_REMASK}" == "true" ]]; then
  echo "Recording remask trace to ${SDAR_REMASK_TRACE_PATH}" >&2
fi
if [[ "${RECORD_REMASK_EVENTS}" == "true" ]]; then
  echo "Recording remask event trace to ${SDAR_REMASK_EVENT_TRACE_PATH}" >&2
fi

"${EVAL_ENV_PREFIX}/bin/python" run.py "${SDAR_EVAL_CONFIG}" &
RUN_PID=$!
mirror_infer_progress "$RUN_PID" "$SDAR_WORK_DIR_ABS" "$LAUNCH_TS"
monitor_running_accuracy "$RUN_PID" "$SDAR_WORK_DIR_ABS" "$LAUNCH_TS" &
ACC_MONITOR_PID="$!"
wait "$RUN_PID"
if [[ -n "${ACC_MONITOR_PID:-}" ]]; then
  wait "${ACC_MONITOR_PID}" 2>/dev/null || true
  ACC_MONITOR_PID=""
fi

locate_exp_dir "$SDAR_WORK_DIR_ABS" "$LAUNCH_TS"

if [[ "${RECORD_REMASK}" == "true" && -f "${SDAR_REMASK_TRACE_PATH}" ]]; then
  RESULTS_PATH=""
  if [[ -n "${EXP_DIR}" ]]; then
    RESULTS_PATH="$(find "${EXP_DIR}/results" -type f -name '*.json' 2>/dev/null | sort | head -n 1 || true)"
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
if [[ "${RECORD_REMASK_EVENTS}" == "true" && -f "${SDAR_REMASK_EVENT_TRACE_PATH}" ]]; then
  if [[ -n "${EXP_DIR}" && "${CUSTOM_REMASK_EVENT_TRACE_PATH}" != "true" ]]; then
    cp "${SDAR_REMASK_EVENT_TRACE_PATH}" "${EXP_DIR}/summary/remask_event_trace.jsonl"
  fi
  echo "Remask event trace written to ${SDAR_REMASK_EVENT_TRACE_PATH}" >&2
fi
