#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_fwdcmp
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/test_gap_single_forward_compare_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/test_gap_single_forward_compare_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

REPO_ROOT="/work/leotsia0416/projects/SDAR"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="${SDAR_FWD_CMP_MODEL_PATH:-${REPO_ROOT}/checkpoint/training_173120/checkpoint-250}"
TEST_RANGE="${SDAR_FWD_CMP_TEST_RANGE:-[365:366]}"
WORK_ROOT="${SDAR_FWD_CMP_WORK_ROOT:-${REPO_ROOT}/outputs/forward_compare_single_${RUN_STAMP}}"
CONFIDENCE_THRESHOLD="${SDAR_FWD_CMP_CONFIDENCE_THRESHOLD:-0.95}"
REMASK_START_TOKENS="${SDAR_FWD_CMP_REMASK_START_TOKENS:-128}"
REMASK_PREFIX_GUARD_TOKENS="${SDAR_FWD_CMP_REMASK_PREFIX_GUARD_TOKENS:-96}"
REMASK_INTERVAL_BLOCKS="${SDAR_FWD_CMP_REMASK_INTERVAL_BLOCKS:-2}"
REMASK_WINDOW_BLOCKS="${SDAR_FWD_CMP_REMASK_WINDOW_BLOCKS:-3}"
REMASK_TAIL_GUARD_BLOCKS="${SDAR_FWD_CMP_REMASK_TAIL_GUARD_BLOCKS:-1}"
MAX_NEW_TOKENS="${SDAR_FWD_CMP_MAX_NEW_TOKENS:-1536}"

mkdir -p "${WORK_ROOT}"

run_case() {
  local label="$1"
  local threshold="$2"
  local case_root="${WORK_ROOT}/${label}"

  mkdir -p "${case_root}"
  export SDAR_MODEL_PATH="${MODEL_PATH}"
  export SDAR_MODEL_NAME="$(basename "${MODEL_PATH}")"
  export SDAR_EVAL_SCOPE="gsm8k"
  export SDAR_EVAL_CONFIG="configs/eval_sdar_gap_hf.py"
  export SDAR_USE_REMASK="true"
  export SDAR_EVAL_GPUS="1"
  export SDAR_INFER_BATCH_SIZE="1"
  export SDAR_FULL_DATASET="false"
  export SDAR_TEST_RANGE="${TEST_RANGE}"
  export SDAR_CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD}"
  export SDAR_REMASK_THRESHOLD="${threshold}"
  export SDAR_REMASK_START_RATIO="0.0"
  export SDAR_REMASK_START_TOKENS="${REMASK_START_TOKENS}"
  export SDAR_REMASK_PREFIX_GUARD_TOKENS="${REMASK_PREFIX_GUARD_TOKENS}"
  export SDAR_REMASK_TAIL_GUARD_BLOCKS="${REMASK_TAIL_GUARD_BLOCKS}"
  export SDAR_REMASK_INTERVAL_BLOCKS="${REMASK_INTERVAL_BLOCKS}"
  export SDAR_REMASK_WINDOW_BLOCKS="${REMASK_WINDOW_BLOCKS}"
  export SDAR_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"
  export SDAR_RECORD_REMASK="true"
  export SDAR_RECORD_REMASK_EVENTS="true"
  export SDAR_WORK_DIR="${case_root}"
  unset SDAR_FORWARD_COUNT_PATH
  unset SDAR_FORWARD_COUNT_SUMMARY_PATH
  unset SDAR_REMASK_TRACE_PATH
  unset SDAR_REMASK_EVENT_TRACE_PATH
  unset SDAR_REMASK_SUMMARY_PATH

  echo "[${label}] threshold=${threshold}" >&2
  bash "${REPO_ROOT}/script/test_gap.sh"
}

run_case "rt0_90" "0.90"
run_case "rt1_00" "1.00"

python - "${WORK_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

work_root = Path(sys.argv[1])
print("label\taccuracy\texamples_with_remask\ttotal_forward_calls")
for label in ("rt0_90", "rt1_00"):
    case_root = work_root / label
    result_path = next(case_root.glob("**/results/*/gsm8k.json"))
    summary_path = next(case_root.glob("**/summary/forward_count_summary.txt"))
    remask_summary_path = next(case_root.glob("**/summary/remask_summary.txt"))
    accuracy = json.loads(result_path.read_text(encoding="utf-8"))["accuracy"]
    forward_total = None
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("total_forward_calls="):
            forward_total = int(line.split("=", 1)[1])
            break
    remask_examples = None
    for line in remask_summary_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("examples_with_remask="):
            remask_examples = int(line.split("=", 1)[1])
            break
    print(f"{label}\t{accuracy}\t{remask_examples}\t{forward_total}")
PY
