#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_eval_lmdeploy
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/test_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

REPO_ROOT="/work/leotsia0416/projects/SDAR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config/eval.yaml"
EVAL_ENV_PREFIX="${EVAL_ENV_PREFIX:-/work/leotsia0416/sdar_eval}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  CONFIG_FILE="${REPO_ROOT}/script/config/eval.yaml"
fi

yaml_get() {
  local key="$1"
  sed -n "s/^${key}:[[:space:]]*//p" "${CONFIG_FILE}" | head -n 1 | sed "s/^['\"]//; s/['\"]$//"
}

sanitize_name() {
  local value="${1:-unnamed}"
  value="${value,,}"
  value="$(printf '%s' "${value}" | sed 's/[^a-z0-9._-]/_/g; s/__*/_/g; s/^_//; s/_$//')"
  if [[ -z "${value}" ]]; then
    value="unnamed"
  fi
  printf '%s\n' "${value}"
}

link_bundle_path() {
  local target="${1:?missing target}"
  local link_path="${2:?missing link path}"
  mkdir -p "$(dirname "${link_path}")"
  rm -f "${link_path}"
  ln -s "${target}" "${link_path}"
}

resolve_work_dir_abs() {
  local work_dir="${1:?missing work dir}"
  case "${work_dir}" in
    /*) printf '%s\n' "${work_dir}" ;;
    *) printf '%s\n' "$(pwd)/${work_dir#./}" ;;
  esac
}

locate_exp_dir() {
  local work_dir="$1"
  local launch_ts="$2"
  local candidate=""
  local candidate_ts=""
  local best_dir=""
  local best_ts=0

  while IFS= read -r candidate; do
    candidate_ts="$(stat -c %Y "${candidate}" 2>/dev/null || echo 0)"
    if [[ "${candidate_ts}" -ge "${launch_ts}" && "${candidate_ts}" -ge "${best_ts}" ]]; then
      best_dir="${candidate}"
      best_ts="${candidate_ts}"
    fi
  done < <(find "${work_dir}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)

  printf '%s\n' "${best_dir}"
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

write_eval_run_snapshot() {
  local output_path="$1"
  mkdir -p "$(dirname "${output_path}")"
  cat > "${output_path}" <<EOF
meta:
  generated_from: "script/test.sh"
  slurm_job_id: "${SLURM_JOB_ID:-local}"
  slurm_job_name: "${SLURM_JOB_NAME:-}"
  launch_ts: "${LAUNCH_TS}"
model:
  model_path: "${SDAR_MODEL_PATH}"
  model_root: "${SDAR_MODEL_ROOT}"
  model_name: "${SDAR_MODEL_NAME}"
eval:
  eval_scope: "${SDAR_EVAL_SCOPE}"
  eval_config: "configs/eval_sdar_lmdeploy.py"
  eval_gpus: "${SDAR_EVAL_GPUS}"
  infer_batch_size: "${SDAR_INFER_BATCH_SIZE}"
  confidence_threshold: "${SDAR_CONFIDENCE_THRESHOLD}"
  block_length: "${SDAR_BLOCK_LENGTH}"
  max_new_tokens: "${SDAR_MAX_NEW_TOKENS}"
work:
  work_dir: "${SDAR_WORK_DIR}"
  work_dir_abs: "${SDAR_WORK_DIR_ABS}"
  run_root: "${SDAR_RUN_ROOT}"
  run_bundle_dir: "${SDAR_RUN_BUNDLE_DIR}"
EOF
}

prepare_eval_run_bundle() {
  local run_root=""
  local run_stamp=""
  local run_label=""
  local slurm_log_prefix=""

  run_root="${SDAR_RUN_ROOT:-$(yaml_get run_root)}"
  run_root="${run_root:-${REPO_ROOT}/runs/eval}"
  mkdir -p "${run_root}"
  export SDAR_RUN_ROOT="${run_root}"

  if [[ -n "${SDAR_RUN_NAME:-}" ]]; then
    run_label="${SDAR_RUN_NAME}"
  else
    run_label="lmdeploy_${SDAR_EVAL_SCOPE:-eval}_${SDAR_MODEL_NAME}"
  fi

  run_stamp="$(date +%Y%m%d_%H%M%S)"
  run_label="$(sanitize_name "${run_label}")"
  export SDAR_RUN_BUNDLE_DIR="${SDAR_RUN_BUNDLE_DIR:-${SDAR_RUN_ROOT}/${run_stamp}_${run_label}_job${SLURM_JOB_ID:-local}}"
  mkdir -p "${SDAR_RUN_BUNDLE_DIR}"

  slurm_log_prefix="${REPO_ROOT}/logs/test_${SLURM_JOB_ID:-local}"
  link_bundle_path "${slurm_log_prefix}.out" "${SDAR_RUN_BUNDLE_DIR}/slurm.out"
  link_bundle_path "${slurm_log_prefix}.err" "${SDAR_RUN_BUNDLE_DIR}/slurm.err"
  link_bundle_path "${SDAR_WORK_DIR_ABS}" "${SDAR_RUN_BUNDLE_DIR}/work_dir"

  cat > "${SDAR_RUN_BUNDLE_DIR}/run_meta.yaml" <<EOF
runner: lmdeploy
generated_from: script/test.sh
slurm_job_id: "${SLURM_JOB_ID:-local}"
slurm_job_name: "${SLURM_JOB_NAME:-}"
launch_ts: "${LAUNCH_TS}"
model_path: "${SDAR_MODEL_PATH}"
model_name: "${SDAR_MODEL_NAME}"
eval_scope: "${SDAR_EVAL_SCOPE}"
work_dir: "${SDAR_WORK_DIR_ABS}"
run_root: "${SDAR_RUN_ROOT}"
EOF
}

finalize_eval_run_bundle() {
  local exp_dir="$1"

  if [[ -z "${exp_dir}" || ! -d "${exp_dir}" ]]; then
    return 0
  fi

  link_bundle_path "${exp_dir}" "${SDAR_RUN_BUNDLE_DIR}/exp_dir"
  [[ -d "${exp_dir}/summary" ]] && link_bundle_path "${exp_dir}/summary" "${SDAR_RUN_BUNDLE_DIR}/summary"
  [[ -d "${exp_dir}/results" ]] && link_bundle_path "${exp_dir}/results" "${SDAR_RUN_BUNDLE_DIR}/results"
  [[ -d "${exp_dir}/predictions" ]] && link_bundle_path "${exp_dir}/predictions" "${SDAR_RUN_BUNDLE_DIR}/predictions"
  [[ -d "${exp_dir}/logs" ]] && link_bundle_path "${exp_dir}/logs" "${SDAR_RUN_BUNDLE_DIR}/logs"
}

export PATH="${EVAL_ENV_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
cd /work/leotsia0416/projects/SDAR/evaluation/opencompass
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export SDAR_MODEL_PATH="${SDAR_MODEL_PATH:-/work/leotsia0416/projects/SDAR/Models/SDAR-1.7B-Chat-}"
validate_hf_model_dir "${SDAR_MODEL_PATH}"
export SDAR_MODEL_ROOT="${SDAR_MODEL_ROOT:-$(dirname "${SDAR_MODEL_PATH}")}"
export SDAR_MODEL_NAME="${SDAR_MODEL_NAME:-$(basename "${SDAR_MODEL_PATH}")}"
export SDAR_EVAL_SCOPE="${SDAR_EVAL_SCOPE:-gsm8k}"
export SDAR_EVAL_GPUS="${SDAR_EVAL_GPUS:-2}"
export SDAR_INFER_BATCH_SIZE="${SDAR_INFER_BATCH_SIZE:-1}"
export SDAR_CONFIDENCE_THRESHOLD="${SDAR_CONFIDENCE_THRESHOLD:-0.95}"
export SDAR_BLOCK_LENGTH="${SDAR_BLOCK_LENGTH:-4}"
export SDAR_MAX_NEW_TOKENS="${SDAR_MAX_NEW_TOKENS:-1536}"
export SDAR_WORK_DIR="${SDAR_WORK_DIR:-./outputs/eval-chat-sdar}"
export SDAR_WORK_DIR_ABS="$(resolve_work_dir_abs "${SDAR_WORK_DIR}")"
mkdir -p "${SDAR_WORK_DIR_ABS}"

LAUNCH_TS="$(date +%s)"
prepare_eval_run_bundle
write_eval_run_snapshot "${SDAR_RUN_BUNDLE_DIR}/eval_run_config.yaml"
cp "${CONFIG_FILE}" "${SDAR_RUN_BUNDLE_DIR}/eval_defaults.yaml"

export HF_HOME="${SDAR_MODEL_ROOT}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$SDAR_MODEL_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1200
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"

echo "Using model path: ${SDAR_MODEL_PATH}" >&2
echo "Run bundle dir: ${SDAR_RUN_BUNDLE_DIR}" >&2
echo "Work dir: ${SDAR_WORK_DIR_ABS}" >&2

/work/leotsia0416/sdar_eval/bin/python run.py configs/eval_sdar_lmdeploy.py

EXP_DIR="$(locate_exp_dir "${SDAR_WORK_DIR_ABS}" "${LAUNCH_TS}")"
finalize_eval_run_bundle "${EXP_DIR}"
