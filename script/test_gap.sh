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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config/eval.yaml"
LIB_FILE="${SCRIPT_DIR}/lib/test_gap_common.sh"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  if [[ -f "${REPO_ROOT}/script/config/eval.yaml" ]]; then
    CONFIG_FILE="${REPO_ROOT}/script/config/eval.yaml"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/script/config/eval.yaml" ]]; then
    CONFIG_FILE="${SLURM_SUBMIT_DIR}/script/config/eval.yaml"
  else
    echo "Cannot locate eval config from SCRIPT_DIR=${SCRIPT_DIR} or SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}" >&2
    exit 1
  fi
fi

if [[ ! -f "${LIB_FILE}" ]]; then
  if [[ -f "${REPO_ROOT}/script/lib/test_gap_common.sh" ]]; then
    LIB_FILE="${REPO_ROOT}/script/lib/test_gap_common.sh"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/script/lib/test_gap_common.sh" ]]; then
    LIB_FILE="${SLURM_SUBMIT_DIR}/script/lib/test_gap_common.sh"
  else
    echo "Cannot locate test_gap common library from SCRIPT_DIR=${SCRIPT_DIR} or SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}" >&2
    exit 1
  fi
fi

source "${LIB_FILE}"

resolve_model_context
apply_eval_runtime_defaults
LAUNCH_TS="$(date +%s)"
prepare_eval_run_bundle
prepare_eval_artifact_paths
log_eval_context
run_eval_job
copy_eval_config_artifacts
summarize_eval_artifacts
