#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_train
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/train_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

REPO_ROOT="/work/leotsia0416/projects/SDAR"
TRAIN_ENV_PREFIX="${TRAIN_ENV_PREFIX:-/work/leotsia0416/sdar_eval}"
VENDOR_PYTHONPATH="${VENDOR_PYTHONPATH:-/work/leotsia0416/projects/SDAR/vendor}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/config/training.yaml" ]]; then
    if [[ -f "${REPO_ROOT}/script/config/training.yaml" ]]; then
        SCRIPT_DIR="${REPO_ROOT}/script"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/script/config/training.yaml" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/script"
    else
        echo "Cannot locate training config from SCRIPT_DIR=${SCRIPT_DIR} or SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}" >&2
        exit 1
    fi
fi
CONFIG_FILE="${SCRIPT_DIR}/config/training.yaml"
TMP_CONFIG_FILE="$(mktemp --suffix=.yaml)"
trap 'rm -f "${TMP_CONFIG_FILE}"' EXIT
job_id="${SLURM_JOB_ID:-local}"

yaml_get() {
    local key="$1"
    sed -n "s/^${key}:[[:space:]]*//p" "${CONFIG_FILE}" | head -n 1 | sed "s/^['\"]//; s/['\"]$//"
}

sed '/^launcher_[^:]*:/d' "${CONFIG_FILE}" | sed "s/%j/${job_id}/g" > "${TMP_CONFIG_FILE}"

cd "$(yaml_get launcher_workdir)"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
if [[ -d "${VENDOR_PYTHONPATH}" ]]; then
    export PYTHONPATH="${PYTHONPATH}:${VENDOR_PYTHONPATH}"
fi
export CUDA_HOME="${CUDA_HOME:-/work/HPC_software/LMOD/nvidia/packages/cuda-12.6}"
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "Cannot find nvcc in ${CUDA_HOME}/bin/nvcc" >&2
    exit 1
fi
export PATH="${CUDA_HOME}/bin:${TRAIN_ENV_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DISABLE_VERSION_CHECK=1

if [[ ! -x "${TRAIN_ENV_PREFIX}/bin/torchrun" ]]; then
    echo "Cannot find torchrun in ${TRAIN_ENV_PREFIX}/bin/torchrun" >&2
    exit 1
fi
export PYTHONNOUSERSITE=1

"${TRAIN_ENV_PREFIX}/bin/torchrun" \
    --nnodes "${SLURM_JOB_NUM_NODES:-$(yaml_get launcher_nnodes)}" \
    --node_rank "${SLURM_NODEID:-$(yaml_get launcher_node_rank)}" \
    --nproc_per_node "${SLURM_GPUS_ON_NODE:-$(yaml_get launcher_nproc_per_node)}" \
    --master_addr "${MASTER_ADDR:-$(yaml_get launcher_master_addr)}" \
    --master_port "${MASTER_PORT:-$(yaml_get launcher_master_port)}" \
    "$(yaml_get launcher_script)" \
    "${TMP_CONFIG_FILE}"
