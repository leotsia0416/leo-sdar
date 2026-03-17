#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_train
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/train_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

REPO_ROOT="/work/leotsia0416/projects/SDAR"
DEFAULT_TRAIN_ENV_PREFIX="/home/leotsia0416/.conda/envs/sdar"
TRAIN_ENV_PREFIX="${TRAIN_ENV_PREFIX:-${DEFAULT_TRAIN_ENV_PREFIX}}"
VENDOR_PYTHONPATH="${VENDOR_PYTHONPATH:-/work/leotsia0416/projects/SDAR/vendor}"
TRAIN_HF_HOME="${TRAIN_HF_HOME:-${REPO_ROOT}/Models/hf}"
HF_HOME="${TRAIN_HF_HOME}"
HF_MODULES_CACHE="${HF_HOME}/modules"
export HF_HOME HF_MODULES_CACHE
unset TRANSFORMERS_CACHE
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
SAVE_STEPS="${SAVE_STEPS:-50}"

yaml_get() {
    local key="$1"
    sed -n "s/^${key}:[[:space:]]*//p" "${CONFIG_FILE}" | head -n 1 | sed "s/^['\"]//; s/['\"]$//"
}

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

TRAIN_HEAD_ONLY="$(normalize_bool "${TRAIN_HEAD_ONLY:-false}")"
HEAD_ONLY_TRAINABLE_MODULES="${HEAD_ONLY_TRAINABLE_MODULES:-gap_remask_head}"
TRAIN_DATASET="${TRAIN_DATASET:-open_r1_math,gsm8k_train_local}"
TRAIN_DATASET_TAG="${TRAIN_DATASET//[^A-Za-z0-9_.-]/_}"
TRAIN_TOKENIZED_PATH="${TRAIN_TOKENIZED_PATH:-${REPO_ROOT}/cache/tokenized_${TRAIN_DATASET_TAG}_cutoff2048_gap_pack}"
TRAIN_RUN_NAME_BASE="${TRAIN_RUN_NAME_BASE:-}"
TRAIN_MIX_STRATEGY="${TRAIN_MIX_STRATEGY:-concat}"
TRAIN_INTERLEAVE_PROBS="${TRAIN_INTERLEAVE_PROBS:-}"
TRAIN_ROLLOUT_SCOPE="${TRAIN_ROLLOUT_SCOPE:-$(yaml_get gap_rollout_scope)}"
TRAIN_REMASK_SCOPE="${TRAIN_REMASK_SCOPE:-$(yaml_get gap_remask_scope)}"
TRAIN_ROLLOUT_SCOPE="${TRAIN_ROLLOUT_SCOPE:-frontier_block}"
TRAIN_REMASK_SCOPE="${TRAIN_REMASK_SCOPE:-frontier_block}"

get_transformers_version() {
    local env_prefix="$1"
    if [[ ! -x "${env_prefix}/bin/python" ]]; then
        return 1
    fi
    PYTHONWARNINGS=ignore "${env_prefix}/bin/python" -c "import transformers; print(transformers.__version__)" 2>/dev/null
}

version_gte() {
    local env_prefix="$1"
    local version="$2"
    local target="$3"
    PYTHONWARNINGS=ignore "${env_prefix}/bin/python" - "$version" "$target" <<'PY'
import re
import sys

def normalize(version: str) -> tuple[int, ...]:
    parts = [int(x) for x in re.findall(r"\d+", version)]
    if len(parts) < 3:
        parts.extend([0] * (3 - len(parts)))
    return tuple(parts[:3])

print(int(normalize(sys.argv[1]) >= normalize(sys.argv[2])))
PY
}

ensure_training_env_compat() {
    local neat_packing_enabled
    neat_packing_enabled="$(normalize_bool "$(yaml_get neat_packing)")"

    local selected_version
    selected_version="$(get_transformers_version "${TRAIN_ENV_PREFIX}" || true)"
    if [[ -z "${selected_version}" ]]; then
        echo "Cannot determine transformers version from ${TRAIN_ENV_PREFIX}/bin/python" >&2
        exit 1
    fi

    if [[ "${neat_packing_enabled}" == "true" ]]; then
        local selected_incompatible
        selected_incompatible="$(version_gte "${TRAIN_ENV_PREFIX}" "${selected_version}" "4.53.0")"
        if [[ "${selected_incompatible}" == "1" ]]; then
            if [[ "${TRAIN_ENV_PREFIX}" != "${DEFAULT_TRAIN_ENV_PREFIX}" ]]; then
                local fallback_version
                fallback_version="$(get_transformers_version "${DEFAULT_TRAIN_ENV_PREFIX}" || true)"
                if [[ -z "${fallback_version}" ]]; then
                    echo "Selected TRAIN_ENV_PREFIX=${TRAIN_ENV_PREFIX} is incompatible with neat_packing (transformers ${selected_version}), and fallback env ${DEFAULT_TRAIN_ENV_PREFIX} is unavailable." >&2
                    exit 1
                fi

                local fallback_incompatible
                fallback_incompatible="$(version_gte "${DEFAULT_TRAIN_ENV_PREFIX}" "${fallback_version}" "4.53.0")"
                if [[ "${fallback_incompatible}" == "0" ]]; then
                    echo "TRAIN_ENV_PREFIX=${TRAIN_ENV_PREFIX} uses transformers ${selected_version}, incompatible with neat_packing. Falling back to ${DEFAULT_TRAIN_ENV_PREFIX} (transformers ${fallback_version})." >&2
                    TRAIN_ENV_PREFIX="${DEFAULT_TRAIN_ENV_PREFIX}"
                    selected_version="${fallback_version}"
                else
                    echo "Selected TRAIN_ENV_PREFIX=${TRAIN_ENV_PREFIX} is incompatible with neat_packing (transformers ${selected_version}), and fallback env ${DEFAULT_TRAIN_ENV_PREFIX} is also incompatible (transformers ${fallback_version})." >&2
                    exit 1
                fi
            else
                echo "TRAIN_ENV_PREFIX=${TRAIN_ENV_PREFIX} uses transformers ${selected_version}, incompatible with neat_packing. Use transformers<4.53.0, preferably 4.52.4." >&2
                exit 1
            fi
        fi
    fi

    echo "Using training env: ${TRAIN_ENV_PREFIX} (transformers ${selected_version})" >&2
}

ensure_training_env_compat

sed '/^launcher_[^:]*:/d' "${CONFIG_FILE}" | sed "s/%j/${job_id}/g" > "${TMP_CONFIG_FILE}"
python - "${TMP_CONFIG_FILE}" "${SAVE_STEPS}" "${TRAIN_HEAD_ONLY}" "${HEAD_ONLY_TRAINABLE_MODULES}" "${TRAIN_DATASET}" "${TRAIN_TOKENIZED_PATH}" "${TRAIN_RUN_NAME_BASE}" "${TRAIN_MIX_STRATEGY}" "${TRAIN_INTERLEAVE_PROBS}" "${TRAIN_ROLLOUT_SCOPE}" "${TRAIN_REMASK_SCOPE}" <<'PY'
import pathlib
import re
import sys

config_path = pathlib.Path(sys.argv[1])
save_steps = sys.argv[2]
train_head_only = sys.argv[3] == "true"
head_only_trainable_modules = sys.argv[4]
train_dataset = sys.argv[5]
train_tokenized_path = sys.argv[6]
train_run_name_base = sys.argv[7]
train_mix_strategy = sys.argv[8]
train_interleave_probs = sys.argv[9]
train_rollout_scope = sys.argv[10]
train_remask_scope = sys.argv[11]
text = config_path.read_text()


def set_key(text: str, key: str, value: str) -> str:
    updated, count = re.subn(
        rf"^{re.escape(key)}:\s*.*$",
        f"{key}: {value}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        if not updated.endswith("\n"):
            updated += "\n"
        updated += f"{key}: {value}\n"
    return updated


updated = set_key(text, "save_steps", save_steps)
updated = set_key(updated, "dataset", train_dataset)
updated = set_key(updated, "tokenized_path", train_tokenized_path)
updated = set_key(updated, "mix_strategy", train_mix_strategy)
updated = set_key(updated, "interleave_probs", train_interleave_probs if train_interleave_probs else "null")
updated = set_key(updated, "gap_rollout_scope", train_rollout_scope)
updated = set_key(updated, "gap_remask_scope", train_remask_scope)

if train_run_name_base:
    run_name_base = train_run_name_base
else:
    safe_dataset_name = re.sub(r"[^A-Za-z0-9]+", "_", train_dataset).strip("_") or "dataset"
    run_name_base = f"sdar_1p7b_{safe_dataset_name}_gap"

if train_head_only:
    updated = set_key(updated, "finetuning_type", "freeze")
    updated = set_key(updated, "freeze_trainable_layers", "0")
    updated = set_key(updated, "freeze_trainable_modules", "all")
    updated = set_key(updated, "freeze_extra_modules", head_only_trainable_modules)
    updated = set_key(updated, "run_name", f"{run_name_base}_head_only")
else:
    updated = set_key(updated, "finetuning_type", "full")
    updated = set_key(updated, "freeze_language_model", "false")
    updated = set_key(updated, "run_name", run_name_base)

config_path.write_text(updated)
PY

echo "Training dataset: ${TRAIN_DATASET}" >&2
echo "Tokenized cache: ${TRAIN_TOKENIZED_PATH}" >&2
echo "Mix strategy: ${TRAIN_MIX_STRATEGY}" >&2
if [[ -n "${TRAIN_INTERLEAVE_PROBS}" ]]; then
    echo "Interleave probs: ${TRAIN_INTERLEAVE_PROBS}" >&2
fi
echo "Rollout scope: ${TRAIN_ROLLOUT_SCOPE}" >&2
echo "Remask scope: ${TRAIN_REMASK_SCOPE}" >&2
if [[ "${TRAIN_HEAD_ONLY}" == "true" ]]; then
    echo "Head-only training enabled: ${HEAD_ONLY_TRAINABLE_MODULES}" >&2
else
    echo "Head-only training disabled." >&2
fi

cd "$(yaml_get launcher_workdir)"
export PYTHONPATH="${REPO_ROOT}:$(pwd)/src:${PYTHONPATH:-}"
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
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The 'repr' attribute with value False was provided to the `Field\\(\\)` function.*:UserWarning,ignore:The 'frozen' attribute with value True was provided to the `Field\\(\\)` function.*:UserWarning}"
export DISABLE_VERSION_CHECK=1

if [[ ! -x "${TRAIN_ENV_PREFIX}/bin/torchrun" ]]; then
    echo "Cannot find torchrun in ${TRAIN_ENV_PREFIX}/bin/torchrun" >&2
    exit 1
fi
export PYTHONNOUSERSITE=1

master_port="${MASTER_PORT:-}"
if [[ -z "${master_port}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        master_port="$((10000 + (SLURM_JOB_ID % 50000)))"
    else
        master_port="$(yaml_get launcher_master_port)"
    fi
fi

nnodes="${SLURM_JOB_NUM_NODES:-$(yaml_get launcher_nnodes)}"
if [[ "${nnodes}" == "1" ]]; then
    candidate_port="${master_port}"
    for _attempt in $(seq 0 31); do
        if python -c "import socket, sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); 
try:
    s.bind(('127.0.0.1', int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    s.close()" "${candidate_port}"; then
            master_port="${candidate_port}"
            break
        fi
        candidate_port="$((candidate_port + 1))"
        if [[ "${candidate_port}" -gt 65535 ]]; then
            candidate_port=10000
        fi
    done
fi

"${TRAIN_ENV_PREFIX}/bin/torchrun" \
    --nnodes "${nnodes}" \
    --node_rank "${SLURM_NODEID:-$(yaml_get launcher_node_rank)}" \
    --nproc_per_node "${SLURM_GPUS_ON_NODE:-$(yaml_get launcher_nproc_per_node)}" \
    --master_addr "${MASTER_ADDR:-$(yaml_get launcher_master_addr)}" \
    --master_port "${master_port}" \
    "$(yaml_get launcher_script)" \
    "${TMP_CONFIG_FILE}"
