#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J SDAR_train
#SBATCH -p normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
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
    local label="${2:-Model path}"
    if [[ ! -d "${model_dir}" ]]; then
        echo "${label} does not exist: ${model_dir}" >&2
        exit 1
    fi
    if [[ ! -f "${model_dir}/config.json" ]]; then
        echo "${label} is missing config.json: ${model_dir}" >&2
        exit 1
    fi
    if ! has_model_weights "${model_dir}"; then
        echo "${label} is missing model weights: ${model_dir}" >&2
        exit 1
    fi
}

TRAIN_HEAD_ONLY="$(normalize_bool "${TRAIN_HEAD_ONLY:-false}")"
HEAD_ONLY_TRAINABLE_MODULES="${HEAD_ONLY_TRAINABLE_MODULES:-gap_remask_head}"
TRAIN_PROFILE="${TRAIN_PROFILE:-hard_math}"
HARD_MATH_RAW_DATASET_NAME="${HARD_MATH_RAW_DATASET_NAME:-math_train_hard_local}"
HARD_MATH_DATASET_NAME="${HARD_MATH_DATASET_NAME:-math_train_hard_boxed_prompt_local}"
DEFAULT_TRAIN_DATASET="${HARD_MATH_DATASET_NAME}"
BASE_MODEL_PATH="${REPO_ROOT}/Models/SDAR-1.7B-Chat-"
DEFAULT_TRAIN_CHECKPOINT_PATH="/work/leotsia0416/projects/SDAR/checkpoint/training_141209/checkpoint-666"
ORCA_MATH_DATASET_NAME="${ORCA_MATH_DATASET_NAME:-orca_math_200k_local}"
HARD_MATH_SOURCE_PATH="${HARD_MATH_SOURCE_PATH:-${REPO_ROOT}/training/llama_factory_sdar/data/${HARD_MATH_RAW_DATASET_NAME}.jsonl}"
HARD_MATH_OUTPUT_PATH="${HARD_MATH_OUTPUT_PATH:-${REPO_ROOT}/training/llama_factory_sdar/data/${HARD_MATH_DATASET_NAME}.jsonl}"
HARD_MATH_REBUILD="$(normalize_bool "${HARD_MATH_REBUILD:-false}")"
FORMAT_ALIGN_DATASET_NAME="${FORMAT_ALIGN_DATASET_NAME:-gsm8k_boxed_format_align}"
FORMAT_ALIGN_SOURCE_PATH="${FORMAT_ALIGN_SOURCE_PATH:-/work/leotsia0416/datasets/gsm8k/train.jsonl}"
FORMAT_ALIGN_OUTPUT_PATH="${FORMAT_ALIGN_OUTPUT_PATH:-${REPO_ROOT}/training/llama_factory_sdar/data/${FORMAT_ALIGN_DATASET_NAME}.json}"
FORMAT_ALIGN_REBUILD="$(normalize_bool "${FORMAT_ALIGN_REBUILD:-false}")"
FORMAT_ALIGN_MAX_REASONING_LINES="${FORMAT_ALIGN_MAX_REASONING_LINES:-6}"
FORMAT_ALIGN_MAX_SAMPLES="${FORMAT_ALIGN_MAX_SAMPLES:-}"
TRAIN_CHECKPOINT_PATH="${TRAIN_CHECKPOINT_PATH:-${DEFAULT_TRAIN_CHECKPOINT_PATH}}"
TRAIN_MODEL_PATH="${TRAIN_MODEL_PATH:-}"
TRAIN_RESUME_FROM_CHECKPOINT="${TRAIN_RESUME_FROM_CHECKPOINT:-}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-}"
TRAIN_NUM_TRAIN_EPOCHS="${TRAIN_NUM_TRAIN_EPOCHS:-}"
TRAIN_CUTOFF_LEN="${TRAIN_CUTOFF_LEN:-}"
TRAIN_OVERWRITE_CACHE="${TRAIN_OVERWRITE_CACHE:-}"
TRAIN_DATASET="${TRAIN_DATASET:-${DEFAULT_TRAIN_DATASET}}"
TRAIN_RUN_NAME_BASE="${TRAIN_RUN_NAME_BASE:-}"
DEFAULT_TRAIN_MIX_STRATEGY="concat"
TRAIN_MIX_STRATEGY="${TRAIN_MIX_STRATEGY:-${DEFAULT_TRAIN_MIX_STRATEGY}}"
TRAIN_INTERLEAVE_PROBS="${TRAIN_INTERLEAVE_PROBS:-}"
TRAIN_NEAT_PACKING="${TRAIN_NEAT_PACKING:-$(yaml_get neat_packing)}"
TRAIN_GAP_TRAINING_MODE="${TRAIN_GAP_TRAINING_MODE:-$(yaml_get gap_training_mode)}"
TRAIN_GAP_ROLLOUT_STEPS="${TRAIN_GAP_ROLLOUT_STEPS:-$(yaml_get gap_rollout_steps)}"
TRAIN_GAP_ROLLOUT_STRATEGY="${TRAIN_GAP_ROLLOUT_STRATEGY:-$(yaml_get gap_rollout_strategy)}"
TRAIN_GAP_ROLLOUT_CONFIDENCE_THRESHOLD="${TRAIN_GAP_ROLLOUT_CONFIDENCE_THRESHOLD:-$(yaml_get gap_rollout_confidence_threshold)}"
TRAIN_ROLLOUT_SCOPE="${TRAIN_ROLLOUT_SCOPE:-$(yaml_get gap_rollout_scope)}"
TRAIN_GAP_REVEAL_RATIO="${TRAIN_GAP_REVEAL_RATIO:-$(yaml_get gap_reveal_ratio)}"
TRAIN_GAP_MIN_REVEAL_TOKENS="${TRAIN_GAP_MIN_REVEAL_TOKENS:-$(yaml_get gap_min_reveal_tokens)}"
TRAIN_GAP_REMASK_THRESHOLD="${TRAIN_GAP_REMASK_THRESHOLD:-$(yaml_get gap_remask_threshold)}"
TRAIN_GAP_REMASK_LOSS_WEIGHT="${TRAIN_GAP_REMASK_LOSS_WEIGHT:-$(yaml_get gap_remask_loss_weight)}"
TRAIN_GAP_GLOBAL_LOSS_WEIGHT="${TRAIN_GAP_GLOBAL_LOSS_WEIGHT:-$(yaml_get gap_global_loss_weight)}"
TRAIN_GAP_DIFFUSION_LOSS_WEIGHT="${TRAIN_GAP_DIFFUSION_LOSS_WEIGHT:-$(yaml_get gap_diffusion_loss_weight)}"
TRAIN_GAP_REMASK_DEFAULT_P_MASK="${TRAIN_GAP_REMASK_DEFAULT_P_MASK:-$(yaml_get gap_remask_default_p_mask)}"
TRAIN_GAP_GRPO_LOSS_WEIGHT="${TRAIN_GAP_GRPO_LOSS_WEIGHT:-$(yaml_get gap_grpo_loss_weight)}"
TRAIN_GAP_GRPO_NUM_SAMPLES="${TRAIN_GAP_GRPO_NUM_SAMPLES:-$(yaml_get gap_grpo_num_samples)}"
TRAIN_GAP_GRPO_CLIP_EPS="${TRAIN_GAP_GRPO_CLIP_EPS:-$(yaml_get gap_grpo_clip_eps)}"
TRAIN_GAP_GRPO_ENTROPY_COEF="${TRAIN_GAP_GRPO_ENTROPY_COEF:-$(yaml_get gap_grpo_entropy_coef)}"
TRAIN_GAP_GRPO_REMASK_PENALTY="${TRAIN_GAP_GRPO_REMASK_PENALTY:-$(yaml_get gap_grpo_remask_penalty)}"
TRAIN_GAP_GRPO_ADVANTAGE_EPS="${TRAIN_GAP_GRPO_ADVANTAGE_EPS:-$(yaml_get gap_grpo_advantage_eps)}"
TRAIN_GAP_GRPO_SAMPLE_PROB_EPS="${TRAIN_GAP_GRPO_SAMPLE_PROB_EPS:-$(yaml_get gap_grpo_sample_prob_eps)}"
TRAIN_GAP_GRPO_DENSE_REWARD_WEIGHT="${TRAIN_GAP_GRPO_DENSE_REWARD_WEIGHT:-$(yaml_get gap_grpo_dense_reward_weight)}"
TRAIN_GAP_GRPO_TERMINAL_REWARD_WEIGHT="${TRAIN_GAP_GRPO_TERMINAL_REWARD_WEIGHT:-$(yaml_get gap_grpo_terminal_reward_weight)}"
TRAIN_GAP_GRPO_FORMAT_REWARD_WEIGHT="${TRAIN_GAP_GRPO_FORMAT_REWARD_WEIGHT:-$(yaml_get gap_grpo_format_reward_weight)}"
TRAIN_DISABLE_GRADIENT_CHECKPOINTING="${TRAIN_DISABLE_GRADIENT_CHECKPOINTING:-$(yaml_get disable_gradient_checkpointing)}"
TRAIN_GRADIENT_CHECKPOINTING="${TRAIN_GRADIENT_CHECKPOINTING:-$(yaml_get gradient_checkpointing)}"
TRAIN_USE_REENTRANT_GC="${TRAIN_USE_REENTRANT_GC:-$(yaml_get use_reentrant_gc)}"
TRAIN_GRADIENT_CHECKPOINTING_KWARGS="${TRAIN_GRADIENT_CHECKPOINTING_KWARGS:-$(yaml_get gradient_checkpointing_kwargs)}"
TRAIN_PER_DEVICE_TRAIN_BATCH_SIZE="${TRAIN_PER_DEVICE_TRAIN_BATCH_SIZE:-$(yaml_get per_device_train_batch_size)}"
TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-$(yaml_get gradient_accumulation_steps)}"
TRAIN_ROLLOUT_SCOPE="${TRAIN_ROLLOUT_SCOPE:-frontier_block}"
TRAIN_REMASK_SCOPE="${TRAIN_REMASK_SCOPE:-frontier_block}"

if [[ -n "${TRAIN_CHECKPOINT_PATH}" ]]; then
    TRAIN_MODEL_PATH="${TRAIN_MODEL_PATH:-${TRAIN_CHECKPOINT_PATH}}"
fi

if [[ "${TRAIN_PROFILE}" == "hard_math" ]]; then
    if [[ "${TRAIN_DATASET}" == "${DEFAULT_TRAIN_DATASET}" ]]; then
        TRAIN_DATASET="${HARD_MATH_DATASET_NAME}"
    fi
    TRAIN_MIX_STRATEGY="${DEFAULT_TRAIN_MIX_STRATEGY}"
    TRAIN_INTERLEAVE_PROBS=""
    TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-5e-6}"
    TRAIN_NUM_TRAIN_EPOCHS="${TRAIN_NUM_TRAIN_EPOCHS:-3.0}"
    TRAIN_CUTOFF_LEN="${TRAIN_CUTOFF_LEN:-2048}"
    TRAIN_RUN_NAME_BASE="${TRAIN_RUN_NAME_BASE:-sdar_1p7b_hard_math_gap_resume666_genwin3_grpo_single8_nopack_nogc_b2ga16}"
elif [[ "${TRAIN_PROFILE}" == "format_align" ]]; then
    if [[ "${TRAIN_DATASET}" == "${DEFAULT_TRAIN_DATASET}" ]]; then
        TRAIN_DATASET="${FORMAT_ALIGN_DATASET_NAME}"
    fi
    TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-1e-5}"
    TRAIN_NUM_TRAIN_EPOCHS="${TRAIN_NUM_TRAIN_EPOCHS:-1.0}"
    TRAIN_CUTOFF_LEN="${TRAIN_CUTOFF_LEN:-1024}"
    TRAIN_RUN_NAME_BASE="${TRAIN_RUN_NAME_BASE:-sdar_1p7b_${FORMAT_ALIGN_DATASET_NAME}_gap}"
elif [[ "${TRAIN_PROFILE}" == "format_align_mix" ]]; then
    if [[ "${TRAIN_DATASET}" == "${DEFAULT_TRAIN_DATASET}" ]]; then
        TRAIN_DATASET="open_r1_math,gsm8k_train_local,${FORMAT_ALIGN_DATASET_NAME}"
    fi
    if [[ "${TRAIN_MIX_STRATEGY}" == "${DEFAULT_TRAIN_MIX_STRATEGY}" ]]; then
        TRAIN_MIX_STRATEGY="interleave_under"
    fi
    TRAIN_INTERLEAVE_PROBS="${TRAIN_INTERLEAVE_PROBS:-0.7,0.2,0.1}"
    TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-2e-6}"
    TRAIN_CUTOFF_LEN="${TRAIN_CUTOFF_LEN:-2048}"
    TRAIN_RUN_NAME_BASE="${TRAIN_RUN_NAME_BASE:-sdar_1p7b_gsm8k_boxed_format_align_mix_gap_resume666}"
elif [[ "${TRAIN_PROFILE}" == "orca_math" ]]; then
    if [[ "${TRAIN_DATASET}" == "${DEFAULT_TRAIN_DATASET}" ]]; then
        TRAIN_DATASET="${ORCA_MATH_DATASET_NAME}"
    fi
    TRAIN_MIX_STRATEGY="${DEFAULT_TRAIN_MIX_STRATEGY}"
    TRAIN_INTERLEAVE_PROBS=""
    TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-2e-6}"
    TRAIN_CUTOFF_LEN="${TRAIN_CUTOFF_LEN:-1024}"
    TRAIN_RUN_NAME_BASE="${TRAIN_RUN_NAME_BASE:-sdar_1p7b_orca_math_gap_resume666}"
elif [[ "${TRAIN_PROFILE}" == "orca_math_remask_from_base" || "${TRAIN_PROFILE}" == "orca_math_puma_remask_from_base" ]]; then
    if [[ "${TRAIN_DATASET}" == "${DEFAULT_TRAIN_DATASET}" ]]; then
        TRAIN_DATASET="${ORCA_MATH_DATASET_NAME}"
    fi
    TRAIN_MIX_STRATEGY="${DEFAULT_TRAIN_MIX_STRATEGY}"
    TRAIN_INTERLEAVE_PROBS=""
    if [[ -z "${TRAIN_MODEL_PATH}" || "${TRAIN_MODEL_PATH}" == "${DEFAULT_TRAIN_CHECKPOINT_PATH}" ]]; then
        TRAIN_MODEL_PATH="${BASE_MODEL_PATH}"
    fi
    TRAIN_CHECKPOINT_PATH=""
    TRAIN_RESUME_FROM_CHECKPOINT=""
    TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-5e-6}"
    TRAIN_CUTOFF_LEN="${TRAIN_CUTOFF_LEN:-1024}"
    if [[ "${TRAIN_PROFILE}" == "orca_math_puma_remask_from_base" ]]; then
        TRAIN_RUN_NAME_BASE="${TRAIN_RUN_NAME_BASE:-sdar_1p7b_orca_math_gap_puma_remask_from_base}"
    else
        TRAIN_RUN_NAME_BASE="${TRAIN_RUN_NAME_BASE:-sdar_1p7b_orca_math_gap_remask_from_base}"
    fi
    TRAIN_GAP_TRAINING_MODE="remask"
    TRAIN_GAP_PUMA_STREAMING="true"
    TRAIN_ROLLOUT_SCOPE="all"
    TRAIN_REMASK_SCOPE="frontier_block"
    TRAIN_GAP_GLOBAL_LOSS_WEIGHT="${TRAIN_GAP_GLOBAL_LOSS_WEIGHT:-0.0}"
    TRAIN_GAP_REMASK_LOSS_WEIGHT="${TRAIN_GAP_REMASK_LOSS_WEIGHT:-1.0}"
fi

if [[ -n "${TRAIN_MAX_STEPS:-}" ]]; then
    echo "Ignoring TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS}; training now uses num_train_epochs only." >&2
fi

if [[ -n "${TRAIN_MODEL_PATH}" ]]; then
    validate_hf_model_dir "${TRAIN_MODEL_PATH}" "TRAIN_MODEL_PATH"
    TRAIN_MODEL_PATH="$(realpath "${TRAIN_MODEL_PATH}")"
fi

if [[ -n "${TRAIN_RESUME_FROM_CHECKPOINT}" ]]; then
    if [[ ! -d "${TRAIN_RESUME_FROM_CHECKPOINT}" ]]; then
        echo "TRAIN_RESUME_FROM_CHECKPOINT must be a checkpoint directory: ${TRAIN_RESUME_FROM_CHECKPOINT}" >&2
        exit 1
    fi
    TRAIN_RESUME_FROM_CHECKPOINT="$(realpath "${TRAIN_RESUME_FROM_CHECKPOINT}")"
fi

TRAIN_RUN_NAME_BASE="${TRAIN_RUN_NAME_BASE:-sdar_1p7b_gap}"

TRAIN_DATASET_TAG="${TRAIN_DATASET//[^A-Za-z0-9_.-]/_}"
if [[ -n "${TRAIN_CUTOFF_LEN}" ]]; then
    TOKENIZED_CUTOFF_LEN="${TRAIN_CUTOFF_LEN}"
else
    TOKENIZED_CUTOFF_LEN="$(yaml_get cutoff_len)"
fi
TOKENIZED_CUTOFF_LEN="${TOKENIZED_CUTOFF_LEN:-2048}"
TOKENIZED_LAYOUT_SUFFIX="gap_nopack"
if [[ "$(normalize_bool "${TRAIN_NEAT_PACKING}")" == "true" ]]; then
    TOKENIZED_LAYOUT_SUFFIX="gap_pack"
fi
TRAIN_TOKENIZED_PATH="${TRAIN_TOKENIZED_PATH:-${REPO_ROOT}/cache/tokenized_${TRAIN_DATASET_TAG}_cutoff${TOKENIZED_CUTOFF_LEN}_${TOKENIZED_LAYOUT_SUFFIX}}"

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

prepare_hard_math_dataset() {
    local prepare_script="${SCRIPT_DIR}/prepare_hard_math_boxed_prompt.py"
    if [[ ! -f "${prepare_script}" ]]; then
        echo "Cannot find hard-math prompt preparation script: ${prepare_script}" >&2
        exit 1
    fi
    if [[ ! -f "${HARD_MATH_SOURCE_PATH}" ]]; then
        echo "Cannot find hard-math source data: ${HARD_MATH_SOURCE_PATH}" >&2
        exit 1
    fi

    echo "Preparing hard-math boxed-prompt dataset: ${HARD_MATH_OUTPUT_PATH}" >&2
    "${TRAIN_ENV_PREFIX}/bin/python" "${prepare_script}" \
        --input "${HARD_MATH_SOURCE_PATH}" \
        --output "${HARD_MATH_OUTPUT_PATH}"
}

prepare_format_align_dataset() {
    local prepare_script="${SCRIPT_DIR}/prepare_gsm8k_boxed_format_align.py"
    if [[ ! -f "${prepare_script}" ]]; then
        echo "Cannot find format-align dataset script: ${prepare_script}" >&2
        exit 1
    fi
    if [[ ! -f "${FORMAT_ALIGN_SOURCE_PATH}" ]]; then
        echo "Cannot find GSM8K source data: ${FORMAT_ALIGN_SOURCE_PATH}" >&2
        exit 1
    fi

    local cmd=(
        "${TRAIN_ENV_PREFIX}/bin/python" "${prepare_script}"
        --input "${FORMAT_ALIGN_SOURCE_PATH}"
        --output "${FORMAT_ALIGN_OUTPUT_PATH}"
        --max-reasoning-lines "${FORMAT_ALIGN_MAX_REASONING_LINES}"
    )
    if [[ -n "${FORMAT_ALIGN_MAX_SAMPLES}" ]]; then
        cmd+=(--max-samples "${FORMAT_ALIGN_MAX_SAMPLES}")
    fi

    echo "Preparing format-align dataset: ${FORMAT_ALIGN_OUTPUT_PATH}" >&2
    "${cmd[@]}"
}

USE_HARD_MATH_DATASET="false"
IFS=',' read -ra TRAIN_DATASET_ITEMS <<< "${TRAIN_DATASET}"
for TRAIN_DATASET_ITEM in "${TRAIN_DATASET_ITEMS[@]}"; do
    TRAIN_DATASET_ITEM="${TRAIN_DATASET_ITEM//[[:space:]]/}"
    if [[ "${TRAIN_DATASET_ITEM}" == "${HARD_MATH_DATASET_NAME}" ]]; then
        USE_HARD_MATH_DATASET="true"
        break
    fi
done

if [[ "${USE_HARD_MATH_DATASET}" == "true" ]]; then
    if [[ "${HARD_MATH_REBUILD}" == "true" || ! -f "${HARD_MATH_OUTPUT_PATH}" ]]; then
        prepare_hard_math_dataset
    else
        echo "Using existing hard-math boxed-prompt dataset: ${HARD_MATH_OUTPUT_PATH}" >&2
    fi
fi

USE_FORMAT_ALIGN_DATASET="false"
IFS=',' read -ra TRAIN_DATASET_ITEMS <<< "${TRAIN_DATASET}"
for TRAIN_DATASET_ITEM in "${TRAIN_DATASET_ITEMS[@]}"; do
    TRAIN_DATASET_ITEM="${TRAIN_DATASET_ITEM//[[:space:]]/}"
    if [[ "${TRAIN_DATASET_ITEM}" == "${FORMAT_ALIGN_DATASET_NAME}" ]]; then
        USE_FORMAT_ALIGN_DATASET="true"
        break
    fi
done

if [[ "${USE_FORMAT_ALIGN_DATASET}" == "true" ]]; then
    if [[ "${FORMAT_ALIGN_REBUILD}" == "true" || ! -f "${FORMAT_ALIGN_OUTPUT_PATH}" ]]; then
        prepare_format_align_dataset
    else
        echo "Using existing format-align dataset: ${FORMAT_ALIGN_OUTPUT_PATH}" >&2
    fi
fi

sed '/^launcher_[^:]*:/d' "${CONFIG_FILE}" | sed "s/%j/${job_id}/g" > "${TMP_CONFIG_FILE}"
python - "${TMP_CONFIG_FILE}" "${SAVE_STEPS}" "${TRAIN_HEAD_ONLY}" "${HEAD_ONLY_TRAINABLE_MODULES}" "${TRAIN_DATASET}" "${TRAIN_TOKENIZED_PATH}" "${TRAIN_RUN_NAME_BASE}" "${TRAIN_MIX_STRATEGY}" "${TRAIN_INTERLEAVE_PROBS}" "${TRAIN_GAP_TRAINING_MODE}" "${TRAIN_GAP_ROLLOUT_STEPS}" "${TRAIN_GAP_ROLLOUT_STRATEGY}" "${TRAIN_GAP_ROLLOUT_CONFIDENCE_THRESHOLD}" "${TRAIN_ROLLOUT_SCOPE}" "${TRAIN_GAP_REVEAL_RATIO}" "${TRAIN_GAP_MIN_REVEAL_TOKENS}" "${TRAIN_GAP_REMASK_THRESHOLD}" "${TRAIN_GAP_REMASK_LOSS_WEIGHT}" "${TRAIN_GAP_GLOBAL_LOSS_WEIGHT}" "${TRAIN_GAP_REMASK_DEFAULT_P_MASK}" "${TRAIN_GAP_GRPO_LOSS_WEIGHT}" "${TRAIN_GAP_GRPO_NUM_SAMPLES}" "${TRAIN_GAP_GRPO_CLIP_EPS}" "${TRAIN_GAP_GRPO_ENTROPY_COEF}" "${TRAIN_GAP_GRPO_REMASK_PENALTY}" "${TRAIN_GAP_GRPO_ADVANTAGE_EPS}" "${TRAIN_GAP_GRPO_SAMPLE_PROB_EPS}" "${TRAIN_GAP_GRPO_DENSE_REWARD_WEIGHT}" "${TRAIN_GAP_GRPO_TERMINAL_REWARD_WEIGHT}" "${TRAIN_DISABLE_GRADIENT_CHECKPOINTING}" "${TRAIN_GRADIENT_CHECKPOINTING}" "${TRAIN_USE_REENTRANT_GC}" "${TRAIN_GRADIENT_CHECKPOINTING_KWARGS}" "${TRAIN_PER_DEVICE_TRAIN_BATCH_SIZE}" "${TRAIN_GRADIENT_ACCUMULATION_STEPS}" "${TRAIN_MODEL_PATH}" "${TRAIN_RESUME_FROM_CHECKPOINT}" "${TRAIN_LEARNING_RATE}" "${TRAIN_NUM_TRAIN_EPOCHS}" "${TRAIN_CUTOFF_LEN}" "${TRAIN_OVERWRITE_CACHE}" "${TRAIN_GAP_DIFFUSION_LOSS_WEIGHT}" "${TRAIN_GAP_GRPO_FORMAT_REWARD_WEIGHT}" <<'PY'
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
train_gap_training_mode = sys.argv[10]
train_gap_rollout_steps = sys.argv[11]
train_gap_rollout_strategy = sys.argv[12]
train_gap_rollout_conf_threshold = sys.argv[13]
train_rollout_scope = sys.argv[14]
train_gap_reveal_ratio = sys.argv[15]
train_gap_min_reveal_tokens = sys.argv[16]
train_gap_remask_threshold = sys.argv[17]
train_gap_remask_loss_weight = sys.argv[18]
train_gap_global_loss_weight = sys.argv[19]
train_gap_remask_default_p_mask = sys.argv[20]
train_gap_grpo_loss_weight = sys.argv[21]
train_gap_grpo_num_samples = sys.argv[22]
train_gap_grpo_clip_eps = sys.argv[23]
train_gap_grpo_entropy_coef = sys.argv[24]
train_gap_grpo_remask_penalty = sys.argv[25]
train_gap_grpo_advantage_eps = sys.argv[26]
train_gap_grpo_sample_prob_eps = sys.argv[27]
train_gap_grpo_dense_reward_weight = sys.argv[28]
train_gap_grpo_terminal_reward_weight = sys.argv[29]
train_disable_gradient_checkpointing = sys.argv[30]
train_gradient_checkpointing = sys.argv[31]
train_use_reentrant_gc = sys.argv[32]
train_gradient_checkpointing_kwargs = sys.argv[33]
train_per_device_train_batch_size = sys.argv[34]
train_gradient_accumulation_steps = sys.argv[35]
train_model_path = sys.argv[36]
train_resume_from_checkpoint = sys.argv[37]
train_learning_rate = sys.argv[38]
train_num_train_epochs = sys.argv[39]
train_cutoff_len = sys.argv[40]
train_overwrite_cache = sys.argv[41]
train_gap_diffusion_loss_weight = sys.argv[42]
train_gap_grpo_format_reward_weight = sys.argv[43]
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
updated = set_key(updated, "gap_training_mode", train_gap_training_mode)
updated = set_key(updated, "gap_rollout_steps", train_gap_rollout_steps)
updated = set_key(updated, "gap_rollout_strategy", train_gap_rollout_strategy)
updated = set_key(updated, "gap_rollout_confidence_threshold", train_gap_rollout_conf_threshold)
updated = set_key(updated, "gap_rollout_scope", train_rollout_scope)
updated = set_key(updated, "gap_reveal_ratio", train_gap_reveal_ratio)
updated = set_key(updated, "gap_min_reveal_tokens", train_gap_min_reveal_tokens)
updated = set_key(updated, "gap_remask_threshold", train_gap_remask_threshold)
updated = set_key(updated, "gap_remask_loss_weight", train_gap_remask_loss_weight)
updated = set_key(updated, "gap_global_loss_weight", train_gap_global_loss_weight)
updated = set_key(updated, "gap_diffusion_loss_weight", train_gap_diffusion_loss_weight)
updated = set_key(updated, "gap_remask_default_p_mask", train_gap_remask_default_p_mask)
updated = set_key(updated, "gap_grpo_loss_weight", train_gap_grpo_loss_weight)
updated = set_key(updated, "gap_grpo_num_samples", train_gap_grpo_num_samples)
updated = set_key(updated, "gap_grpo_clip_eps", train_gap_grpo_clip_eps)
updated = set_key(updated, "gap_grpo_entropy_coef", train_gap_grpo_entropy_coef)
updated = set_key(updated, "gap_grpo_remask_penalty", train_gap_grpo_remask_penalty)
updated = set_key(updated, "gap_grpo_advantage_eps", train_gap_grpo_advantage_eps)
updated = set_key(updated, "gap_grpo_sample_prob_eps", train_gap_grpo_sample_prob_eps)
updated = set_key(updated, "gap_grpo_dense_reward_weight", train_gap_grpo_dense_reward_weight)
updated = set_key(updated, "gap_grpo_terminal_reward_weight", train_gap_grpo_terminal_reward_weight)
updated = set_key(updated, "gap_grpo_format_reward_weight", train_gap_grpo_format_reward_weight)
updated = set_key(updated, "disable_gradient_checkpointing", train_disable_gradient_checkpointing)
updated = set_key(updated, "gradient_checkpointing", train_gradient_checkpointing)
updated = set_key(updated, "use_reentrant_gc", train_use_reentrant_gc)
updated = set_key(updated, "gradient_checkpointing_kwargs", train_gradient_checkpointing_kwargs)
updated = set_key(updated, "per_device_train_batch_size", train_per_device_train_batch_size)
updated = set_key(updated, "gradient_accumulation_steps", train_gradient_accumulation_steps)
updated = set_key(updated, "resume_from_checkpoint", train_resume_from_checkpoint if train_resume_from_checkpoint else "null")

if train_model_path:
    updated = set_key(updated, "model_name_or_path", train_model_path)
if train_learning_rate:
    updated = set_key(updated, "learning_rate", train_learning_rate)
if train_num_train_epochs:
    updated = set_key(updated, "num_train_epochs", train_num_train_epochs)
if train_cutoff_len:
    updated = set_key(updated, "cutoff_len", train_cutoff_len)
if train_overwrite_cache:
    updated = set_key(updated, "overwrite_cache", train_overwrite_cache)

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
echo "Training profile: ${TRAIN_PROFILE}" >&2
if [[ -n "${TRAIN_INTERLEAVE_PROBS}" ]]; then
    echo "Interleave probs: ${TRAIN_INTERLEAVE_PROBS}" >&2
fi
echo "GAP training mode: ${TRAIN_GAP_TRAINING_MODE}" >&2
echo "GAP rollout steps: ${TRAIN_GAP_ROLLOUT_STEPS}" >&2
echo "GAP rollout strategy: ${TRAIN_GAP_ROLLOUT_STRATEGY}" >&2
echo "GAP rollout confidence threshold: ${TRAIN_GAP_ROLLOUT_CONFIDENCE_THRESHOLD}" >&2
echo "GAP reveal ratio: ${TRAIN_GAP_REVEAL_RATIO}" >&2
echo "GAP min reveal tokens: ${TRAIN_GAP_MIN_REVEAL_TOKENS}" >&2
echo "GAP remask threshold: ${TRAIN_GAP_REMASK_THRESHOLD}" >&2
echo "GAP remask loss weight: ${TRAIN_GAP_REMASK_LOSS_WEIGHT}" >&2
echo "GAP global loss weight: ${TRAIN_GAP_GLOBAL_LOSS_WEIGHT}" >&2
echo "GAP diffusion loss weight: ${TRAIN_GAP_DIFFUSION_LOSS_WEIGHT}" >&2
echo "GAP remask default p_mask: ${TRAIN_GAP_REMASK_DEFAULT_P_MASK}" >&2
echo "GAP GRPO loss weight: ${TRAIN_GAP_GRPO_LOSS_WEIGHT}" >&2
echo "GAP GRPO num samples: ${TRAIN_GAP_GRPO_NUM_SAMPLES}" >&2
echo "GAP GRPO clip eps: ${TRAIN_GAP_GRPO_CLIP_EPS}" >&2
echo "GAP GRPO entropy coef: ${TRAIN_GAP_GRPO_ENTROPY_COEF}" >&2
echo "GAP GRPO remask penalty: ${TRAIN_GAP_GRPO_REMASK_PENALTY}" >&2
echo "GAP GRPO dense reward weight: ${TRAIN_GAP_GRPO_DENSE_REWARD_WEIGHT}" >&2
echo "GAP GRPO terminal reward weight: ${TRAIN_GAP_GRPO_TERMINAL_REWARD_WEIGHT}" >&2
echo "GAP GRPO format reward weight: ${TRAIN_GAP_GRPO_FORMAT_REWARD_WEIGHT}" >&2
echo "Disable gradient checkpointing: ${TRAIN_DISABLE_GRADIENT_CHECKPOINTING}" >&2
echo "Training args gradient checkpointing: ${TRAIN_GRADIENT_CHECKPOINTING}" >&2
echo "Use reentrant GC: ${TRAIN_USE_REENTRANT_GC}" >&2
echo "Gradient checkpointing kwargs: ${TRAIN_GRADIENT_CHECKPOINTING_KWARGS}" >&2
echo "Per-device train batch size: ${TRAIN_PER_DEVICE_TRAIN_BATCH_SIZE}" >&2
echo "Gradient accumulation steps: ${TRAIN_GRADIENT_ACCUMULATION_STEPS}" >&2
if [[ -n "${TRAIN_MODEL_PATH}" ]]; then
    echo "Model path override: ${TRAIN_MODEL_PATH}" >&2
fi
if [[ -n "${TRAIN_RESUME_FROM_CHECKPOINT}" ]]; then
    echo "Resume from checkpoint: ${TRAIN_RESUME_FROM_CHECKPOINT}" >&2
fi
if [[ -n "${TRAIN_LEARNING_RATE}" ]]; then
    echo "Learning rate override: ${TRAIN_LEARNING_RATE}" >&2
fi
if [[ -n "${TRAIN_NUM_TRAIN_EPOCHS}" ]]; then
    echo "Num train epochs override: ${TRAIN_NUM_TRAIN_EPOCHS}" >&2
fi
if [[ -n "${TRAIN_CUTOFF_LEN}" ]]; then
    echo "Cutoff length override: ${TRAIN_CUTOFF_LEN}" >&2
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
DEFAULT_PYTHONWARNINGS='ignore:The '"'"'repr'"'"' attribute with value False was provided to the `Field\(\)` function.*:UserWarning,ignore:The '"'"'frozen'"'"' attribute with value True was provided to the `Field\(\)` function.*:UserWarning'
export PYTHONWARNINGS="${PYTHONWARNINGS:-${DEFAULT_PYTHONWARNINGS}}"
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
