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
export SDAR_GAP_REMASK_THRESHOLD_SCHEDULE="${SDAR_GAP_REMASK_THRESHOLD_SCHEDULE:-0.00:0.30,0.30:0.50,0.70:0.65}"
export SDAR_GAP_REMASK_LOSS_WEIGHT_SCHEDULE="${SDAR_GAP_REMASK_LOSS_WEIGHT_SCHEDULE:-0.00:0.08,0.33:0.02,0.67:0.0}"
export SDAR_GAP_GRPO_LOSS_WEIGHT_SCHEDULE="${SDAR_GAP_GRPO_LOSS_WEIGHT_SCHEDULE:-0.00:0.25,0.33:0.35,0.67:0.4}"
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

yaml_get_optional() {
    local value
    value="$(yaml_get "$1")"
    case "${value,,}" in
        ""|null|none)
            echo ""
            ;;
        *)
            echo "${value}"
            ;;
    esac
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

prepare_training_run_bundle() {
    local output_dir="${1:?missing output dir}"
    local launcher_workdir="${2:?missing launcher workdir}"
    local run_root=""
    local run_stamp=""
    local run_label=""
    local slurm_log_prefix=""

    run_root="${TRAIN_RUN_ROOT:-$(yaml_get_optional run_root)}"
    run_root="${run_root:-${REPO_ROOT}/runs/train}"
    mkdir -p "${run_root}"
    export TRAIN_RUN_ROOT="${run_root}"

    if [[ -n "${TRAIN_RUN_NAME_BASE:-}" ]]; then
        run_label="${TRAIN_RUN_NAME_BASE}"
    else
        run_label="train_${TRAIN_PROFILE}_${TRAIN_DATASET_TAG}"
    fi

    run_stamp="$(date +%Y%m%d_%H%M%S)"
    run_label="$(sanitize_name "${run_label}")"
    export TRAIN_RUN_BUNDLE_DIR="${TRAIN_RUN_BUNDLE_DIR:-${TRAIN_RUN_ROOT}/${run_stamp}_${run_label}_job${job_id}}"
    mkdir -p "${TRAIN_RUN_BUNDLE_DIR}"

    slurm_log_prefix="${REPO_ROOT}/logs/train_${job_id}"
    link_bundle_path "${slurm_log_prefix}.out" "${TRAIN_RUN_BUNDLE_DIR}/slurm.out"
    link_bundle_path "${slurm_log_prefix}.err" "${TRAIN_RUN_BUNDLE_DIR}/slurm.err"
    link_bundle_path "${output_dir}" "${TRAIN_RUN_BUNDLE_DIR}/output_dir"
    link_bundle_path "${output_dir}" "${TRAIN_RUN_BUNDLE_DIR}/checkpoints"

    cp "${TMP_CONFIG_FILE}" "${TRAIN_RUN_BUNDLE_DIR}/training_run_config.yaml"
    cp "${CONFIG_FILE}" "${TRAIN_RUN_BUNDLE_DIR}/training_defaults.yaml"

    cat > "${TRAIN_RUN_BUNDLE_DIR}/run_meta.yaml" <<EOF
runner: train
generated_from: script/training.sh
slurm_job_id: "${job_id}"
slurm_job_name: "${SLURM_JOB_NAME:-}"
train_profile: "${TRAIN_PROFILE}"
model_path: "${TRAIN_MODEL_PATH}"
resume_from_checkpoint: "${TRAIN_RESUME_FROM_CHECKPOINT}"
dataset: "${TRAIN_DATASET}"
tokenized_path: "${TRAIN_TOKENIZED_PATH}"
output_dir: "${output_dir}"
launcher_workdir: "${launcher_workdir}"
run_root: "${TRAIN_RUN_ROOT}"
EOF
}

TRAIN_HEAD_ONLY="$(normalize_bool "${TRAIN_HEAD_ONLY:-false}")"
HEAD_ONLY_TRAINABLE_MODULES="${HEAD_ONLY_TRAINABLE_MODULES:-gap_remask_head}"
TRAIN_PROFILE="${TRAIN_PROFILE:-hard_math}"
HARD_MATH_RAW_DATASET_NAME="${HARD_MATH_RAW_DATASET_NAME:-math_train_hard_local}"
HARD_MATH_DATASET_NAME="${HARD_MATH_DATASET_NAME:-math_train_hard_boxed_prompt_local}"
BASE_MODEL_PATH="${REPO_ROOT}/Models/SDAR-1.7B-Chat-"
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
TRAIN_CHECKPOINT_PATH="${TRAIN_CHECKPOINT_PATH:-}"
TRAIN_MODEL_PATH="${TRAIN_MODEL_PATH:-$(yaml_get model_name_or_path)}"
TRAIN_MODEL_PATH="${TRAIN_MODEL_PATH:-${BASE_MODEL_PATH}}"
TRAIN_RESUME_FROM_CHECKPOINT="${TRAIN_RESUME_FROM_CHECKPOINT:-$(yaml_get_optional resume_from_checkpoint)}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-}"
TRAIN_NUM_TRAIN_EPOCHS="${TRAIN_NUM_TRAIN_EPOCHS:-}"
TRAIN_CUTOFF_LEN="${TRAIN_CUTOFF_LEN:-}"
TRAIN_OVERWRITE_CACHE="${TRAIN_OVERWRITE_CACHE:-}"
TRAIN_DATASET="${TRAIN_DATASET:-$(yaml_get dataset)}"
TRAIN_DATASET="${TRAIN_DATASET:-${HARD_MATH_DATASET_NAME}}"
DEFAULT_TRAIN_DATASET="${TRAIN_DATASET}"
TRAIN_RUN_NAME_BASE="${TRAIN_RUN_NAME_BASE:-$(yaml_get_optional run_name)}"
TRAIN_MIX_STRATEGY="${TRAIN_MIX_STRATEGY:-$(yaml_get mix_strategy)}"
TRAIN_MIX_STRATEGY="${TRAIN_MIX_STRATEGY:-concat}"
DEFAULT_TRAIN_MIX_STRATEGY="${TRAIN_MIX_STRATEGY}"
TRAIN_INTERLEAVE_PROBS="${TRAIN_INTERLEAVE_PROBS:-$(yaml_get_optional interleave_probs)}"
TRAIN_TOKENIZED_PATH="${TRAIN_TOKENIZED_PATH:-$(yaml_get_optional tokenized_path)}"
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
TRAIN_GAP_GRPO_NUM_PARENTS="${TRAIN_GAP_GRPO_NUM_PARENTS:-$(yaml_get_optional gap_grpo_num_parents)}"
TRAIN_GAP_GRPO_MIN_VISIBLE_BLOCKS="${TRAIN_GAP_GRPO_MIN_VISIBLE_BLOCKS:-$(yaml_get gap_grpo_min_visible_blocks)}"
TRAIN_GAP_GRPO_CANDIDATE_WINDOW_BLOCKS="${TRAIN_GAP_GRPO_CANDIDATE_WINDOW_BLOCKS:-$(yaml_get gap_grpo_candidate_window_blocks)}"
TRAIN_GAP_GRPO_DIVERSE_ROLLBACKS="${TRAIN_GAP_GRPO_DIVERSE_ROLLBACKS:-$(yaml_get gap_grpo_diverse_rollbacks)}"
TRAIN_GAP_GRPO_CLIP_EPS="${TRAIN_GAP_GRPO_CLIP_EPS:-$(yaml_get gap_grpo_clip_eps)}"
TRAIN_GAP_GRPO_ENTROPY_COEF="${TRAIN_GAP_GRPO_ENTROPY_COEF:-$(yaml_get gap_grpo_entropy_coef)}"
TRAIN_GAP_GRPO_REMASK_PENALTY="${TRAIN_GAP_GRPO_REMASK_PENALTY:-$(yaml_get gap_grpo_remask_penalty)}"
TRAIN_GAP_GRPO_ADVANTAGE_EPS="${TRAIN_GAP_GRPO_ADVANTAGE_EPS:-$(yaml_get gap_grpo_advantage_eps)}"
TRAIN_GAP_GRPO_SAMPLE_PROB_EPS="${TRAIN_GAP_GRPO_SAMPLE_PROB_EPS:-$(yaml_get gap_grpo_sample_prob_eps)}"
TRAIN_GAP_GRPO_DENSE_REWARD_WEIGHT="${TRAIN_GAP_GRPO_DENSE_REWARD_WEIGHT:-$(yaml_get gap_grpo_dense_reward_weight)}"
TRAIN_GAP_GRPO_TERMINAL_REWARD_WEIGHT="${TRAIN_GAP_GRPO_TERMINAL_REWARD_WEIGHT:-$(yaml_get gap_grpo_terminal_reward_weight)}"
TRAIN_GAP_GRPO_FORMAT_REWARD_WEIGHT="${TRAIN_GAP_GRPO_FORMAT_REWARD_WEIGHT:-$(yaml_get gap_grpo_format_reward_weight)}"
TRAIN_GAP_GRPO_MIXED_TERMINAL_FILTER="${TRAIN_GAP_GRPO_MIXED_TERMINAL_FILTER:-$(yaml_get gap_grpo_mixed_terminal_filter)}"
TRAIN_GAP_GRPO_MIXED_TERMINAL_FILTER="${TRAIN_GAP_GRPO_MIXED_TERMINAL_FILTER:-true}"
TRAIN_GAP_GRPO_CORRECT_THRESHOLD="${TRAIN_GAP_GRPO_CORRECT_THRESHOLD:-$(yaml_get gap_grpo_correct_threshold)}"
TRAIN_GAP_GRPO_CORRECT_THRESHOLD="${TRAIN_GAP_GRPO_CORRECT_THRESHOLD:-0.5}"
TRAIN_GAP_GRPO_MIN_CORRECT_COUNT="${TRAIN_GAP_GRPO_MIN_CORRECT_COUNT:-$(yaml_get gap_grpo_min_correct_count)}"
TRAIN_GAP_GRPO_MIN_CORRECT_COUNT="${TRAIN_GAP_GRPO_MIN_CORRECT_COUNT:-1}"
TRAIN_GAP_GRPO_MAX_CORRECT_COUNT="${TRAIN_GAP_GRPO_MAX_CORRECT_COUNT:-$(yaml_get gap_grpo_max_correct_count)}"
TRAIN_GAP_GRPO_MAX_CORRECT_COUNT="${TRAIN_GAP_GRPO_MAX_CORRECT_COUNT:--1}"
TRAIN_GAP_GRPO_ROLLOUT_TEMPERATURE="${TRAIN_GAP_GRPO_ROLLOUT_TEMPERATURE:-$(yaml_get gap_grpo_rollout_temperature)}"
TRAIN_GAP_GRPO_INITIAL_ROLLOUT_TEMPERATURE="${TRAIN_GAP_GRPO_INITIAL_ROLLOUT_TEMPERATURE:-$(yaml_get gap_grpo_initial_rollout_temperature)}"
TRAIN_GAP_GRPO_ROLLOUT_TOP_K="${TRAIN_GAP_GRPO_ROLLOUT_TOP_K:-$(yaml_get gap_grpo_rollout_top_k)}"
TRAIN_GAP_GRPO_ROLLOUT_TOP_P="${TRAIN_GAP_GRPO_ROLLOUT_TOP_P:-$(yaml_get gap_grpo_rollout_top_p)}"
TRAIN_DISABLE_GRADIENT_CHECKPOINTING="${TRAIN_DISABLE_GRADIENT_CHECKPOINTING:-$(yaml_get disable_gradient_checkpointing)}"
TRAIN_GRADIENT_CHECKPOINTING="${TRAIN_GRADIENT_CHECKPOINTING:-$(yaml_get gradient_checkpointing)}"
TRAIN_USE_REENTRANT_GC="${TRAIN_USE_REENTRANT_GC:-$(yaml_get use_reentrant_gc)}"
TRAIN_GRADIENT_CHECKPOINTING_KWARGS="${TRAIN_GRADIENT_CHECKPOINTING_KWARGS:-$(yaml_get gradient_checkpointing_kwargs)}"
TRAIN_PER_DEVICE_TRAIN_BATCH_SIZE="${TRAIN_PER_DEVICE_TRAIN_BATCH_SIZE:-$(yaml_get per_device_train_batch_size)}"
TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-$(yaml_get gradient_accumulation_steps)}"
TRAIN_ROLLOUT_SCOPE="${TRAIN_ROLLOUT_SCOPE:-frontier_block}"

if [[ -n "${TRAIN_CHECKPOINT_PATH}" ]]; then
    TRAIN_MODEL_PATH="${TRAIN_CHECKPOINT_PATH}"
fi

if [[ "${TRAIN_PROFILE}" == "format_align" ]]; then
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
    if [[ -z "${TRAIN_MODEL_PATH}" || -n "${TRAIN_CHECKPOINT_PATH}" ]]; then
        TRAIN_MODEL_PATH="${BASE_MODEL_PATH}"
    fi
    TRAIN_RESUME_FROM_CHECKPOINT=""
    TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-5e-6}"
    TRAIN_CUTOFF_LEN="${TRAIN_CUTOFF_LEN:-1024}"
    if [[ "${TRAIN_PROFILE}" == "orca_math_puma_remask_from_base" ]]; then
        TRAIN_RUN_NAME_BASE="${TRAIN_RUN_NAME_BASE:-sdar_1p7b_orca_math_gap_puma_remask_from_base}"
    else
        TRAIN_RUN_NAME_BASE="${TRAIN_RUN_NAME_BASE:-sdar_1p7b_orca_math_gap_remask_from_base}"
    fi
    TRAIN_GAP_TRAINING_MODE="remask"
    TRAIN_ROLLOUT_SCOPE="all"
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
    neat_packing_enabled="$(normalize_bool "${TRAIN_NEAT_PACKING}")"

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
    local prepare_script="${SCRIPT_DIR}/helpers/prepare_hard_math_boxed_prompt.py"
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
    local prepare_script="${SCRIPT_DIR}/helpers/prepare_gsm8k_boxed_format_align.py"
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
python - "${TMP_CONFIG_FILE}" "${SAVE_STEPS}" "${TRAIN_HEAD_ONLY}" "${HEAD_ONLY_TRAINABLE_MODULES}" "${TRAIN_DATASET}" "${TRAIN_TOKENIZED_PATH}" "${TRAIN_RUN_NAME_BASE}" "${TRAIN_MIX_STRATEGY}" "${TRAIN_INTERLEAVE_PROBS}" "${TRAIN_GAP_TRAINING_MODE}" "${TRAIN_GAP_ROLLOUT_STEPS}" "${TRAIN_GAP_ROLLOUT_STRATEGY}" "${TRAIN_GAP_ROLLOUT_CONFIDENCE_THRESHOLD}" "${TRAIN_ROLLOUT_SCOPE}" "${TRAIN_GAP_REVEAL_RATIO}" "${TRAIN_GAP_MIN_REVEAL_TOKENS}" "${TRAIN_GAP_REMASK_THRESHOLD}" "${TRAIN_GAP_REMASK_LOSS_WEIGHT}" "${TRAIN_GAP_GLOBAL_LOSS_WEIGHT}" "${TRAIN_GAP_REMASK_DEFAULT_P_MASK}" "${TRAIN_GAP_GRPO_LOSS_WEIGHT}" "${TRAIN_GAP_GRPO_NUM_SAMPLES}" "${TRAIN_GAP_GRPO_NUM_PARENTS}" "${TRAIN_GAP_GRPO_MIN_VISIBLE_BLOCKS}" "${TRAIN_GAP_GRPO_CANDIDATE_WINDOW_BLOCKS}" "${TRAIN_GAP_GRPO_DIVERSE_ROLLBACKS}" "${TRAIN_GAP_GRPO_CLIP_EPS}" "${TRAIN_GAP_GRPO_ENTROPY_COEF}" "${TRAIN_GAP_GRPO_REMASK_PENALTY}" "${TRAIN_GAP_GRPO_ADVANTAGE_EPS}" "${TRAIN_GAP_GRPO_SAMPLE_PROB_EPS}" "${TRAIN_GAP_GRPO_DENSE_REWARD_WEIGHT}" "${TRAIN_GAP_GRPO_TERMINAL_REWARD_WEIGHT}" "${TRAIN_GAP_GRPO_FORMAT_REWARD_WEIGHT}" "${TRAIN_GAP_GRPO_ROLLOUT_TEMPERATURE}" "${TRAIN_GAP_GRPO_INITIAL_ROLLOUT_TEMPERATURE}" "${TRAIN_GAP_GRPO_ROLLOUT_TOP_K}" "${TRAIN_GAP_GRPO_ROLLOUT_TOP_P}" "${TRAIN_DISABLE_GRADIENT_CHECKPOINTING}" "${TRAIN_GRADIENT_CHECKPOINTING}" "${TRAIN_USE_REENTRANT_GC}" "${TRAIN_GRADIENT_CHECKPOINTING_KWARGS}" "${TRAIN_PER_DEVICE_TRAIN_BATCH_SIZE}" "${TRAIN_GRADIENT_ACCUMULATION_STEPS}" "${TRAIN_MODEL_PATH}" "${TRAIN_RESUME_FROM_CHECKPOINT}" "${TRAIN_LEARNING_RATE}" "${TRAIN_NUM_TRAIN_EPOCHS}" "${TRAIN_CUTOFF_LEN}" "${TRAIN_OVERWRITE_CACHE}" "${TRAIN_GAP_DIFFUSION_LOSS_WEIGHT}" "${TRAIN_GAP_GRPO_MIXED_TERMINAL_FILTER}" "${TRAIN_GAP_GRPO_CORRECT_THRESHOLD}" "${TRAIN_GAP_GRPO_MIN_CORRECT_COUNT}" "${TRAIN_GAP_GRPO_MAX_CORRECT_COUNT}" <<'PY'
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
train_gap_grpo_num_parents = sys.argv[23]
train_gap_grpo_min_visible_blocks = sys.argv[24]
train_gap_grpo_candidate_window_blocks = sys.argv[25]
train_gap_grpo_diverse_rollbacks = sys.argv[26]
train_gap_grpo_clip_eps = sys.argv[27]
train_gap_grpo_entropy_coef = sys.argv[28]
train_gap_grpo_remask_penalty = sys.argv[29]
train_gap_grpo_advantage_eps = sys.argv[30]
train_gap_grpo_sample_prob_eps = sys.argv[31]
train_gap_grpo_dense_reward_weight = sys.argv[32]
train_gap_grpo_terminal_reward_weight = sys.argv[33]
train_gap_grpo_format_reward_weight = sys.argv[34]
train_gap_grpo_rollout_temperature = sys.argv[35]
train_gap_grpo_initial_rollout_temperature = sys.argv[36]
train_gap_grpo_rollout_top_k = sys.argv[37]
train_gap_grpo_rollout_top_p = sys.argv[38]
train_disable_gradient_checkpointing = sys.argv[39]
train_gradient_checkpointing = sys.argv[40]
train_use_reentrant_gc = sys.argv[41]
train_gradient_checkpointing_kwargs = sys.argv[42]
train_per_device_train_batch_size = sys.argv[43]
train_gradient_accumulation_steps = sys.argv[44]
train_model_path = sys.argv[45]
train_resume_from_checkpoint = sys.argv[46]
train_learning_rate = sys.argv[47]
train_num_train_epochs = sys.argv[48]
train_cutoff_len = sys.argv[49]
train_overwrite_cache = sys.argv[50]
train_gap_diffusion_loss_weight = sys.argv[51]
train_gap_grpo_mixed_terminal_filter = sys.argv[52]
train_gap_grpo_correct_threshold = sys.argv[53]
train_gap_grpo_min_correct_count = sys.argv[54]
train_gap_grpo_max_correct_count = sys.argv[55]
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
updated = set_key(
    updated,
    "gap_grpo_num_parents",
    train_gap_grpo_num_parents if train_gap_grpo_num_parents else "null",
)
updated = set_key(
    updated,
    "gap_grpo_min_visible_blocks",
    train_gap_grpo_min_visible_blocks if train_gap_grpo_min_visible_blocks else "null",
)
updated = set_key(
    updated,
    "gap_grpo_candidate_window_blocks",
    train_gap_grpo_candidate_window_blocks if train_gap_grpo_candidate_window_blocks else "null",
)
updated = set_key(
    updated,
    "gap_grpo_diverse_rollbacks",
    train_gap_grpo_diverse_rollbacks if train_gap_grpo_diverse_rollbacks else "null",
)
updated = set_key(updated, "gap_grpo_clip_eps", train_gap_grpo_clip_eps)
updated = set_key(updated, "gap_grpo_entropy_coef", train_gap_grpo_entropy_coef)
updated = set_key(updated, "gap_grpo_remask_penalty", train_gap_grpo_remask_penalty)
updated = set_key(updated, "gap_grpo_advantage_eps", train_gap_grpo_advantage_eps)
updated = set_key(updated, "gap_grpo_sample_prob_eps", train_gap_grpo_sample_prob_eps)
updated = set_key(updated, "gap_grpo_dense_reward_weight", train_gap_grpo_dense_reward_weight)
updated = set_key(updated, "gap_grpo_terminal_reward_weight", train_gap_grpo_terminal_reward_weight)
updated = set_key(updated, "gap_grpo_format_reward_weight", train_gap_grpo_format_reward_weight)
updated = set_key(updated, "gap_grpo_mixed_terminal_filter", train_gap_grpo_mixed_terminal_filter)
updated = set_key(updated, "gap_grpo_correct_threshold", train_gap_grpo_correct_threshold)
updated = set_key(updated, "gap_grpo_min_correct_count", train_gap_grpo_min_correct_count)
updated = set_key(updated, "gap_grpo_max_correct_count", train_gap_grpo_max_correct_count)
updated = set_key(
    updated,
    "gap_grpo_rollout_temperature",
    train_gap_grpo_rollout_temperature if train_gap_grpo_rollout_temperature else "null",
)
updated = set_key(
    updated,
    "gap_grpo_initial_rollout_temperature",
    train_gap_grpo_initial_rollout_temperature if train_gap_grpo_initial_rollout_temperature else "null",
)
updated = set_key(
    updated,
    "gap_grpo_rollout_top_k",
    train_gap_grpo_rollout_top_k if train_gap_grpo_rollout_top_k else "null",
)
updated = set_key(
    updated,
    "gap_grpo_rollout_top_p",
    train_gap_grpo_rollout_top_p if train_gap_grpo_rollout_top_p else "null",
)
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

TRAIN_OUTPUT_DIR="$(sed -n 's/^output_dir:[[:space:]]*//p' "${TMP_CONFIG_FILE}" | head -n 1 | sed "s/^['\"]//; s/['\"]$//")"
if [[ -z "${TRAIN_OUTPUT_DIR}" ]]; then
    echo "Cannot determine output_dir from ${TMP_CONFIG_FILE}" >&2
    exit 1
fi
case "${TRAIN_OUTPUT_DIR}" in
    /*) ;;
    *) TRAIN_OUTPUT_DIR="$(yaml_get launcher_workdir)/${TRAIN_OUTPUT_DIR#./}" ;;
esac

prepare_training_run_bundle "${TRAIN_OUTPUT_DIR}" "$(yaml_get launcher_workdir)"

echo "Training dataset: ${TRAIN_DATASET}" >&2
echo "Tokenized cache: ${TRAIN_TOKENIZED_PATH}" >&2
echo "Run bundle dir: ${TRAIN_RUN_BUNDLE_DIR}" >&2
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
if [[ -n "${SDAR_GAP_REMASK_THRESHOLD_SCHEDULE:-}" ]]; then
    echo "GAP remask threshold schedule: ${SDAR_GAP_REMASK_THRESHOLD_SCHEDULE}" >&2
fi
echo "GAP remask loss weight: ${TRAIN_GAP_REMASK_LOSS_WEIGHT}" >&2
echo "GAP global loss weight: ${TRAIN_GAP_GLOBAL_LOSS_WEIGHT}" >&2
echo "GAP diffusion loss weight: ${TRAIN_GAP_DIFFUSION_LOSS_WEIGHT}" >&2
echo "GAP remask default p_mask: ${TRAIN_GAP_REMASK_DEFAULT_P_MASK}" >&2
echo "GAP GRPO loss weight: ${TRAIN_GAP_GRPO_LOSS_WEIGHT}" >&2
echo "GAP GRPO num samples: ${TRAIN_GAP_GRPO_NUM_SAMPLES}" >&2
echo "GAP GRPO num parents: ${TRAIN_GAP_GRPO_NUM_PARENTS}" >&2
echo "GAP GRPO min visible blocks: ${TRAIN_GAP_GRPO_MIN_VISIBLE_BLOCKS}" >&2
echo "GAP GRPO candidate window blocks: ${TRAIN_GAP_GRPO_CANDIDATE_WINDOW_BLOCKS}" >&2
echo "GAP GRPO diverse rollbacks: ${TRAIN_GAP_GRPO_DIVERSE_ROLLBACKS}" >&2
echo "GAP GRPO clip eps: ${TRAIN_GAP_GRPO_CLIP_EPS}" >&2
echo "GAP GRPO entropy coef: ${TRAIN_GAP_GRPO_ENTROPY_COEF}" >&2
echo "GAP GRPO remask penalty: ${TRAIN_GAP_GRPO_REMASK_PENALTY}" >&2
echo "GAP GRPO dense reward weight: ${TRAIN_GAP_GRPO_DENSE_REWARD_WEIGHT}" >&2
echo "GAP GRPO terminal reward weight: ${TRAIN_GAP_GRPO_TERMINAL_REWARD_WEIGHT}" >&2
echo "GAP GRPO format reward weight: ${TRAIN_GAP_GRPO_FORMAT_REWARD_WEIGHT}" >&2
echo "GAP GRPO mixed terminal filter: ${TRAIN_GAP_GRPO_MIXED_TERMINAL_FILTER}" >&2
echo "GAP GRPO correct threshold: ${TRAIN_GAP_GRPO_CORRECT_THRESHOLD}" >&2
echo "GAP GRPO min correct count: ${TRAIN_GAP_GRPO_MIN_CORRECT_COUNT}" >&2
echo "GAP GRPO max correct count: ${TRAIN_GAP_GRPO_MAX_CORRECT_COUNT}" >&2
echo "GAP GRPO rollout temperature: ${TRAIN_GAP_GRPO_ROLLOUT_TEMPERATURE}" >&2
echo "GAP GRPO initial rollout temperature: ${TRAIN_GAP_GRPO_INITIAL_ROLLOUT_TEMPERATURE}" >&2
echo "GAP GRPO rollout top-k: ${TRAIN_GAP_GRPO_ROLLOUT_TOP_K}" >&2
echo "GAP GRPO rollout top-p: ${TRAIN_GAP_GRPO_ROLLOUT_TOP_P}" >&2
echo "Disable gradient checkpointing: ${TRAIN_DISABLE_GRADIENT_CHECKPOINTING}" >&2
echo "Training args gradient checkpointing: ${TRAIN_GRADIENT_CHECKPOINTING}" >&2
echo "Use reentrant GC: ${TRAIN_USE_REENTRANT_GC}" >&2
echo "Gradient checkpointing kwargs: ${TRAIN_GRADIENT_CHECKPOINTING_KWARGS}" >&2
echo "Per-device train batch size: ${TRAIN_PER_DEVICE_TRAIN_BATCH_SIZE}" >&2
echo "Gradient accumulation steps: ${TRAIN_GRADIENT_ACCUMULATION_STEPS}" >&2
if [[ -n "${TRAIN_MODEL_PATH}" ]]; then
    echo "Model path: ${TRAIN_MODEL_PATH}" >&2
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
