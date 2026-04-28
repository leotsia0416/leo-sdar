#!/usr/bin/env bash

set -euo pipefail

EXP_BASE="${1:?missing exp base}"
EXAMPLE_INDEX="${2:?missing example index}"
OUTPUT_DIR="${3:?missing output dir}"
GSM8K_TEST="${4:-/work/leotsia0416/datasets/gsm8k/test.jsonl}"

EXP_DIR="$(find "${EXP_BASE}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
if [[ -z "${EXP_DIR}" ]]; then
  echo "No experiment directory found under ${EXP_BASE}" >&2
  exit 1
fi

rm -rf "${OUTPUT_DIR}"
/work/leotsia0416/sdar_eval/bin/python \
  /work/leotsia0416/projects/SDAR/script/analysis/extract_full_eval_example_artifacts.py \
  --exp-dir "${EXP_DIR}" \
  --gsm8k-test "${GSM8K_TEST}" \
  --example-index "${EXAMPLE_INDEX}" \
  --output-dir "${OUTPUT_DIR}"

printf '%s\n' "${EXP_DIR}" > "${OUTPUT_DIR}/source_exp_dir.txt"
