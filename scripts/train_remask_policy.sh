#!/bin/bash
#SBATCH -A MST114566
#SBATCH -J remask_policy_train
#SBATCH -p normal,normal2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=20:00:00
#SBATCH -o /work/leotsia0416/projects/SDAR/logs/remask_train_%j.out
#SBATCH -e /work/leotsia0416/projects/SDAR/logs/remask_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leotsia.ee13@nycu.edu.tw

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/work/leotsia0416/sdar_eval/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/remask_train.yaml}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT_DIR/checkpoint}"
JOB_ID="${SLURM_JOB_ID:-local}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-$CHECKPOINT_ROOT/remask_train_${JOB_ID}}"
RUNTIME_CONFIG_PATH="${RUNTIME_CONFIG_PATH:-/tmp/remask_train_${JOB_ID}.yaml}"

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR" "$CHECKPOINT_ROOT" "$RUN_OUTPUT_DIR"
trap 'rm -f "$RUNTIME_CONFIG_PATH"' EXIT

sed "s|^output_dir:.*|output_dir: ${RUN_OUTPUT_DIR}|" "$CONFIG_PATH" > "$RUNTIME_CONFIG_PATH"

"$PYTHON_BIN" "$ROOT_DIR/generate_policy.py" \
  --mode train \
  --config "$RUNTIME_CONFIG_PATH"
