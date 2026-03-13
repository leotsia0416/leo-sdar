#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/generate_policy.py" \
  --mode infer \
  --config "$ROOT_DIR/configs/remask_infer.yaml"
