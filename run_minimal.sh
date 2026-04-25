#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${SD_MODEL_PATH:-}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT_DIR/sexual_eraser_example/minimal_prompts.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/sexual_eraser_example/demo_outputs}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "Please set SD_MODEL_PATH to a local Stable Diffusion checkpoint directory."
  echo "Example: export SD_MODEL_PATH=/path/to/local-sd-checkpoint"
  exit 1
fi

python "$ROOT_DIR/sexual_eraser_example/effective_erae_nudity.py" \
  --model-path "$MODEL_PATH" \
  --prompt-txt "$PROMPT_FILE" \
  --output-root "$OUTPUT_ROOT" \
  "$@"
