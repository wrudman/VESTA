#!/bin/bash
set -euo pipefail

MODEL_ID="${VESTA_MODEL_ID:-anthropic/claude-sonnet-4.6}"
LITELLM_PARAMS="${VESTA_LITELLM_PARAMS:-}"
TOOLKIT_MODE="${VESTA_TOOLKIT_MODE:-generate_only}"

MODEL_ID="$MODEL_ID" LITELLM_PARAMS="$LITELLM_PARAMS" TOOLKIT_MODE="$TOOLKIT_MODE" python /solution/solve.py
