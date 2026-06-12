#!/usr/bin/env bash
#
# Tutorial 1: Run VESTA with your own API key.
#
# Mirrors the "Run with your own API key" section of the top-level README.
# It runs the bundled DAWN distribution-fitting task through Harbor using the
# model and credentials you put in .env.
#
# Usage (from the repo root):
#   bash tutorials/1_run_with_your_api_key.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TASK_PATH="harbor/tasks/vesta-distribution-fitting/"

# 1. Make sure you have a .env with your provider's API key.
#    VESTA reads LiteLLM-compatible model names (provider/model-name).
#    See https://docs.litellm.ai/docs/providers for the full list.
if [ ! -f .env ]; then
    echo "No .env found. Creating one from .env.example."
    echo "Edit .env to add your API key (e.g. ANTHROPIC_API_KEY), then re-run."
    cp .env.example .env
    exit 1
fi

# 2. Pick the model. The running example is anthropic/claude-sonnet-4.6 with
#    reasoning_effort=low. Override these in .env to use any other provider:
#       VESTA_MODEL_ID=openai/gpt-4o-mini
#       VESTA_LITELLM_PARAMS='{"reasoning_effort": "low"}'
echo "Using model/params from .env (VESTA_MODEL_ID, VESTA_LITELLM_PARAMS)."

# 3. Run via Harbor. --agent oracle executes the reference VESTA pipeline.
CMD=(harbor run --path "$TASK_PATH" --agent oracle --env-file .env)
echo "+ ${CMD[*]}"
"${CMD[@]}"
