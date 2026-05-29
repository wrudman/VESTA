#!/bin/bash
set -euo pipefail

# Box Loop Time Series baseline command matrix.
# Run from this directory:
#   cd baselines/box_loop/
#   bash runs_time_series_box_loop.sh
#
# Each command uses --rounds 5, meaning 1 initial proposal call + 4 improvement calls.
# --parallel.max-rpm is accepted as the Box Loop equivalent of experiments.py's
# --parallel.max-rpm and is divided across --nproc workers by run.py.
# --output is a run name only; run.py writes timestamped .pkl files under outputs/.
# Match experiments.py run-script naming: omit chunk suffix for the first 0:50 chunk;
# add suffixes only for later chunks such as 50:100 and 100:110.

# =============================================================================
# Easy: dataset_ts_easy_50
# =============================================================================

# Easy - Claude Sonnet 4.6 - 0:50
python run.py --task box_loop_ts --data dataset_ts_easy_50.pkl --dataset-idx "0:50" --rounds 5 --model "openrouter/anthropic/claude-sonnet-4.6" --nproc 10 --parallel.max-rpm 30 --output box_loop_ts_easy_claude_sonnet46

# Easy - Kimi K2.5 - 0:50
python run.py --task box_loop_ts --data dataset_ts_easy_50.pkl --dataset-idx "0:50" --rounds 5 --model "openrouter/moonshotai/kimi-k2.5" --nproc 10 --parallel.max-rpm 30 --output box_loop_ts_easy_kimi25

# Easy - GPT-5.4 Mini - 0:50
python run.py --task box_loop_ts --data dataset_ts_easy_50.pkl --dataset-idx "0:50" --rounds 5 --model "azure/gpt-5.4-mini" --nproc 10 --parallel.max-rpm 10 --output box_loop_ts_easy_gpt54_mini

# =============================================================================
# Medium: dataset_ts_medium_110
# =============================================================================

# Medium - Claude Sonnet 4.6 - 0:50
python run.py --task box_loop_ts --data dataset_ts_medium_110.pkl --dataset-idx "0:50" --rounds 5 --model "openrouter/anthropic/claude-sonnet-4.6" --nproc 10 --parallel.max-rpm 30 --output box_loop_ts_medium_claude_sonnet46

# Medium - Claude Sonnet 4.6 - 50:100
python run.py --task box_loop_ts --data dataset_ts_medium_110.pkl --dataset-idx "50:100" --rounds 5 --model "openrouter/anthropic/claude-sonnet-4.6" --nproc 10 --parallel.max-rpm 30 --output box_loop_ts_medium_claude_sonnet46_50to100

# Medium - Claude Sonnet 4.6 - 100:110
python run.py --task box_loop_ts --data dataset_ts_medium_110.pkl --dataset-idx "100:110" --rounds 5 --model "openrouter/anthropic/claude-sonnet-4.6" --nproc 2 --parallel.max-rpm 30 --output box_loop_ts_medium_claude_sonnet46_100to110

# Medium - Kimi K2.5 - 0:50
python run.py --task box_loop_ts --data dataset_ts_medium_110.pkl --dataset-idx "0:50" --rounds 5 --model "openrouter/moonshotai/kimi-k2.5" --nproc 10 --parallel.max-rpm 30 --output box_loop_ts_medium_kimi25

# Medium - Kimi K2.5 - 50:100
python run.py --task box_loop_ts --data dataset_ts_medium_110.pkl --dataset-idx "50:100" --rounds 5 --model "openrouter/moonshotai/kimi-k2.5" --nproc 10 --parallel.max-rpm 30 --output box_loop_ts_medium_kimi25_50to100

# Medium - Kimi K2.5 - 100:110
python run.py --task box_loop_ts --data dataset_ts_medium_110.pkl --dataset-idx "100:110" --rounds 5 --model "openrouter/moonshotai/kimi-k2.5" --nproc 2 --parallel.max-rpm 30 --output box_loop_ts_medium_kimi25_100to110

# Medium - GPT-5.4 Mini - 0:50
python run.py --task box_loop_ts --data dataset_ts_medium_110.pkl --dataset-idx "0:50" --rounds 5 --model "azure/gpt-5.4-mini" --nproc 10 --parallel.max-rpm 10 --output box_loop_ts_medium_gpt54_mini

# Medium - GPT-5.4 Mini - 50:100
python run.py --task box_loop_ts --data dataset_ts_medium_110.pkl --dataset-idx "50:100" --rounds 5 --model "azure/gpt-5.4-mini" --nproc 10 --parallel.max-rpm 10 --output box_loop_ts_medium_gpt54_mini_50to100

# Medium - GPT-5.4 Mini - 100:110
python run.py --task box_loop_ts --data dataset_ts_medium_110.pkl --dataset-idx "100:110" --rounds 5 --model "azure/gpt-5.4-mini" --nproc 2 --parallel.max-rpm 10 --output box_loop_ts_medium_gpt54_mini_100to110

# =============================================================================
# Hard: dataset_ts_gravitational_chirp_50
# =============================================================================

# Hard - Claude Sonnet 4.6 - 0:50
python run.py --task box_loop_ts --data dataset_ts_gravitational_chirp_50.pkl --dataset-idx "0:50" --rounds 5 --model "openrouter/anthropic/claude-sonnet-4.6" --nproc 10 --parallel.max-rpm 30 --output box_loop_ts_gravitational_chirp_claude_sonnet46

# Hard - Kimi K2.5 - 0:50
python run.py --task box_loop_ts --data dataset_ts_gravitational_chirp_50.pkl --dataset-idx "0:50" --rounds 5 --model "openrouter/moonshotai/kimi-k2.5" --nproc 10 --parallel.max-rpm 30 --output box_loop_ts_gravitational_chirp_kimi25

# Hard - GPT-5.4 Mini - 0:50
python run.py --task box_loop_ts --data dataset_ts_gravitational_chirp_50.pkl --dataset-idx "0:50" --rounds 5 --model "azure/gpt-5.4-mini" --nproc 10 --parallel.max-rpm 10 --output box_loop_ts_gravitational_chirp_gpt54_mini
