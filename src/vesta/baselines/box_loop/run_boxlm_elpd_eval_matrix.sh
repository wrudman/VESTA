#!/usr/bin/env bash
# BoxLM Time Series ELPD evaluation matrix.
#
# This file intentionally lists one independent command per model/dataset pair.
# You can copy/paste or dispatch these lines in parallel; each command writes to a
# distinct output directory under evals/.

set -euo pipefail
cd "$(dirname "$0")"


# Easy: dataset_ts_easy_50
clear && python run_boxlm_elpd_eval.py --model claude --dataset easy_50
clear && python run_boxlm_elpd_eval.py --model gpt --dataset easy_50
clear && python run_boxlm_elpd_eval.py --model kimi --dataset easy_50

# Hard: dataset_ts_hard_110. The wrapper resolves and concatenates all three
# shards for each model: 0:50, 50:100, and 100:110.
clear && python run_boxlm_elpd_eval.py --model claude --dataset hard_110
clear && python run_boxlm_elpd_eval.py --model gpt --dataset hard_110
clear && python run_boxlm_elpd_eval.py --model kimi --dataset hard_110

# Astro: dataset_ts_astro_chirp_50
clear && python run_boxlm_elpd_eval.py --model claude --dataset astro_chirp_50
clear && python run_boxlm_elpd_eval.py --model gpt --dataset astro_chirp_50
clear && python run_boxlm_elpd_eval.py --model kimi --dataset astro_chirp_50
