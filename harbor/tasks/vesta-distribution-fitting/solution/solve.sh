#!/bin/bash
set -e

# Model and LiteLLM overrides are read from the environment so the task can be
# run against any provider/model without editing this script:
#   VESTA_MODEL_ID        LiteLLM model string (default: azure/gpt-5.4-mini)
#   VESTA_LITELLM_PARAMS  JSON dict forwarded verbatim to the backend; keys
#                         here override the computed reasoning_effort/api_base
#                         params (e.g. '{"reasoning_effort": "high"}').
MODEL_ID="${VESTA_MODEL_ID:-azure/gpt-5.4-mini}"
LITELLM_PARAMS="${VESTA_LITELLM_PARAMS:-}"

MODEL_ID="$MODEL_ID" LITELLM_PARAMS="$LITELLM_PARAMS" python -c "
import json
import os

from vesta.data import load_distribution_parquet, save_datasets_pickle
from vesta import ExperimentConfig, run_all
from vesta.core.experiment_config import ModelConfig, ToolkitConfig, OutputConfig

datasets = load_distribution_parquet('/app/data/data.parquet', value_column='value')
save_datasets_pickle(datasets, '/app/data.pkl')

model_kwargs = {'litellm_model': os.environ['MODEL_ID']}
litellm_params_raw = os.environ.get('LITELLM_PARAMS', '')
if len(litellm_params_raw) > 0:
    model_kwargs['litellm_params'] = json.loads(litellm_params_raw)

config = ExperimentConfig(
    domain='distribution_fitting',
    data_pkl='/app/data.pkl',
    max_steps=3,
    model=ModelConfig(**model_kwargs),
    toolkit=ToolkitConfig(mode='generate_only'),
    output=OutputConfig(expt='harbor_task'),
)

results = run_all(config=config)

with open('/app/report.md', 'w') as f:
    f.write('# VESTA Distribution Fitting Report\n\n')
    f.write('## Results\n\n')
    if results:
        for i, result in enumerate(results):
            f.write(f'### Dataset {i}\n\n')
            f.write(f'- Status: {result.get(\"status\", \"unknown\")}\n')
            if 'run_best_model_structure' in result:
                f.write(f'- Best Distribution: {result[\"run_best_model_structure\"]}\n')
            if 'run_best_model_aic' in result:
                f.write(f'- AIC: {result[\"run_best_model_aic\"]}\n')
            f.write(f'- Steps: {len(result.get(\"steps\", []))}\n\n')
    f.write('\n## Interpretation\n\n')
    f.write('The model selection process has identified the best-fitting distribution based on AIC criteria.\n')
"
