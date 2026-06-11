#!/bin/bash
set -e

python -c "
from vesta.data import load_distribution_parquet, save_datasets_pickle
from vesta import ExperimentConfig, run_all

# Load the parquet data
datasets = load_distribution_parquet('/app/data/data.parquet', value_column='value')
save_datasets_pickle(datasets, '/app/data.pkl')

# Create config
config = ExperimentConfig(
    domain='distribution_fitting',
    data_pkl='/app/data.pkl',
    max_steps=3,
    model_id='azure/gpt-5.4-mini',
    toolkit_mode='generate_only',
    parallel_nproc=0,
    parallel_nthread=0,
    parallel_max_rpm=120,
    output_expt='harbor_task',
)

# Run VESTA
results = run_all(config)

# Generate report
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
