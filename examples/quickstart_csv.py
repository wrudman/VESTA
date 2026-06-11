#!/usr/bin/env python
"""Quickstart example: run VESTA on a CSV file."""
from pathlib import Path

from vesta.data import load_distribution_csv, save_datasets_pickle
from vesta import ExperimentConfig, run_all
from vesta.core.experiment_config import ModelConfig, ToolkitConfig, OutputConfig


def main():
    # 1. Load your CSV (single column of numeric observations)
    csv_path = "my_data.csv"
    datasets = load_distribution_csv(csv_path, value_column="value")

    # 2. Save as pickle (VESTA's internal format)
    pkl_path = "my_data.pkl"
    save_datasets_pickle(datasets, pkl_path)

    # 3. Create config with nested model/toolkit/output objects
    config = ExperimentConfig(
        domain="distribution_fitting",
        data_pkl=pkl_path,
        max_steps=3,
        model=ModelConfig(litellm_model="azure/gpt-5-mini"),
        toolkit=ToolkitConfig(mode="generate_only"),
        output=OutputConfig(expt="my_experiment"),
    )

    # 4. Run
    results = run_all(config=config)

    # 5. Print results
    for i, result in enumerate(results):
        print(f"Dataset {i}: {result.get('status', 'unknown')}")
        if "run_best_model_structure" in result:
            print(f"  Best distribution: {result['run_best_model_structure']}")


if __name__ == "__main__":
    main()
