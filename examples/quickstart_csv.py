#!/usr/bin/env python
"""Quickstart example: run VESTA on a CSV file."""
import pickle
from pathlib import Path

from vesta.data import load_distribution_csv, save_datasets_pickle
from vesta import ExperimentConfig, run_all


def main():
    # 1. Load your CSV
    csv_path = "my_data.csv"
    datasets = load_distribution_csv(csv_path)

    # 2. Save as pickle (VESTA expects pkl format)
    pkl_path = "my_data.pkl"
    save_datasets_pickle(datasets, pkl_path)

    # 3. Create config
    config = ExperimentConfig(
        domain="distribution_fitting",
        data_pkl=pkl_path,
        max_steps=3,
        model_id="azure/gpt-5.4-mini",
        toolkit_mode="generate_only",
        parallel_nproc=0,  # single process
        parallel_nthread=0,
        parallel_max_rpm=120,
        output_expt="my_experiment",
    )

    # 4. Run
    run_all(config)


if __name__ == "__main__":
    main()
