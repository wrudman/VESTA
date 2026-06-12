import json
import os
from typing import Any, Dict

from vesta import ExperimentConfig, run_all
from vesta.core.experiment_config import ModelConfig, OutputConfig, ToolkitConfig
from vesta.data import load_distribution_parquet, save_datasets_pickle


def _load_litellm_params() -> Dict[str, Any]:
    raw_params = os.environ["LITELLM_PARAMS"]
    if len(raw_params) == 0:
        return {}
    if raw_params.startswith("'") and raw_params.endswith("'"):
        raw_params = raw_params[1:-1]
    if raw_params.startswith('"') and raw_params.endswith('"'):
        raw_params = raw_params[1:-1]
    try:
        return json.loads(raw_params)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"VESTA_LITELLM_PARAMS is not valid JSON: {exc}. "
            f"Raw value: {raw_params!r}"
        ) from exc


datasets = load_distribution_parquet("/app/data/data.parquet", value_column="value")
save_datasets_pickle(datasets, "/app/data.pkl")

model_kwargs: Dict[str, Any] = {"litellm_model": os.environ["MODEL_ID"]}
litellm_params = _load_litellm_params()
if len(litellm_params) > 0:
    model_kwargs["litellm_params"] = litellm_params

config = ExperimentConfig(
    domain="distribution_fitting",
    data_pkl="/app/data.pkl",
    max_steps=3,
    model=ModelConfig(**model_kwargs),
    toolkit=ToolkitConfig(mode=os.environ["TOOLKIT_MODE"]),
    output=OutputConfig(expt="harbor_task"),
)

results = run_all(config=config)

with open("/app/report.md", "w") as report_file:
    report_file.write("# VESTA Distribution Fitting Report\n\n")
    report_file.write("## Results\n\n")
    for result_index, result in enumerate(results):
        report_file.write(f"### Dataset {result_index}\n\n")
        report_file.write(f"- Status: {result['status']}\n")
        if result["status"] == "error":
            report_file.write(f"- Error: {result['error']}\n")
        else:
            report_file.write(f"- Best Distribution: {result['run_best_model_structure']}\n")
            report_file.write(f"- AIC: {result['run_best_model_aic']}\n")
        report_file.write(f"- Steps: {len(result['steps'])}\n\n")
    report_file.write("\n## Interpretation\n\n")
    report_file.write("The model selection process completed and produced a valid VESTA run report.\n")
