# AGENTS.md

VESTA: VLM-guided PyMC model selection for distribution fitting and time-series forecasting. This file orients AI coding agents on how to use, configure, and extend the VESTA codebase.

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/adivekar-utexas/VESTA
cd VESTA
cp .env.example .env
# Edit .env to add your LLM API key(s) (see .env.example for provider-specific vars)

# 2. Install (editable, with dev tools)
pip install -e ".[dev]"

# 3. Run the pre-built harbor task on the bundled dataset
pip install harbor
harbor run --path harbor/tasks/vesta-distribution-fitting/ --agent oracle --env-file .env
```

That is enough to run VESTA. Everything below is for customization.

## Project Layout

```
VESTA/
  src/vesta/              # Main package
    core/                 # Domain-agnostic pipeline: config, run loop, backend wiring
      experiments.py      # Main run() and run_all() functions
      experiment_config.py  # Pydantic ExperimentConfig (CLI + programmatic)
      experiment_enums.py # Domain, ToolkitMode, ReasoningEffort, etc.
      dynamic_toolkit.py  # LLM-generated tool execution
      processing_utils.py # PyMC model fitting, parameter cleaning
      experiment_step_state.py  # Per-step state types
      api_repair_diagnostics.py # Code repair prompts for failed generations
    domains/              # Domain-specific logic
      distribution_fitting/  # DF toolkit, prompts, plotting
      time_series/        # TS toolkit, prompts, plotting
    vlm_backends/         # VLM transport (SlowBurn/LiteLLM)
    data/                 # Dataset loaders (CSV, parquet, pickle)
    runtime/              # Early-init: thread caps, compile dir
    cli.py                # `vesta` CLI entrypoint
  harbor/tasks/           # Harbor task definitions
    vesta-distribution-fitting/  # The pre-built task
      task.toml           # Task config
      instruction.md      # Agent-facing instructions
      environment/Dockerfile  # Container build
      solution/solve.sh   # Reference solution (calls VESTA Python API)
      tests/test.sh       # Verifier
      data/data.parquet   # Bundled test dataset
  DAWN/                   # DAWN dataset generators (out of scope for agents)
  examples/               # Python API quickstarts
  .env.example            # Template for your API keys
```

## Running with Your Own API Keys

VESTA uses LiteLLM for multi-provider LLM access. You need valid API keys for at least one provider.

### Which providers does LiteLLM support?

LiteLLM is a unified API that normalizes requests across 100+ providers via a model name prefix. The full list is at https://docs.litellm.ai/docs/providers

Common model strings:
- `azure/gpt-5-mini` -- Azure OpenAI GPT-5 mini
- `bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0` -- AWS Bedrock Claude Sonnet 4
- `together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` -- Together AI Llama 4
- `openrouter/anthropic/claude-sonnet-4-20250514` -- OpenRouter Claude Sonnet 4
- `gpt-4o-mini` -- OpenAI direct
- `anthropic/claude-sonnet-4.6` -- Anthropic direct

### Step 1: Set your API key and model

Copy `.env.example` to `.env` and uncomment/fill the variables for your provider.

For example, to use Azure OpenAI:
```dotenv
AZURE_API_KEY=your-actual-azure-api-key
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-12-01-preview
```

For Together AI:
```dotenv
TOGETHERAI_API_KEY=tgp_v1_your_actual_together_key
```

For OpenRouter:
```dotenv
OPENROUTER_API_KEY=sk-or-v1-your_actual_key
```

Also set the model override variable in `.env`:
```dotenv
VESTA_MODEL_ID=azure/gpt-5-mini
```

The harbor task reads this env var; the Docker container inherits it when run with `--env-file .env`.

### Step 2: Run

```bash
harbor run --path harbor/tasks/vesta-distribution-fitting/ --agent oracle --env-file .env
```

Harbor builds the Docker image (you need Docker running), installs VESTA inside the container, installs your dataset, runs the VESTA pipeline, and writes results to `/app/report.md`. The verifier checks that the report was produced. You can check `/logs/agent/` inside the container if debugging is needed.

### Development (runs without Docker)

For iterating on VESTA itself, use the editable install:

```bash
pip install -e ".[dev]"

# Run directly via Python API:
python examples/quickstart_csv.py

# Or use the CLI (reads config from CLI flags):
vesta --data-pkl my_data.pkl --max-steps 3 --model.litellm-model azure/gpt-5-mini --toolkit.mode generate_only --output.expt my_experiment
```

The CLI flags use pydantic-settings (kebab-case). Every nested config field is a dot-separated flag: `--model.litellm-model`, `--toolkit.mode`, `--output.expt`.

## Bringing Your Own Data

VESTA supports three formats: CSV, Parquet, and Pickle.

### Step 1: Convert your file to VESTA's format

VESTA expects a `list[dict]` where each dict is one dataset. The dict keys differ by domain.

**Distribution Fitting:**
```python
from vesta.data import load_distribution_csv, load_distribution_parquet, save_datasets_pickle

# CSV (single column of numeric observations)
datasets = load_distribution_csv("my_data.csv", value_column="values_col")
# Parquet
datasets = load_distribution_parquet("my_data.parquet", value_column="values_col")
# Save to pickle (VESTA's internal format)
save_datasets_pickle(datasets, "my_data.pkl")
```

The dict has keys: `data` (np.ndarray of floats), `idx` (int or str), `dist_choice` (str or list of str), `true_params` (dict, can be empty `{}`).

If your file has a single column, omit `value_column` (it is auto-detected). If it has multiple columns, pass `value_column` to select which column to use.

**Time Series:**
```python
from vesta.data import load_timeseries_csv, load_timeseries_parquet, save_datasets_pickle

# CSV (optionally specify time_column and value_column)
datasets = load_timeseries_csv("my_timeseries.csv", value_column="value", time_column="date")
# Parquet
datasets = load_timeseries_parquet("my_timeseries.parquet", value_column="value")
# Save to pickle
save_datasets_pickle(datasets, "my_timeseries.pkl")
```

The dict has keys: `data` (pd.Series), `series_id` (int or str), `anomaly_info` (str, can be "none").

### Step 2: Run via harbor

Replace the bundled data with your own parquet (or pkl). You can either bake it into the Dockerfile or mount it at runtime. The pre-built harbor task expects a parquet file at `/app/data/data.parquet`. To use your own:

1. Copy your file to `harbor/tasks/vesta-distribution-fitting/data/data.parquet`
2. Run `harbor run --path harbor/tasks/vesta-distribution-fitting/ --agent oracle --env-file .env`

To use a different harbor task directory (e.g. for time-series), create a new task directory under `harbor/tasks/` following the same structure as `vesta-distribution-fitting/`. The `solution/solve.sh` script is what actually runs the VESTA pipeline; the `task.toml` controls time limits and resource requirements.

### Quick test without harbor

To test locally (for development only), you can run the pipeline directly:

```python
from vesta.data import load_distribution_parquet, save_datasets_pickle
from vesta import ExperimentConfig, run_all
from vesta.core.experiment_config import ModelConfig, ToolkitConfig, OutputConfig

datasets = load_distribution_parquet("my_data.parquet", value_column="value")
save_datasets_pickle(datasets, "my_data.pkl")

config = ExperimentConfig(
    domain="distribution_fitting",
    data_pkl="my_data.pkl",
    max_steps=3,
    model=ModelConfig(litellm_model="azure/gpt-5-mini"),
    toolkit=ToolkitConfig(mode="generate_only"),
    output=OutputConfig(expt="my_experiment"),
)
results = run_all(config=config)
```

Note: `ModelConfig`, `ToolkitConfig`, and `OutputConfig` are required nested objects. You cannot pass flat kwargs like `model_id=` or `toolkit_mode=` directly to `ExperimentConfig` -- those were removed in favor of Pydantic-validated nested config.

## Using Different Models and Overriding LLM Params

### Switching models via env var

The harbor task's `solution/solve.sh` reads `VESTA_MODEL_ID` from the environment. Set it in your `.env` file:

```dotenv
VESTA_MODEL_ID=openrouter/anthropic/claude-sonnet-4-20250514
```

Then run `harbor run --path harbor/tasks/vesta-distribution-fitting/ --agent oracle --env-file .env`.

### Overriding LiteLLM params

Some models need provider-specific params (e.g. reasoning effort, extra_body). VESTA computes these automatically from `reasoning_effort` (a ModelConfig field), but you can override any computed value by passing `litellm_params` as a JSON dict.

**Via env var (for harbor):**
```dotenv
VESTA_LITELLM_PARAMS='{"reasoning_effort": "high"}'
```

The solve.sh reads this env var, parses it as JSON, and passes it to `ModelConfig`. Keys in `litellm_params` override any computed values.

**Via Python API:**
```python
model = ModelConfig(
    litellm_model="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
    litellm_params={"output_config": {"effort": "high"}},
)
config = ExperimentConfig(
    domain="distribution_fitting",
    data_pkl="my_data.pkl",
    max_steps=3,
    model=model,
    toolkit=ToolkitConfig(mode="generate_only"),
    output=OutputConfig(expt="my_experiment"),
)
```

### Common model configs

**Azure OpenAI GPT-5 mini (default):**
```python
ModelConfig(litellm_model="azure/gpt-5-mini")
# VESTA auto-sets reasoning_effort="low" for iterative tasks
```

**Claude Sonnet 4 via Bedrock:**
```python
ModelConfig(
    litellm_model="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
    litellm_params={"output_config": {"effort": "low"}},
)
```

**Llama 4 via Together AI:**
```python
ModelConfig(litellm_model="together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
```

**Claude Sonnet 4 via OpenRouter:**
```python
ModelConfig(litellm_model="openrouter/anthropic/claude-sonnet-4-20250514")
```

The model string must be a valid LiteLLM model identifier. See https://docs.litellm.ai/docs/providers for the full list.

## Adding Expert Tools to an Existing Domain

Each domain has a Tool Registry base class. To add a new tool, define a class that subclasses the domain's registry.

### Distribution Fitting

Edit `src/vesta/domains/distribution_fitting/toolkit.py`.

Every tool must:
1. Subclass `DistributionFittingTool` (already imported)
2. Set `tool_description: ClassVar[str]` (a 1-2 sentence description for the VLM)
3. Set `output_type: ClassVar[str]` (one of `"visualization"` or `"numeric"`)
4. Set `parameters_schema: ClassVar[Dict[str, Any]]` (JSON Schema dict; empty dict `{}` for no-arg tools)
5. Implement `execute()` returning a `DiagnosticToolResult`

Example:
```python
class MyCustomTool(DistributionFittingTool):
    tool_description: ClassVar[str] = "Computes the Jarque-Bera test for normality."
    output_type: ClassVar[str] = "numeric"
    parameters_schema: ClassVar[Dict[str, Any]] = {}

    def execute(self, *, data, fit_state, best_idx, fit_path, selected_tool_args=None):
        from scipy.stats import jarque_bera
        stat, p_value = jarque_bera(data)
        return DiagnosticToolResult(
            tool_name=self.tool_name,
            tool_description=self.tool_description,
            artifacts=[
                DiagnosticArtifact(
                    artifact_type="text",
                    description=f"JB stat={stat:.4f}, p={p_value:.4f}",
                )
            ],
        )
```

Everything else is automatic: the tool registers itself into the `DistributionFittingTool` Registry under its snake_case class name, its OpenAI function-calling schema is auto-generated from `tool_description` and `parameters_schema`, and `get_expert_tools()` includes it automatically.

### Time Series

Same pattern, but subclass `TimeSeriesExpertTool` in `src/vesta/domains/time_series/toolkit.py`.

## Adding a New Domain

A domain consists of four components. Each is a Morphic `Typed + Registry` subclass.

### Step 1: Create the directory

```bash
mkdir -p src/vesta/domains/my_new_domain
touch src/vesta/domains/my_new_domain/__init__.py
touch src/vesta/domains/my_new_domain/toolkit.py
touch src/vesta/domains/my_new_domain/prompts.py
touch src/vesta/domains/my_new_domain/plotting.py
```

### Step 2: Define the domain aliases

In `src/vesta/domains/my_new_domain/__init__.py`:
```python
DOMAIN_ALIASES = ["my-domain", "my_new_domain"]

from vesta.domains.my_new_domain.plotting import MyNewDomainPlotting  # noqa: F401
from vesta.domains.my_new_domain.prompts import MyNewDomainPrompts  # noqa: F401
from vesta.domains.my_new_domain.toolkit import MyNewDomainToolkit  # noqa: F401
```

The aliases are used by `DomainPrompts.of("my-domain")` etc.

### Step 3: Implement the toolkit

In `src/vesta/domains/my_new_domain/toolkit.py`:
```python
from abc import ABC
from typing import ClassVar, Any, Dict, List, Optional, Union

from morphic import Registry

from vesta.domains import Tool, FitState, DomainToolkit, DiagnosticToolResult, DiagnosticArtifact

class MyNewDomainTool(Tool, Registry, ABC):
    pass

class MyNewDomainToolkit(DomainToolkit):
    aliases: ClassVar[List[str]] = ["my-domain", "my_new_domain"]

    def get_expert_tools(self) -> List[Dict[str, Any]]:
        return [cls.to_openai_schema() for cls in MyNewDomainTool.subclasses()]

    def execute_tool(self, *, selected_tool, selected_tool_args, data, fit_state, best_idx, fit_path, plot_type_descriptions):
        tool = MyNewDomainTool.of(selected_tool)
        return tool.execute(data=data, fit_state=fit_state, best_idx=best_idx, fit_path=fit_path, selected_tool_args=selected_tool_args)

    def supports_dynamic_generation(self) -> bool:
        return True  # or False if you only want static tools
```

### Step 4: Implement the prompts

In `src/vesta/domains/my_new_domain/prompts.py`, subclass `DomainPrompts` (from `vesta.domains`). The most important methods:
- `get_response_type()` -- return a `Typed` class for parsing VLM JSON proposals
- `render_proposal_prompt()` -- the initial prompt asking VLM to propose models
- `get_feedback_prompt_template()` -- the iterative feedback prompt

Study `src/vesta/domains/distribution_fitting/prompts.py` for a complete example.

### Step 5: Implement the plotting

In `src/vesta/domains/my_new_domain/plotting.py`, subclass `DomainPlotting` (from `vesta.domains`). You also need a `FitState` subclass for storing the fitted model state.

### Step 6: Register the domain

In `src/vesta/core/experiment_enums.py`, add a new member to `Domain`:
```python
class Domain(AutoEnum):
    distribution_fitting = auto()
    time_series = auto()
    my_new_domain = auto()
```

Then import your domain's `__init__.py` in `src/vesta/domains/__init__.py` so Morphic sees the subclasses at startup.

## Toolkit Modes

VESTA supports four toolkit modes (set via `ToolkitConfig(mode=...)` or `--toolkit.mode`):

| Mode | What it does |
|---|---|
| `none` | No tools at all. The VLM still sees a histogram/line-plot and proposes models. |
| `expert` | Fixed expert-written tools (e.g. QQ plot, moments calculator, ACF plot). |
| `generate_only` | Only the dynamic tool generator (no expert tools). VESTA creates new tools each run. |
| `dynamic` | Expert tools PLUS the dynamic generator. The VLM can pick an existing tool or create a new one. |
| `accumulated_only` | Uses previously accumulated tools from a registry file (for tool-persistence experiments). |

The default in the harbor task is `generate_only`. For production use, `dynamic` is recommended.

## The Harbor Task Structure

Each task lives at `harbor/tasks/<task-name>/` and has:

- `task.toml` -- metadata, timeouts, resource limits. The `task.name` field must be in `org/name` format.
- `instruction.md` -- natural-language instructions shown to the agent at runtime.
- `environment/Dockerfile` -- container build instructions. The build context is the VESTA repo root.
- `data/` -- bundled test data. The default distribution-fitting task expects `data/data.parquet`.
- `solution/solve.sh` -- the reference solution. The default runs the VESTA Python API on the bundled data.
- `tests/test.sh` -- the verifier. Default checks that `/app/report.md` was created.

To create a new harbor task, copy the existing one and modify:
- `task.toml` -- change `task.name` and `task.description`
- `data/` -- replace with your own data
- `solution/solve.sh` -- change the model, domain, or data path
- `tests/test.sh` -- change what counts as success

## Verifying Your Setup

After making changes, verify the package imports and the harbor task can build:

```bash
python -c "import vesta; from vesta import ExperimentConfig, run_all; print('OK')"
```

For the harbor task, check that the Dockerfile can build (requires Docker):
```bash
cd harbor/tasks/vesta-distribution-fitting/environment
docker build -t vesta-test -f Dockerfile ../../
```

The `../..` is the VESTA repo root, which is the build context (the Dockerfile uses `COPY src`, `COPY pyproject.toml`, `COPY harbor/...` from the repo root).
