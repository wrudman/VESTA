# Tutorial 3: Add expert tools to a domain

Expert tools are diagnostics the VLM can call during refinement (QQ plots, moment
calculators, autocorrelation plots, statistical tests). Adding one is a pure
code edit: you subclass the domain's tool registry, and the tool auto-registers.
After editing, you evaluate your change by running it through Harbor exactly like
[Tutorial 1](1_run_with_your_api_key.sh).

For the full machine-readable recipe (every ClassVar, the `execute()` contract,
the return types), see the "Adding Expert Tools to an Existing Domain" section of
[`AGENTS.md`](../AGENTS.md).

## Which file to edit

| Domain | Toolkit file | Base class to subclass |
|---|---|---|
| Distribution fitting | `src/vesta/domains/distribution_fitting/toolkit.py` | `DistributionFittingTool` |
| Time series | `src/vesta/domains/time_series/toolkit.py` | `TimeSeriesExpertTool` |

## Minimal example (distribution fitting)

Add this class to `src/vesta/domains/distribution_fitting/toolkit.py`:

```python
class JarqueBeraTool(DistributionFittingTool):
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

The tool registers itself under its snake_case class name, its function-calling
schema is generated from `tool_description` + `parameters_schema`, and
`get_expert_tools()` picks it up automatically. No other wiring is needed.

## Evaluate your tool

Run the task with a toolkit mode that exposes expert tools (`expert` or
`dynamic`). Set this in `.env`:

```bash
VESTA_TOOLKIT_MODE=dynamic
```

Then run as in Tutorial 1:

```bash
harbor run --path harbor/tasks/vesta-distribution-fitting/ --agent oracle --env-file .env
```
