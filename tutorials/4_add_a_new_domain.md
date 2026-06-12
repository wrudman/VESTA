# Tutorial 4: Add a new domain

A domain is a self-contained modeling problem (distribution fitting, time series,
or something you define). Adding one is a code edit: you create a domain package
with four components, register it, and create a Harbor task that runs it. You then
evaluate it through Harbor like [Tutorial 1](1_run_with_your_api_key.sh).

For the full machine-readable recipe (every base class, the `Registry` wiring, the
prompt/plotting contracts), see the "Adding a New Domain" section of
[`AGENTS.md`](../AGENTS.md).

## Files to create

Create a package under `src/vesta/domains/<your_domain>/` with four files:

| File | Responsibility | Base class |
|---|---|---|
| `toolkit.py` | Expert tools for the domain | `DomainToolkit` |
| `prompts.py` | VLM prompt templates and the response schema | `DomainPrompts` |
| `plotting.py` | Data visualization and fit-state extraction | `DomainPlotting` |
| `__init__.py` | Domain aliases + imports so Morphic sees the subclasses | (none) |

The cleanest way to start is to copy `src/vesta/domains/distribution_fitting/`
and adapt it.

## Register the domain

1. Add a member to the `Domain` enum in `src/vesta/core/experiment_enums.py`.
2. Import your domain package in `src/vesta/domains/__init__.py` so its
   `Registry` subclasses are discovered at startup.

## Create a Harbor task

Copy `harbor/tasks/vesta-distribution-fitting/` to a new task directory and edit:

- `task.toml` (the `task.name` must be `org/name` format)
- `data/` (your dataset)
- `solution/solve.sh` (set `domain='<your_domain>'`)
- `tests/test.sh` (what counts as success)

## Evaluate

```bash
harbor run --path harbor/tasks/<your-new-task>/ --agent oracle --env-file .env
```
