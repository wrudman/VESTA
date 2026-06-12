# VESTA Tutorials

Runnable companions to the tutorials in the top-level [`README.md`](../README.md).
Every tutorial runs VESTA the same way the docs describe: through `harbor run`.

| Tutorial | Script | README section |
|---|---|---|
| 1. Run with your own API key | [`1_run_with_your_api_key.sh`](1_run_with_your_api_key.sh) | "Run with your own API key" |
| 2. Bring your own data (CSV / Parquet) | [`2_bring_your_own_data.sh`](2_bring_your_own_data.sh) | "Run VESTA on your own data" |
| 3. Add expert tools to a domain | [`3_add_expert_tools.md`](3_add_expert_tools.md) | "Add expert tools to a domain" |
| 4. Add a new domain | [`4_add_a_new_domain.md`](4_add_a_new_domain.md) | "Add a new domain" |

Tutorials 1 and 2 are runnable scripts: they end in `harbor run`. Tutorials 3
and 4 are code-editing walkthroughs (you edit the repo, then run tutorial 1 or 2
to evaluate your changes through Harbor). For machine-readable, step-by-step
detail aimed at AI agents, see [`AGENTS.md`](../AGENTS.md).

## Prerequisites

```bash
pip install harbor          # Harbor runs the containerized VESTA task
cp .env.example .env        # then add your provider API key to .env
```

The running example model throughout these tutorials is
`anthropic/claude-sonnet-4.6` with `reasoning_effort=low`. Swap in any
[LiteLLM-compatible model name](https://docs.litellm.ai/docs/providers) by
editing the `VESTA_MODEL_ID` line in your `.env`.

## Running a tutorial

```bash
cd <repo-root>
bash tutorials/1_run_with_your_api_key.sh
```

Each script echoes the exact `harbor run` command it executes, so you can copy
it, tweak flags, and re-run by hand.
