# VESTA Distribution Fitting Task

## Objective

Use VESTA to fit a Bayesian probabilistic model to a dataset. VESTA will iteratively propose candidate distributions, generate PyMC code, fit the models, and select the best one based on AIC (Akaike Information Criterion).

## Input

A parquet file at `/app/data/data.parquet` containing a single column of numeric observations.

## Expected Output

1. Run VESTA on the provided data
2. Generate a report at `/app/report.md` containing:
   - The best-fitting distribution family
   - The estimated parameters
   - The AIC score
   - A brief interpretation of the results

## Evaluation

Your solution will be evaluated on:
- Successful execution of VESTA without errors
- Generation of a valid report with all required fields
- Reasonableness of the fitted distribution and parameters

## Hints

- VESTA is already installed in the environment
- Use the `vesta` CLI or Python API
- The task may take several minutes as VESTA runs multiple iterations
- Check `/logs/agent/` for execution logs if debugging is needed

Good luck!
