"""Dataset schema documentation for VESTA.

This module documents the required keys for datasets used by VESTA. Datasets are
represented as ``list[dict]`` where each dict represents one dataset with the
following domain-specific keys:

Distribution Fitting Domain
---------------------------
Each dataset dict must contain:

- ``data``: ``np.ndarray`` — the observed values (1D array of samples)
- ``idx``: ``int | str`` — unique identifier for the dataset
- ``dist_choice``: ``str | list[str]`` — ground truth family name(s) (e.g. ``"gaussian"`` or ``["gaussian", "lognormal"]``)
- ``true_params``: ``dict`` — ground truth parameters (optional, can be empty dict ``{}``)

Example:
    {
        "data": np.array([1.2, 2.3, 1.8, 2.1, 1.9]),
        "idx": 0,
        "dist_choice": "gaussian",
        "true_params": {"mu": 2.0, "sigma": 0.5}
    }

Time Series Domain
------------------
Each dataset dict must contain:

- ``data``: ``pd.Series`` — the observed time series (indexed by time)
- ``series_id``: ``int | str`` — unique identifier for the series
- ``anomaly_info``: ``str`` — description of anomalies (can be ``"none"`` or ``"unknown"``)

Example:
    {
        "data": pd.Series([1.2, 2.3, 1.8, 2.1, 1.9], index=pd.date_range("2020-01-01", periods=5)),
        "series_id": 0,
        "anomaly_info": "none"
    }
"""
