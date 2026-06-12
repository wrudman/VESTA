"""Dataset loaders for bringing your own data (CSV/parquet/pickle)."""
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def load_distribution_csv(path: Union[str, Path], *, value_column: Optional[str] = None, label: Optional[str] = None) -> List[Dict[str, Any]]:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    if value_column is None:
        if df.shape[1] == 1:
            value_column = df.columns[0]
        else:
            raise ValueError(f"Multiple columns found. Specify value_column. Available: {list(df.columns)}")
    data = df[value_column].dropna().values.astype(float)
    return [{
        "data": data,
        "idx": 0,
        "dist_choice": label if label is not None else "unknown",
        "true_params": {},
    }]


def load_distribution_parquet(path: Union[str, Path], *, value_column: Optional[str] = None, label: Optional[str] = None) -> List[Dict[str, Any]]:
    pq_path = Path(path)
    df = pd.read_parquet(pq_path)
    if value_column is None:
        if df.shape[1] == 1:
            value_column = df.columns[0]
        else:
            raise ValueError(f"Multiple columns found. Specify value_column. Available: {list(df.columns)}")
    data = df[value_column].dropna().values.astype(float)
    return [{
        "data": data,
        "idx": 0,
        "dist_choice": label if label is not None else "unknown",
        "true_params": {},
    }]


def load_distribution_pickle(path: Union[str, Path]) -> List[Dict[str, Any]]:
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def load_timeseries_csv(path: Union[str, Path], *, value_column: Optional[str] = None, time_column: Optional[str] = None, series_id: Union[int, str, None] = None) -> List[Dict[str, Any]]:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    if value_column is None:
        if df.shape[1] == 1:
            value_column = df.columns[0]
        else:
            raise ValueError(f"Multiple columns found. Specify value_column. Available: {list(df.columns)}")
    if time_column is not None and time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column])
        series = pd.Series(df[value_column].values, index=df[time_column])
    else:
        series = pd.Series(df[value_column].values)
    return [{
        "data": series,
        "series_id": series_id if series_id is not None else 0,
        "anomaly_info": "unknown",
    }]


def load_timeseries_parquet(path: Union[str, Path], *, value_column: Optional[str] = None, time_column: Optional[str] = None, series_id: Union[int, str, None] = None) -> List[Dict[str, Any]]:
    pq_path = Path(path)
    df = pd.read_parquet(pq_path)
    if value_column is None:
        if df.shape[1] == 1:
            value_column = df.columns[0]
        else:
            raise ValueError(f"Multiple columns found. Specify value_column. Available: {list(df.columns)}")
    if time_column is not None and time_column in df.columns:
        series = pd.Series(df[value_column].values, index=df[time_column])
    else:
        series = pd.Series(df[value_column].values)
    return [{
        "data": series,
        "series_id": series_id if series_id is not None else 0,
        "anomaly_info": "unknown",
    }]


def load_timeseries_pickle(path: Union[str, Path]) -> List[Dict[str, Any]]:
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def save_datasets_pickle(datasets: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    with open(Path(path), "wb") as f:
        pickle.dump(datasets, f)
