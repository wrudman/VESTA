"""Test the BYO data path end-to-end: pkl -> parquet -> loader -> VESTA-ready dataset."""
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from vesta.data import (
    load_distribution_csv,
    load_distribution_parquet,
    load_distribution_pickle,
    save_datasets_pickle,
)


def test_pkl_to_parquet_roundtrip():
    """Test that we can convert an existing pkl to parquet and load it back."""
    # Source: real dataset from DAWN
    src_pkl = "DAWN/datast_distribution_fitting/data_single.pkl"
    with open(src_pkl, "rb") as f:
        datasets = pickle.load(f)
    print(f"Loaded {len(datasets)} datasets from pkl")

    # Take first dataset, convert to parquet (flat format: value column)
    first = datasets[0]
    data_array = np.asarray(first["data"], dtype=float)
    print(f"  data: shape={data_array.shape}, first 5: {data_array[:5]}")
    print(f"  idx: {first['idx']}, dist_choice: {first['dist_choice']}")

    df = pd.DataFrame({"value": data_array})
    parquet_path = "DAWN/datast_distribution_fitting/test_single.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved parquet to {parquet_path}")

    # Load it back via the new loader
    loaded = load_distribution_parquet(parquet_path, value_column="value")
    assert len(loaded) == 1
    loaded_data = np.asarray(loaded[0]["data"], dtype=float)
    assert np.allclose(loaded_data, data_array), "Parquet round-trip corrupted data"
    print(f"  round-tripped data shape: {loaded_data.shape}, first 5: {loaded_data[:5]}")
    assert loaded[0]["idx"] == 0
    assert loaded[0]["dist_choice"] == "unknown"
    assert loaded[0]["true_params"] == {}
    print("Parquet round-trip: OK")

    # Now test that load_distribution_pickle also works (sanity)
    loaded_pkl = load_distribution_pickle(src_pkl)
    assert len(loaded_pkl) == len(datasets)
    print(f"pkl loader: OK ({len(loaded_pkl)} datasets)")

    # Test save_datasets_pickle
    out_pkl = "DAWN/datast_distribution_fitting/test_resaved.pkl"
    save_datasets_pickle(loaded, out_pkl)
    re_loaded = pickle.load(open(out_pkl, "rb"))
    assert len(re_loaded) == 1
    print("save_datasets_pickle: OK")

    return parquet_path, loaded


if __name__ == "__main__":
    test_pkl_to_parquet_roundtrip()
    print("\nAll data path tests passed.")
