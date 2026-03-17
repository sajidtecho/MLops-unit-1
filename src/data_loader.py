"""data_loader.py – Generate or load a dataset for the MLOps example."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_dataset(
    n_samples: int = 500,
    n_features: int = 10,
    n_informative: int = 5,
    random_state: int = 42,
    output_path: str = "data/raw/dataset.csv",
) -> pd.DataFrame:
    """Generate a synthetic binary classification dataset and save it to CSV.

    Parameters
    ----------
    n_samples:
        Number of samples to generate.
    n_features:
        Total number of features.
    n_informative:
        Number of informative features.
    random_state:
        Random seed for reproducibility.
    output_path:
        Path where the CSV will be written.

    Returns
    -------
    pd.DataFrame
        The generated dataset including a ``target`` column.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        random_state=random_state,
    )

    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}  ({len(df)} rows)")
    return df


def load_dataset(path: str = "data/raw/dataset.csv") -> pd.DataFrame:
    """Load a dataset from a CSV file.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run data_loader.generate_dataset() first."
        )
    return pd.read_csv(path)
