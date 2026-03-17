"""predict.py – Make predictions with the saved model.

Usage
-----
    python src/predict.py
"""
from __future__ import annotations

import os

import joblib
import numpy as np

MODEL_PATH = "models/model.joblib"


def predict(X: np.ndarray, model_path: str = MODEL_PATH) -> np.ndarray:
    """Load the trained model and return predictions.

    Parameters
    ----------
    X:
        2-D array of shape ``(n_samples, n_features)``.
    model_path:
        Path to the persisted model file.

    Returns
    -------
    np.ndarray
        Predicted class labels.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Run 'python src/train.py' first."
        )

    model = joblib.load(model_path)
    return model.predict(X)


if __name__ == "__main__":
    import yaml

    # Read feature count from config so the demo matches the trained model
    with open("configs/params.yaml") as _fh:
        _cfg = yaml.safe_load(_fh)
    n_features = _cfg["data"]["n_features"]

    rng = np.random.default_rng(seed=0)
    sample = rng.standard_normal((5, n_features))
    predictions = predict(sample)
    print("Sample predictions:", predictions)
