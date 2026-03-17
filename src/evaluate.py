"""evaluate.py – Load the saved model and print evaluation metrics.

Usage
-----
    python src/evaluate.py
"""
from __future__ import annotations

import json
import os

METRICS_PATH = "models/metrics.json"


def evaluate(metrics_path: str = METRICS_PATH) -> dict:
    """Print the metrics that were saved during training.

    Parameters
    ----------
    metrics_path:
        Path to the JSON file produced by ``train.py``.

    Returns
    -------
    dict
        The metrics dictionary.
    """
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Metrics file not found at '{metrics_path}'. "
            "Run 'python src/train.py' first."
        )

    with open(metrics_path, "r") as fh:
        metrics = json.load(fh)

    print("=" * 40)
    print("Model Evaluation Results")
    print("=" * 40)
    for key, value in metrics.items():
        print(f"  {key:<20}: {value}")
    print("=" * 40)
    return metrics


if __name__ == "__main__":
    evaluate()
