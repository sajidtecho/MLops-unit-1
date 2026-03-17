"""train.py – Train a model using settings from configs/params.yaml.

This script is intentionally simple so that it is easy to understand and
reproduce.  Every meaningful choice (hyperparameters, split ratios, random
seeds) is read from the config file so that a single ``git diff
configs/params.yaml`` reveals exactly what changed between two experiments.

Usage
-----
    python src/train.py
"""
from __future__ import annotations

import json
import os

import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from data_loader import generate_dataset, load_dataset

CONFIG_PATH = "configs/params.yaml"
MODEL_PATH = "models/model.joblib"
METRICS_PATH = "models/metrics.json"
DATA_PATH = "data/raw/dataset.csv"


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load YAML configuration."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def build_model(cfg: dict):
    """Instantiate a model from the configuration.

    Supports ``RandomForestClassifier`` and ``LogisticRegression``.
    """
    model_type = cfg["model"]["type"]
    if model_type == "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=cfg["model"].get("n_estimators", 100),
            max_depth=cfg["model"].get("max_depth", None),
            random_state=cfg["model"].get("random_state", 42),
        )
    if model_type == "LogisticRegression":
        return LogisticRegression(
            max_iter=cfg["model"].get("max_iter", 1000),
            random_state=cfg["model"].get("random_state", 42),
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def train(config_path: str = CONFIG_PATH) -> dict:
    """Run the full training pipeline.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.

    Returns
    -------
    dict
        Dictionary with ``accuracy`` and ``f1`` metrics.
    """
    cfg = load_config(config_path)

    # ------------------------------------------------------------------
    # 1. Load (or generate) data
    # ------------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print("Dataset not found – generating synthetic data …")
        df = generate_dataset(
            n_samples=cfg["data"]["n_samples"],
            n_features=cfg["data"]["n_features"],
            n_informative=cfg["data"]["n_informative"],
            random_state=cfg["training"]["random_seed"],
            output_path=DATA_PATH,
        )
    else:
        df = load_dataset(DATA_PATH)

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    # ------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_seed"],
    )

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    model = build_model(cfg)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "model_type": cfg["model"]["type"],
        "n_estimators": cfg["model"].get("n_estimators"),
        "max_depth": cfg["model"].get("max_depth"),
    }

    # ------------------------------------------------------------------
    # 5. Persist model and metrics
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Training complete.")
    print(f"  Accuracy : {metrics['accuracy']}")
    print(f"  F1 score : {metrics['f1']}")
    print(f"Model saved to  {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train()
