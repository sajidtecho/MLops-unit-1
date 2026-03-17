"""Tests for the MLOps unit-1 training pipeline."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

# Allow imports from src/ when running tests from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import generate_dataset, load_dataset
from train import build_model, load_config, train


# ---------------------------------------------------------------------------
# data_loader tests
# ---------------------------------------------------------------------------


class TestGenerateDataset:
    def test_returns_dataframe_with_correct_shape(self, tmp_path):
        output = str(tmp_path / "dataset.csv")
        df = generate_dataset(
            n_samples=50, n_features=5, n_informative=3, output_path=output
        )
        assert df.shape == (50, 6)  # 5 features + target

    def test_saves_csv_to_disk(self, tmp_path):
        output = str(tmp_path / "dataset.csv")
        generate_dataset(n_samples=20, n_features=4, n_informative=2, output_path=output)
        assert os.path.exists(output)

    def test_target_column_is_binary(self, tmp_path):
        output = str(tmp_path / "dataset.csv")
        df = generate_dataset(n_samples=100, n_features=4, n_informative=2, output_path=output)
        assert set(df["target"].unique()).issubset({0, 1})


class TestLoadDataset:
    def test_raises_when_file_missing(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/dataset.csv")

    def test_roundtrip(self, tmp_path):
        output = str(tmp_path / "dataset.csv")
        original = generate_dataset(
            n_samples=30, n_features=4, n_informative=2, output_path=output
        )
        loaded = load_dataset(output)
        assert loaded.shape == original.shape


# ---------------------------------------------------------------------------
# train tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        cfg_file = tmp_path / "params.yaml"
        cfg_file.write_text(
            "model:\n  type: RandomForestClassifier\ntraining:\n  test_size: 0.2\n  random_seed: 42\ndata:\n  n_samples: 50\n  n_features: 5\n  n_informative: 3\n"
        )
        cfg = load_config(str(cfg_file))
        assert cfg["model"]["type"] == "RandomForestClassifier"
        assert cfg["training"]["test_size"] == 0.2


class TestBuildModel:
    def test_random_forest(self):
        cfg = {"model": {"type": "RandomForestClassifier", "n_estimators": 10, "max_depth": 3, "random_state": 42}}
        model = build_model(cfg)
        assert model.__class__.__name__ == "RandomForestClassifier"

    def test_logistic_regression(self):
        cfg = {"model": {"type": "LogisticRegression", "max_iter": 100, "random_state": 0}}
        model = build_model(cfg)
        assert model.__class__.__name__ == "LogisticRegression"

    def test_unknown_model_raises(self):
        cfg = {"model": {"type": "UnknownModel"}}
        with pytest.raises(ValueError, match="Unsupported model type"):
            build_model(cfg)


class TestTrain:
    def test_full_pipeline_produces_metrics(self, tmp_path):
        cfg_content = (
            "model:\n"
            "  type: RandomForestClassifier\n"
            "  n_estimators: 10\n"
            "  max_depth: 3\n"
            "  random_state: 42\n"
            "training:\n"
            "  test_size: 0.2\n"
            "  random_seed: 42\n"
            "data:\n"
            "  n_samples: 100\n"
            "  n_features: 5\n"
            "  n_informative: 3\n"
        )
        cfg_file = tmp_path / "params.yaml"
        cfg_file.write_text(cfg_content)

        # Patch module-level paths so outputs land in tmp_path
        import train as train_module

        orig_model = train_module.MODEL_PATH
        orig_metrics = train_module.METRICS_PATH
        orig_data = train_module.DATA_PATH
        data_dir = tmp_path / "data" / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            train_module.MODEL_PATH = str(tmp_path / "model.joblib")
            train_module.METRICS_PATH = str(tmp_path / "metrics.json")
            train_module.DATA_PATH = str(data_dir / "dataset.csv")

            metrics = train(config_path=str(cfg_file))
        finally:
            train_module.MODEL_PATH = orig_model
            train_module.METRICS_PATH = orig_metrics
            train_module.DATA_PATH = orig_data

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
