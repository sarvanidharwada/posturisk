"""Unit tests for posturisk.train."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from posturisk.train import (
    build_rf_grid,
    build_svm_grid,
    load_data,
    sensitivity_score,
    specificity_score,
    train_models,
)


@pytest.fixture()
def sample_features_csv(tmp_path: Path) -> Path:
    """Create a dummy features.csv with a balanced binary target."""
    data = {
        "subject_id": [f"ID{i}" for i in range(20)],
        "is_faller": [0]*10 + [1]*10,
        "feature_1": np.random.randn(20),
        "feature_2": np.random.randn(20) * 5 + 2,
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestLoadData:
    def test_loads_and_separates_correctly(self, sample_features_csv: Path):
        X, y = load_data(sample_features_csv)
        assert "is_faller" not in X.columns
        assert "subject_id" not in X.columns
        assert "feature_1" in X.columns
        assert y.sum() == 10  # 10 fallers in dummy data


class TestCustomMetrics:
    def test_sensitivity_score(self):
        y_true = [1, 1, 1, 0, 0]
        y_pred = [1, 1, 0, 0, 1]  # 2 TP, 1 FN
        sens = sensitivity_score(y_true, y_pred)
        assert sens == pytest.approx(2/3)

    def test_specificity_score(self):
        y_true = [1, 1, 1, 0, 0, 0]
        y_pred = [1, 0, 0, 0, 0, 1]  # 2 TN, 1 FP
        spec = specificity_score(y_true, y_pred)
        assert spec == pytest.approx(2/3)


class TestModelGrids:
    def test_svm_grid_structure(self):
        gs = build_svm_grid()
        assert gs.refit == "roc_auc"
        # Access the base pipeline
        pipeline = gs.estimator
        assert isinstance(pipeline, Pipeline)
        assert "scaler" in pipeline.named_steps
        assert "clf" in pipeline.named_steps

    def test_rf_grid_structure(self):
        gs = build_rf_grid()
        assert gs.refit == "roc_auc"
        pipeline = gs.estimator
        assert "scaler" in pipeline.named_steps
        assert "clf" in pipeline.named_steps


class TestTrainModels:
    def test_training_pipeline_executes_successfully(self, sample_features_csv: Path):
        # Mute logging outputs in test output
        X, y = load_data(sample_features_csv)
        
        # We need more samples to successfully do 5-fold CV
        # Multiply testing data
        X = pd.concat([X] * 3, ignore_index=True)
        y = pd.concat([y] * 3, ignore_index=True)
        
        results = train_models(X, y)
        assert "best_model" in results
        assert results["best_name"] in ["SVM", "RandomForest"]
        # Resulting model should be fitted
        assert hasattr(results["best_model"], "predict")
