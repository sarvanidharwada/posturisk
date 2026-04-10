"""Unit tests for the explainability module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')

from posturisk.explain import (
    get_tree_contributions,
    plot_feature_importance,
    plot_individual_force,
    plot_surrogate_beeswarm,
)


@pytest.fixture
def mock_random_forest_pipeline() -> Pipeline:
    """Provides a dummy trained RF model."""
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    scaler = StandardScaler()
    
    # Train heavily separable data
    X_train = np.random.randn(100, 5)
    y_train = (X_train[:, 0] > 0).astype(int)
    
    pipe = Pipeline([
        ("scaler", scaler),
        ("clf", clf)
    ])
    pipe.fit(X_train, y_train)
    return pipe


@pytest.fixture
def mock_feature_matrix() -> pd.DataFrame:
    """Provides dummy feature matrix mapping to the RF."""
    return pd.DataFrame(
        np.random.randn(20, 5),
        columns=["F1", "F2", "F3", "F4", "F5"]
    )


def test_get_tree_contributions(mock_random_forest_pipeline, mock_feature_matrix):
    preds, conts, bias = get_tree_contributions(mock_random_forest_pipeline, mock_feature_matrix)
    
    assert len(preds) == len(mock_feature_matrix), "Predictions mismatch"
    assert conts.shape == (20, 5), "Contributions matrix shape mismatch"
    assert isinstance(bias, float), "Bias should be a float base probability"
    
    # TreeInterpreter logic: pred = bias + sum(contributions)
    reconstructed_preds = bias + np.sum(conts, axis=1)
    np.testing.assert_allclose(preds, reconstructed_preds, rtol=1e-5, atol=1e-5)


def test_plot_feature_importance_generates_file(tmp_path: Path):
    conts = np.random.randn(20, 5)
    out_dir = tmp_path / "figures"
    out_dir.mkdir()
    
    plot_feature_importance(conts, ["F1", "F2", "F3", "F4", "F5"], out_dir, top_n=3)
    
    assert (out_dir / "feature_importance_bar.png").exists()


def test_plot_surrogate_beeswarm_generates_file(tmp_path: Path, mock_feature_matrix):
    conts = np.random.randn(20, 5)
    out_dir = tmp_path / "figures"
    out_dir.mkdir()
    
    plot_surrogate_beeswarm(conts, mock_feature_matrix, out_dir, top_n=3)
    
    assert (out_dir / "summary_beeswarm.png").exists()


def test_plot_individual_force_generates_file(tmp_path: Path):
    conts = np.random.randn(20, 5)
    out_dir = tmp_path / "figures"
    out_dir.mkdir()
    
    plot_individual_force(
        contributions=conts,
        bias=0.45,
        prediction=0.85,
        feature_names=["A", "B", "C", "D", "E"],
        sample_idx=0,
        label="Test Force",
        out_path=out_dir
    )
    
    assert (out_dir / "force_plot_test_force.png").exists()
