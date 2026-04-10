"""Model training and evaluation pipeline.

Trains SVM and Random Forest models using Nested / Repeated Stratified
K-Fold CV, extracts standardized evaluation metrics, and saves the best model.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from posturisk.preprocess import DEFAULT_PROCESSED_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def load_data(features_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and separate into X and y.

    Parameters
    ----------
    features_path : Path
        Path to features.csv.

    Returns
    -------
    X, y
        Feature dataframe (dropped IDs) and target series.
    """
    df = pd.read_csv(features_path)
    
    if "is_faller" not in df.columns:
        raise ValueError("Target 'is_faller' not found in dataset.")

    y = df["is_faller"]
    X = df.drop(columns=["is_faller", "subject_id"], errors="ignore")
    
    return X, y


def sensitivity_score(y_true, y_pred) -> float:
    """Calculate sensitivity (recall for positive class)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def specificity_score(y_true, y_pred) -> float:
    """Calculate specificity (recall for negative class)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


# Custom scorer dictionary for GridSearchCV refit (if desired)
# Though we will use standard accuracy or roc_auc for refitting
SCORING = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "sensitivity": make_scorer(sensitivity_score),
    "specificity": make_scorer(specificity_score),
}


def build_svm_grid() -> GridSearchCV:
    """Initialize an SVM GridSearchCV block."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=42))
    ])
    
    param_grid = {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto", 0.01, 0.1],
        "clf__kernel": ["rbf"]
    }
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    return GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=SCORING,
        refit="roc_auc",  # Refit the best model using ROC AUC
        return_train_score=False,
        n_jobs=-1
    )


def build_rf_grid() -> GridSearchCV:
    """Initialize a Random Forest GridSearchCV block."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # RF doesn't strictly need it, but good for uniformity
        ("clf", RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4]
    }
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    return GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=SCORING,
        refit="roc_auc",
        return_train_score=False,
        n_jobs=-1
    )


def train_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train estimators and select the best one based on CV AUC.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Target labels.

    Returns
    -------
    dict
        Dictionary containing both trained GridSearchCV objects and the best overall model.
    """
    logger.info("Training Support Vector Machine (RBF) grid...")
    svm_gs = build_svm_grid()
    svm_gs.fit(X, y)
    
    logger.info("Training Random Forest grid...")
    rf_gs = build_rf_grid()
    rf_gs.fit(X, y)
    
    best_svm_auc = svm_gs.best_score_
    best_rf_auc = rf_gs.best_score_
    
    logger.info(f"SVM Best CV ROC-AUC: {best_svm_auc:.3f}")
    logger.info(f"RF Best CV ROC-AUC:  {best_rf_auc:.3f}")
    
    if best_rf_auc > best_svm_auc:
        best_model = rf_gs.best_estimator_
        best_name = "RandomForest"
    else:
        best_model = svm_gs.best_estimator_
        best_name = "SVM"
        
    logger.info(f"Winner: {best_name}")
    
    return {
        "svm": svm_gs,
        "rf": rf_gs,
        "best_model": best_model,
        "best_name": best_name
    }


def evaluate_model_holdout(model, X_test, y_test) -> dict:
    """Evaluate a trained model against a holdout test set (if required)."""
    # Not used inside the main training script right now as we rely entirely on CV
    # due to n=73, but provided for the notebook utility.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "sensitivity": sensitivity_score(y_test, y_pred),
        "specificity": specificity_score(y_test, y_pred)
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train fall-risk ML models.")
    parser.add_argument("--features", type=Path, default=DEFAULT_PROCESSED_DIR / "features.csv",
                        help="Path to preprocessed features.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MODELS_DIR,
                        help="Directory to save the trained best model.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {args.features}")
    X, y = load_data(args.features)
    
    logger.info(f"Dataset shape: {X.shape}. Fallers: {y.sum()} / {len(y)}")
    
    results = train_models(X, y)
    
    out_path = args.out_dir / "best_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results["best_model"], f)
        
    logger.info(f"Saved best model ({results['best_name']}) pipeline to {out_path}")

if __name__ == "__main__":
    main()
