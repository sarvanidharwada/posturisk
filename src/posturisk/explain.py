"""Explainability module mimicking SHAP with pure python (treeinterpreter).

Provides Feature Importance, Swarm-like plots, and Individual Force plots.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from treeinterpreter import treeinterpreter as ti

from posturisk.preprocess import PROJECT_ROOT
from posturisk.train import load_data

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"


def get_tree_contributions(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate feature contributions using treeinterpreter.
    
    Returns
    -------
    contributions (n_samples, n_features)
    base_value
    """
    scaler = model.named_steps.get("scaler")
    clf = model.named_steps["clf"]
    
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values
        
    prediction, bias, contributions = ti.predict(clf, X_scaled)
    
    # Random Forest returns contributions covering all classes:
    # shape is (n_samples, n_features, n_classes)
    # We slice out the contributions pointing to Class 1 (Faller)
    conts_class_1 = contributions[:, :, 1]
    bias_class_1 = bias[0, 1]
    
    return prediction[:, 1], conts_class_1, bias_class_1


def plot_feature_importance(contributions: np.ndarray, feature_names: list[str],
                            out_path: Path, top_n: int = 15) -> None:
    """Plot global feature importance based on absolute mean contribution."""
    mean_abs_cont = np.abs(contributions).mean(axis=0)
    
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": mean_abs_cont
    }).sort_values("Importance", ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=df_imp, palette="viridis")
    plt.title("Global Feature Importance (Mean Absolute Contribution)")
    plt.xlabel("mean(|Contribution|)")
    plt.tight_layout()
    plt.savefig(out_path / "feature_importance_bar.png", dpi=300)
    plt.close()


def plot_surrogate_beeswarm(contributions: np.ndarray, X: pd.DataFrame, 
                            out_path: Path, top_n: int = 15) -> None:
    """Simulate a SHAP summary beeswarm plot mapping values to impacts."""
    mean_abs_cont = np.abs(contributions).mean(axis=0)
    top_indices = np.argsort(mean_abs_cont)[::-1][:top_n]
    
    plot_data = []
    
    for idx in top_indices:
        feat_name = X.columns[idx]
        feat_vals = X.iloc[:, idx].values
        
        # Normalize feature values to [0, 1] for coloring
        min_v, max_v = feat_vals.min(), feat_vals.max()
        norm_v = (feat_vals - min_v) / (max_v - min_v + 1e-9)
        
        conts = contributions[:, idx]
        
        for v, c in zip(norm_v, conts):
            plot_data.append({
                "Feature": feat_name,
                "Contribution": c,
                "Value": v
            })
            
    df_plot = pd.DataFrame(plot_data)
    
    # Must preserve the exact order of top features
    order = X.columns[top_indices]
    
    plt.figure(figsize=(10, 7))
    sns.stripplot(x="Contribution", y="Feature", data=df_plot, hue="Value",
                  palette="coolwarm", size=4, alpha=0.8, order=order, dodge=False, legend=False)
    
    plt.axvline(0, color="gray", linestyle="--", alpha=0.5)
    plt.title("Summary Impact Plot (Surrogate SHAP Beeswarm)")
    plt.xlabel("Feature Contribution to 'Faller' Risk")
    plt.tight_layout()
    plt.savefig(out_path / "summary_beeswarm.png", dpi=300)
    plt.close()


def plot_individual_force(contributions: np.ndarray, bias: float, prediction: float,
                          feature_names: list[str], sample_idx: int, label: str,
                          out_path: Path) -> None:
    """Plot an individual waterfall/force diagram for a single prediction."""
    conts = contributions[sample_idx]
    
    # Filter only meaningful contributions to reduce noise
    mask = np.abs(conts) > 0.01
    filt_names = np.array(feature_names)[mask]
    filt_conts = conts[mask]
    
    # Sort by contribution value
    sort_idx = np.argsort(filt_conts)
    filt_names = filt_names[sort_idx]
    filt_conts = filt_conts[sort_idx]
    
    plt.figure(figsize=(10, len(filt_names)*0.3 + 2))
    colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in filt_conts]
    
    plt.barh(filt_names, filt_conts, color=colors)
    plt.axvline(0, color='black', lw=1)
    
    plt.title(f"Individual Force Plot: {label}\nBase Probability: {bias:.2f} \u2192 Final Probability: {prediction:.2f}")
    plt.xlabel("Contribution to Default Probability")
    plt.tight_layout()
    plt.savefig(out_path / f"force_plot_{label.replace(' ', '_').lower()}.png", dpi=300)
    plt.close()


def generate_explanations(model_path: Path, features_path: Path, out_dir: Path) -> None:
    """Load model, compute tree explanations, and save plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    if not hasattr(model.named_steps["clf"], "estimators_"):
        logger.error("TreeInterpreter requires a Tree-based model (Random Forest). "
                     "If the best model was an SVM, Explainability operates differently natively.")
        return
        
    X, y = load_data(features_path)
    
    logger.info("Computing tree interpreter contributions...")
    preds, conts, bias = get_tree_contributions(model, X)
    
    logger.info("Generating Global Feature Importance...")
    plot_feature_importance(conts, X.columns.tolist(), out_dir)
    
    logger.info("Generating Surrogate Beeswarm Plot...")
    plot_surrogate_beeswarm(conts, X, out_dir)
    
    # Generate Force plots for some specific samples
    y_pred_binary = (preds > 0.5).astype(int)
    
    correct_faller = np.where((y == 1) & (y_pred_binary == 1))[0]
    incorrect_faller = np.where((y == 1) & (y_pred_binary == 0))[0]
    
    if len(correct_faller) > 0:
        idx = correct_faller[0]
        logger.info(f"Generating Correctly Classified Faller Plot (Index {idx})")
        plot_individual_force(conts, bias, preds[idx], X.columns.tolist(), idx, "Correct Faller", out_dir)
        
    if len(incorrect_faller) > 0:
        idx = incorrect_faller[0]
        logger.info(f"Generating Missed Faller Plot (Index {idx})")
        plot_individual_force(conts, bias, preds[idx], X.columns.tolist(), idx, "False Negative", out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    model_path = DEFAULT_MODELS_DIR / "best_model.pkl"
    features_path = PROJECT_ROOT / "data" / "processed" / "features.csv"
    
    generate_explanations(model_path, features_path, DEFAULT_REPORTS_DIR)


if __name__ == "__main__":
    main()
