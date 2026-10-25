"""Visualization routines for publication-ready performance reports."""

from typing import Dict, List, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.config import FIGURES_DIR
from src.utils.logger import logger

# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

def plot_roc_curves(
    model_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Path = FIGURES_DIR / "roc_curves_comparison.png",
):
    """Plots multi-model ROC Curves."""
    plt.figure(figsize=(8, 6), dpi=300)
    
    for name, y_scores in model_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (ROC-AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    plt.title("ROC Curves - Credit Card Fraud Detection", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC Curves plot to {save_path}")

def plot_pr_curves(
    model_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Path = FIGURES_DIR / "pr_curves_comparison.png",
):
    """Plots Precision-Recall Curves (the most informative metric for 0.172% fraud imbalance)."""
    plt.figure(figsize=(8, 6), dpi=300)
    
    base_rate = float(y_true.mean())
    plt.plot([0, 1], [base_rate, base_rate], linestyle="--", color="gray", lw=1.5,
             label=f"No-Skill Baseline ({base_rate*100:.3f}%)")

    for name, y_scores in model_predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Fraction of Frauds Caught)", fontweight="bold")
    plt.ylabel("Precision (Fraction of Alerts that are Real Fraud)", fontweight="bold")
    plt.title("Precision-Recall Curves (Critical for Imbalanced Fraud)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"📊 Saved Precision-Recall Curves plot to {save_path}")

def plot_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    save_path: Path = FIGURES_DIR / "confusion_matrices.png",
):
    """Plots side-by-side Confusion Matrix heatmaps."""
    n_models = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5), dpi=300)
    if n_models == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(f"Confusion Matrix: {name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label (0: Legitimate, 1: Fraud)", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
        ax.set_xticklabels(["Normal", "Fraud"])
        ax.set_yticklabels(["Normal", "Fraud"])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"📊 Saved Confusion Matrices plot to {save_path}")

def plot_feature_importance(
    df_importance: pd.DataFrame,
    top_n: int = 15,
    save_path: Path = FIGURES_DIR / "feature_importance.png",
):
    """Plots top N predictive features."""
    plt.figure(figsize=(9, 6), dpi=300)
    top_df = df_importance.head(top_n).sort_values(by=df_importance.columns[1], ascending=True)
    
    col_name = df_importance.columns[1]
    colors = sns.color_palette("viridis", len(top_df))
    plt.barh(top_df["feature"], top_df[col_name], color=colors)
    plt.xlabel(f"Relative Importance ({col_name})", fontweight="bold")
    plt.title(f"Top {top_n} Fraud Predictive Features", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"📊 Saved Feature Importance plot to {save_path}")

def plot_cost_savings(
    evaluations: List[Dict[str, Any]],
    save_path: Path = FIGURES_DIR / "cost_savings_comparison.png",
):
    """Plots Net Financial Savings ($) across models."""
    plt.figure(figsize=(8, 5), dpi=300)
    names = [ev["model_name"] for ev in evaluations]
    savings = [ev["cost_sensitive_analysis"]["net_financial_savings_usd"] for ev in evaluations]
    prevented = [ev["cost_sensitive_analysis"]["fraud_prevented_usd"] for ev in evaluations]
    overhead = [ev["cost_sensitive_analysis"]["investigation_overhead_usd"] for ev in evaluations]

    x = np.arange(len(names))
    width = 0.28

    plt.bar(x - width, prevented, width, label="Fraud Prevented ($)", color="#2ca02c")
    plt.bar(x, overhead, width, label="Investigation Overhead ($)", color="#d62728")
    plt.bar(x + width, savings, width, label="Net Financial Savings ($)", color="#1f77b4")

    plt.xticks(x, names, fontweight="bold")
    plt.ylabel("USD ($)", fontweight="bold")
    plt.title("Dal Pozzolo Cost-Sensitive Financial Impact Analysis", fontsize=12, fontweight="bold", pad=12)
    plt.legend(frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"📊 Saved Cost-Savings plot to {save_path}")

def plot_and_save_all_figures(
    model_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    evaluations: List[Dict[str, Any]],
    feature_importance_df: pd.DataFrame = None,
):
    """Generates all evaluation figures."""
    plot_roc_curves(model_predictions, y_true)
    plot_pr_curves(model_predictions, y_true)

    # Build confusion matrices dict
    cms = {}
    for ev in evaluations:
        opt = ev["metrics_at_optimal_threshold"]
        cm = np.array([
            [opt["true_negatives"], opt["false_positives"]],
            [opt["false_negatives"], opt["true_positives"]],
        ])
        cms[ev["model_name"]] = cm

    plot_confusion_matrices(cms)
    plot_cost_savings(evaluations)

    if feature_importance_df is not None:
        plot_feature_importance(feature_importance_df)
