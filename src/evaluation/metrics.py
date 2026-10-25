"""Comprehensive fraud detection evaluation metrics and cost-sensitive analysis."""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import ADMINISTRATIVE_INVESTIGATION_COST
from src.utils.logger import logger

def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Finds decision threshold that optimizes F1-score or balance between precision and recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    # Avoid division by zero
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-9)
    best_idx = np.argmax(f1_scores)
    # thresholds array has length len(precisions) - 1
    best_thresh = float(thresholds[min(best_idx, len(thresholds) - 1)])
    best_val = float(f1_scores[best_idx])
    return best_thresh, best_val

def compute_cost_sensitive_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    c_fp: float = ADMINISTRATIVE_INVESTIGATION_COST,
) -> Dict[str, float]:
    """Computes real-world financial cost metrics (Dal Pozzolo et al., 2015).
    
    Cost model:
    - Missed fraud (FN): Loss = Transaction Amount
    - False Alarm (FP): Loss = Administrative Investigation Cost ($10)
    - Caught Fraud (TP): Loss = $0 (or C_fp, but transaction was stopped, saving the Amount)
    - True Negative (TN): Loss = $0
    """
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    tp_mask = (y_true == 1) & (y_pred == 1)

    total_fraud_loss_potential = float(amounts[y_true == 1].sum())
    fraud_amount_prevented = float(amounts[tp_mask].sum())
    fraud_amount_missed = float(amounts[fn_mask].sum())
    investigation_cost = float(fp_mask.sum() * c_fp)

    net_financial_savings = fraud_amount_prevented - investigation_cost
    savings_ratio = (net_financial_savings / total_fraud_loss_potential * 100) if total_fraud_loss_potential > 0 else 0.0

    return {
        "total_fraud_exposure_usd": round(total_fraud_loss_potential, 2),
        "fraud_prevented_usd": round(fraud_amount_prevented, 2),
        "fraud_missed_usd": round(fraud_amount_missed, 2),
        "investigation_overhead_usd": round(investigation_cost, 2),
        "net_financial_savings_usd": round(net_financial_savings, 2),
        "savings_percentage": round(savings_ratio, 2),
    }

def evaluate_model_performance(
    model_name: str,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    amounts: np.ndarray,
    default_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Generates complete statistical scorecard for a model."""
    # AUC metrics
    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_scores))
    except Exception:
        pr_auc = 0.0

    # Optimal threshold by F1
    best_thresh, best_f1 = find_optimal_threshold(y_true, y_scores)

    # Standard 0.5 threshold evaluation
    y_pred_def = (y_scores >= default_threshold).astype(int)
    cm_def = confusion_matrix(y_true, y_pred_def)
    tn_d, fp_d, fn_d, tp_d = cm_def.ravel() if cm_def.size == 4 else (0, 0, 0, 0)

    # Optimal threshold evaluation
    y_pred_opt = (y_scores >= best_thresh).astype(int)
    cm_opt = confusion_matrix(y_true, y_pred_opt)
    tn_o, fp_o, fn_o, tp_o = cm_opt.ravel() if cm_opt.size == 4 else (0, 0, 0, 0)

    # Cost-sensitive evaluation at optimal threshold
    cost_stats = compute_cost_sensitive_stats(y_true, y_pred_opt, amounts)

    # Top alerts quality (e.g., top 100 highest-confidence alerts)
    top_k = min(200, len(y_scores))
    top_k_indices = np.argsort(y_scores)[::-1][:top_k]
    top_k_frauds = int(y_true[top_k_indices].sum())
    top_k_precision = round((top_k_frauds / top_k) * 100, 2)

    stats = {
        "model_name": model_name,
        "roc_auc": round(roc_auc, 5),
        "pr_auc": round(pr_auc, 5),
        "optimal_threshold": round(best_thresh, 4),
        "metrics_at_optimal_threshold": {
            "f1_score": round(float(f1_score(y_true, y_pred_opt, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred_opt, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred_opt, zero_division=0)), 4),
            "true_positives": int(tp_o),
            "false_positives": int(fp_o),
            "false_negatives": int(fn_o),
            "true_negatives": int(tn_o),
        },
        "metrics_at_default_threshold_0_5": {
            "f1_score": round(float(f1_score(y_true, y_pred_def, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred_def, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred_def, zero_division=0)), 4),
            "true_positives": int(tp_d),
            "false_positives": int(fp_d),
            "false_negatives": int(fn_d),
            "true_negatives": int(tn_d),
        },
        "cost_sensitive_analysis": cost_stats,
        "top_k_alert_precision": {
            f"top_{top_k}_alerts": f"{top_k_frauds}/{top_k} frauds detected ({top_k_precision}%)"
        },
    }

    logger.info(
        f"🎯 [{model_name}] ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | "
        f"Opt F1: {stats['metrics_at_optimal_threshold']['f1_score']} | "
        f"Savings: ${cost_stats['net_financial_savings_usd']:,} ({cost_stats['savings_percentage']}%)"
    )

    return stats

def compare_all_models(evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
    """Creates a consolidated comparison table across all evaluated models."""
    rows = []
    for ev in evaluations:
        opt = ev["metrics_at_optimal_threshold"]
        cost = ev["cost_sensitive_analysis"]
        rows.append({
            "Model": ev["model_name"],
            "ROC-AUC": ev["roc_auc"],
            "PR-AUC": ev["pr_auc"],
            "Optimal Thresh": ev["optimal_threshold"],
            "Opt F1": opt["f1_score"],
            "Opt Precision": opt["precision"],
            "Opt Recall": opt["recall"],
            "TP (Caught)": opt["true_positives"],
            "FN (Missed)": opt["false_negatives"],
            "FP (False Alarm)": opt["false_positives"],
            "Net Savings ($)": cost["net_financial_savings_usd"],
            "Savings (%)": f"{cost['savings_percentage']}%",
        })
    df_compare = pd.DataFrame(rows)
    return df_compare.sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
