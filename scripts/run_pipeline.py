"""End-to-End Orchestration Pipeline for Credit Card Fraud Detection ML Engine.

Runs:
1. Hardware & GPU device check
2. Dataset streaming download & verification
3. BigQuery ML equivalent preprocessing & sequential split
4. Training Baseline Logistic Regression (BigQuery ML replica)
5. Training GPU-Accelerated XGBoost
6. Training Deep Neural Network (FraudNet)
7. Training Deep Autoencoder Anomaly Detector
8. Full evaluation, cost-sensitive economic benchmark, and metric reporting
9. High-resolution figure generation
10. Query simulation on Time = 85285.0 (as specified in user prompt)
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    DEVICE,
    TARGET_COL,
    AMOUNT_COL,
    TIME_COL,
)
from src.utils.logger import logger
from src.utils.gpu_utils import check_gpu_status
from src.data.download import download_dataset
from src.data.preprocess import preprocess_and_split
from src.data.dataset import create_data_loaders
from src.models.baseline import BigQueryLogisticRegressionBaseline
from src.models.xgboost_gpu import GPUXGBoostModel
from src.models.neural_net import FraudNet
from src.models.autoencoder import FraudAutoencoder
from src.engine.trainer import train_fraudnet, train_autoencoder
from src.engine.predictor import FraudPredictor
from src.evaluation.metrics import evaluate_model_performance, compare_all_models
from src.evaluation.visualizer import plot_and_save_all_figures

console = Console()

def main():
    total_start = time.time()
    console.print(Panel.fit(
        "[bold cyan]CREDIT CARD FRAUD DETECTION - ML ENGINE PIPELINE[/bold cyan]\n"
        "[dim]GPU-Accelerated Tabular Deep Learning & Gradient Boosting Architecture[/dim]",
        border_style="cyan"
    ))

    # ==========================================================
    # STEP 1: Hardware & Device Status
    # ==========================================================
    console.rule("[bold yellow]Step 1: Hardware & GPU Status[/bold yellow]")
    gpu_info = check_gpu_status()
    console.print(f"[green]Compute Configuration:[/green] {gpu_info['status_summary']}")
    if gpu_info.get("total_vram_gb", 0) > 0:
        console.print(f"[green]GPU VRAM Detected:[/green] {gpu_info['total_vram_gb']} GB")

    # ==========================================================
    # STEP 2: Download Dataset
    # ==========================================================
    console.rule("[bold yellow]Step 2: Dataset Ingestion[/bold yellow]")
    dataset_path = download_dataset(force=False)
    console.print(f"[green]Raw dataset stored at:[/green] {dataset_path}")

    # ==========================================================
    # STEP 3: Preprocessing & Temporal Sequential Split
    # ==========================================================
    console.rule("[bold yellow]Step 3: Preprocessing & BigQuery ML Sequential Split[/bold yellow]")
    train_df, val_df, test_df, metadata = preprocess_and_split()

    feature_cols = metadata["split_summary"]["feature_cols"]
    v_features = [f"V{i}" for i in range(1, 29)]
    console.print(f"[dim]Features engineered: {len(feature_cols)} features ({', '.join(feature_cols[:4])}...)[/dim]")

    # Extract NumPy matrices for sklearn / xgboost
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values

    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values

    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COL].values
    test_amounts = test_df[AMOUNT_COL].values

    # Build PyTorch DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, feature_cols
    )

    evaluations = []
    test_scores_dict = {}

    # ==========================================================
    # STEP 4: BigQuery Baseline Model (Logistic Regression)
    # ==========================================================
    console.rule("[bold yellow]Step 4: Train BigQuery ML Logistic Regression Baseline[/bold yellow]")
    baseline = BigQueryLogisticRegressionBaseline()
    baseline.fit(X_train, y_train, feature_cols=feature_cols)
    baseline.save()

    baseline_test_probs = baseline.predict_proba(X_test)[:, 1]
    test_scores_dict["BigQuery Baseline (LogReg)"] = baseline_test_probs
    eval_baseline = evaluate_model_performance(
        "BigQuery Baseline (LogReg)", y_test, baseline_test_probs, test_amounts
    )
    evaluations.append(eval_baseline)

    # ==========================================================
    # STEP 5: GPU-Accelerated XGBoost Classifier
    # ==========================================================
    console.rule("[bold yellow]Step 5: Train GPU-Accelerated XGBoost Classifier[/bold yellow]")
    xgb_model = GPUXGBoostModel()
    xgb_model.fit(X_train, y_train, X_val, y_val, feature_cols=feature_cols)
    xgb_model.save()

    xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]
    test_scores_dict["XGBoost (GPU)"] = xgb_test_probs
    eval_xgb = evaluate_model_performance(
        "XGBoost (GPU)", y_test, xgb_test_probs, test_amounts
    )
    evaluations.append(eval_xgb)

    # ==========================================================
    # STEP 6: Deep Tabular Neural Network (FraudNet)
    # ==========================================================
    console.rule("[bold yellow]Step 6: Train Deep Neural Network (FraudNet with Focal Loss)[/bold yellow]")
    fraudnet = FraudNet(input_dim=len(feature_cols))
    train_fraudnet(fraudnet, train_loader, val_loader, device=DEVICE)

    # Evaluate FraudNet on Test
    fraudnet.eval()
    fraudnet_probs = []
    import torch
    with torch.no_grad():
        for batch in test_loader:
            X_b = batch[0].to(DEVICE)
            probs = fraudnet.predict_proba(X_b)
            fraudnet_probs.extend(probs.cpu().numpy().flatten())
    fraudnet_probs = np.array(fraudnet_probs)
    test_scores_dict["FraudNet (Deep MLP)"] = fraudnet_probs
    eval_fraudnet = evaluate_model_performance(
        "FraudNet (Deep MLP)", y_test, fraudnet_probs, test_amounts
    )
    evaluations.append(eval_fraudnet)

    # ==========================================================
    # STEP 7: Deep Autoencoder Anomaly Detector
    # ==========================================================
    console.rule("[bold yellow]Step 7: Train Deep Autoencoder (Unsupervised Anomaly Detector)[/bold yellow]")
    # Normal transactions only for training
    train_normal_df = train_df[train_df[TARGET_COL] == 0]
    val_normal_df = val_df[val_df[TARGET_COL] == 0]
    train_ae_loader, val_ae_loader, _ = create_data_loaders(
        train_normal_df, val_normal_df, test_df, feature_cols, batch_size=2048
    )

    autoencoder = FraudAutoencoder(input_dim=len(feature_cols))
    train_autoencoder(autoencoder, train_ae_loader, val_ae_loader, device=DEVICE)

    # Compute validation normal scores to set anomaly threshold (99th percentile)
    val_normal_tensor = torch.tensor(val_normal_df[feature_cols].values, dtype=torch.float32).to(DEVICE)
    val_normal_scores = autoencoder.compute_anomaly_scores(val_normal_tensor)
    autoencoder.set_threshold(val_normal_scores, percentile=98.5)
    autoencoder.save()

    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    ae_anomaly_scores = autoencoder.compute_anomaly_scores(test_tensor)
    # Min-max scale anomaly scores into [0, 1] pseudo-probabilities for ROC comparison
    ae_norm_scores = (ae_anomaly_scores - ae_anomaly_scores.min()) / max(1e-9, (ae_anomaly_scores.max() - ae_anomaly_scores.min()))
    test_scores_dict["Autoencoder (Anomaly Detection)"] = ae_norm_scores
    eval_ae = evaluate_model_performance(
        "Autoencoder (Anomaly Detection)", y_test, ae_norm_scores, test_amounts
    )
    evaluations.append(eval_ae)

    # ==========================================================
    # STEP 8: Model Comparison & Statistics Scorecard
    # ==========================================================
    console.rule("[bold green]Step 8: Comparative Evaluation Scorecard[/bold green]")
    comparison_df = compare_all_models(evaluations)

    table = Table(title="Fraud Detection Engine Benchmark Results", show_lines=True)
    for col in comparison_df.columns:
        table.add_column(col, justify="center" if col != "Model" else "left")

    for _, row in comparison_df.iterrows():
        table.add_row(*[str(val) for val in row.values])
    console.print(table)

    # Save summary json
    summary_path = REPORTS_DIR / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2)
    console.print(f"[green]Saved evaluation summary to:[/green] {summary_path}")

    # ==========================================================
    # STEP 9: Generate High-Resolution Visualization Figures
    # ==========================================================
    console.rule("[bold green]Step 9: Generating Figures & Curves[/bold green]")
    feat_imp = xgb_model.get_feature_importance()
    plot_and_save_all_figures(test_scores_dict, y_test, evaluations, feat_imp)
    console.print(f"[green]Saved ROC/PR curves and heatmaps to:[/green] {REPORTS_DIR / 'figures'}")

    # ==========================================================
    # STEP 10: BigQuery ML Test Query Re-creation (Time = 85285.0)
    # ==========================================================
    console.rule("[bold cyan]Step 10: BigQuery ML Prompt Query Validation (Time = 85285.0)[/bold cyan]")
    # Locate rows with Time = 85285.0 in the raw dataset (matches BigQuery ML test query)
    raw_df = pd.read_csv(RAW_DATA_PATH)
    target_samples = raw_df[np.isclose(raw_df[TIME_COL], 85285.0, atol=0.1)]

    console.print(f"[bold]Query Target:[/bold] Found {len(target_samples)} transactions at Time = 85285.0")

    pred_table = Table(title="BigQuery ML.PREDICT Replication (SELECT * WHERE Time = 85285.0)", show_lines=True)
    pred_table.add_column("Tx #", justify="center")
    pred_table.add_column("Model", style="bold cyan")
    pred_table.add_column("Amount ($)", justify="right")
    pred_table.add_column("P(Normal / 0)", justify="right")
    pred_table.add_column("P(Fraud / 1)", justify="right")
    pred_table.add_column("Predicted Class", justify="center")
    pred_table.add_column("Actual Class", justify="center")
    pred_table.add_column("Status", justify="center")

    predictors = {
        "BIGQUERY_LOGREG": FraudPredictor(model_name="baseline"),
        "XGBOOST_GPU": FraudPredictor(model_name="xgboost"),
        "FRAUDNET_MLP": FraudPredictor(model_name="fraudnet"),
    }

    for idx, (_, row) in enumerate(target_samples.iterrows(), 1):
        tx_row = row.to_dict()
        for model_name, predictor in predictors.items():
            pred_res = predictor.predict_transaction(tx_row)
            p0 = pred_res["predicted_class_probs"][0]["prob"]
            p1 = pred_res["predicted_class_probs"][1]["prob"]
            pred_cls = pred_res["predicted_class"]
            actual_cls = int(tx_row["Class"])
            status = "[green]CORRECT[/green]" if pred_cls == actual_cls else "[red]MISSED[/red]"

            pred_table.add_row(
                str(idx),
                model_name,
                f"${tx_row['Amount']:.2f}",
                f"{p0:.6f}",
                f"{p1:.6f}",
                str(pred_cls),
                str(actual_cls),
                status,
            )

    console.print(pred_table)

    total_time = time.time() - total_start
    console.print(Panel(
        f"[bold green]PIPELINE EXECUTION COMPLETE IN {total_time:.1f}s[/bold green]\n"
        f"Artifacts saved in: [cyan]data/[/cyan], [cyan]models_saved/[/cyan], and [cyan]reports/[/cyan]",
        border_style="green"
    ))

if __name__ == "__main__":
    main()
