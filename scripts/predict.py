"""CLI tool for scoring transactions with the trained fraud detection models."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.engine.predictor import FraudPredictor
from src.config import RAW_DATA_PATH, TIME_COL, AMOUNT_COL, TARGET_COL

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Prediction CLI")
    parser.add_argument("--model", type=str, default="xgboost", choices=["baseline", "xgboost", "fraudnet", "autoencoder"],
                        help="Model architecture to use for prediction")
    parser.add_argument("--time", type=float, default=None,
                        help="Transaction timestamp (in seconds from dataset origin)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV containing transactions to score in batch")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of transactions to display for batch scoring")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for fraud probability")

    args = parser.parse_args()

    predictor = FraudPredictor(model_name=args.model)
    console.print(f"[bold cyan]Loaded Model:[/bold cyan] {args.model.upper()} (Threshold = {args.threshold})")

    if args.time is not None:
        # Search for transaction at that timestamp in raw dataset
        raw_df = pd.read_csv(RAW_DATA_PATH)
        matched = raw_df[raw_df[TIME_COL] == args.time]
        if len(matched) == 0:
            console.print(f"[yellow]No exact transaction found at Time = {args.time}. Taking closest match.[/yellow]")
            matched = raw_df.iloc[(raw_df[TIME_COL] - args.time).abs().argsort()[:1]]

        row = matched.iloc[0].to_dict()
        pred = predictor.predict_transaction(row, threshold=args.threshold)
        
        table = Table(title=f"Prediction for Time = {row[TIME_COL]}")
        table.add_column("Field", style="bold")
        table.add_column("Value")

        table.add_row("Amount", f"${row[AMOUNT_COL]:.2f}")
        table.add_row("P(Legitimate / 0)", f"{pred['predicted_class_probs'][0]['prob']:.6f}")
        table.add_row("P(Fraud / 1)", f"{pred['predicted_class_probs'][1]['prob']:.6f}")
        table.add_row("Predicted Label", "[bold red]FRAUD (1)[/bold red]" if pred["predicted_class"] == 1 else "[bold green]NORMAL (0)[/bold green]")
        table.add_row("Actual Ground Truth", str(int(row[TARGET_COL])))
        
        console.print(table)

    elif args.csv is not None:
        df = pd.read_csv(args.csv)
        results = predictor.predict_batch(df, threshold=args.threshold)
        console.print(f"[green]Scored {len(df)} transactions.[/green]")
        console.print(results[["Time", "Amount", "prob_fraud", "predicted_class"]].head(args.limit))
    else:
        # Default: test the famous Time = 85285.0 from the prompt
        raw_df = pd.read_csv(RAW_DATA_PATH)
        matched = raw_df[raw_df[TIME_COL] == 85285.0]
        if len(matched) > 0:
            row = matched.iloc[0].to_dict()
            pred = predictor.predict_transaction(row, threshold=args.threshold)
            console.print(pred)

if __name__ == "__main__":
    main()
