"""Data preprocessing, feature engineering, and temporal/stratified splitting."""

import json
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TIME_COL,
    AMOUNT_COL,
    TARGET_COL,
    SPLIT_METHOD,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_SEED,
)
from src.utils.logger import logger

SCALER_SAVE_PATH = MODELS_DIR / "scaler.joblib"
METADATA_SAVE_PATH = PROCESSED_DATA_DIR / "dataset_metadata.json"

def compute_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates comprehensive statistical distribution of transactions and frauds."""
    total_samples = len(df)
    fraud_samples = int((df[TARGET_COL] == 1).sum())
    normal_samples = int((df[TARGET_COL] == 0).sum())
    fraud_ratio = fraud_samples / total_samples

    stats = {
        "total_transactions": total_samples,
        "normal_transactions": normal_samples,
        "fraud_transactions": fraud_samples,
        "fraud_percentage": round(fraud_ratio * 100, 4),
        "imbalance_ratio": f"1:{normal_samples // fraud_samples}",
        "time_range_seconds": [float(df[TIME_COL].min()), float(df[TIME_COL].max())],
        "amount_summary": {
            "min": float(df[AMOUNT_COL].min()),
            "max": float(df[AMOUNT_COL].max()),
            "mean": round(float(df[AMOUNT_COL].mean()), 2),
            "median": round(float(df[AMOUNT_COL].median()), 2),
            "fraud_mean_amount": round(float(df[df[TARGET_COL] == 1][AMOUNT_COL].mean()), 2),
            "normal_mean_amount": round(float(df[df[TARGET_COL] == 0][AMOUNT_COL].mean()), 2),
        }
    }
    return stats

def preprocess_and_split(
    raw_path: Path = RAW_DATA_PATH,
    split_method: str = SPLIT_METHOD,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Loads raw dataset, transforms features (matching BigQuery ML specifications),
    and creates train/val/test splits.
    
    Returns:
        train_df, val_df, test_df, metadata
    """
    logger.info(f"📂 Loading dataset from {raw_path}...")
    df = pd.read_csv(raw_path)

    # Basic data integrity checks
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        logger.warning(f"Found {null_count} missing values. Imputing with column medians.")
        df = df.fillna(df.median())

    stats = compute_dataset_stats(df)
    logger.info(
        f"📊 Dataset Stats: {stats['total_transactions']:,} rows | "
        f"{stats['fraud_transactions']:,} Frauds ({stats['fraud_percentage']}%) | "
        f"Imbalance {stats['imbalance_ratio']}"
    )

    # BigQuery ML Transform equivalent:
    # TRANSFORM( * EXCEPT(Amount), SAFE.LOG(Amount) AS log_amount)
    # Using np.log1p (log(1 + amount)) to prevent -inf on zero-dollar authorization checks
    df["log_amount"] = np.log1p(df[AMOUNT_COL])

    # Splitting strategy
    if split_method == "seq":
        # Sequential split by Time (Dal Pozzolo et al., BigQuery DATA_SPLIT_METHOD='seq')
        # Prevents temporal lookahead bias in fraud transaction streams
        logger.info("⏱️ Performing Sequential (Temporal) Split based on 'Time' column...")
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * (1.0 - test_size - val_size))
        val_end = int(n * (1.0 - test_size))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
    else:
        # Stratified random split
        logger.info("🔀 Performing Stratified Random Split preserving fraud distribution...")
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df[TARGET_COL], random_state=RANDOM_SEED
        )
        val_relative = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_relative, stratify=train_val_df[TARGET_COL], random_state=RANDOM_SEED
        )

    # Feature Scaling:
    # V1..V28 are PCA components (already normalized).
    # We fit a RobustScaler on 'log_amount' and 'Time' using Train set only.
    scaler = RobustScaler()
    train_df[["scaled_time", "scaled_amount"]] = scaler.fit_transform(train_df[[TIME_COL, "log_amount"]])
    val_df[["scaled_time", "scaled_amount"]] = scaler.transform(val_df[[TIME_COL, "log_amount"]])
    test_df[["scaled_time", "scaled_amount"]] = scaler.transform(test_df[[TIME_COL, "log_amount"]])

    # Save fitted scaler
    joblib.dump(scaler, SCALER_SAVE_PATH)
    logger.info(f"💾 Saved feature scaler to {SCALER_SAVE_PATH}")

    # Feature column definitions
    v_features = [f"V{i}" for i in range(1, 29)]
    processed_feature_cols = v_features + ["scaled_amount", "scaled_time"]

    split_stats = {
        "split_method": split_method,
        "train": {
            "samples": len(train_df),
            "frauds": int(train_df[TARGET_COL].sum()),
            "fraud_rate": round(float(train_df[TARGET_COL].mean()) * 100, 4),
        },
        "val": {
            "samples": len(val_df),
            "frauds": int(val_df[TARGET_COL].sum()),
            "fraud_rate": round(float(val_df[TARGET_COL].mean()) * 100, 4),
        },
        "test": {
            "samples": len(test_df),
            "frauds": int(test_df[TARGET_COL].sum()),
            "fraud_rate": round(float(test_df[TARGET_COL].mean()) * 100, 4),
        },
        "feature_cols": processed_feature_cols,
    }

    logger.info(
        f"✂️ Splits created: Train={split_stats['train']['samples']:,} ({split_stats['train']['frauds']} frauds) | "
        f"Val={split_stats['val']['samples']:,} ({split_stats['val']['frauds']} frauds) | "
        f"Test={split_stats['test']['samples']:,} ({split_stats['test']['frauds']} frauds)"
    )

    # Save processed splits and metadata
    train_df.to_parquet(PROCESSED_DATA_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DATA_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DATA_DIR / "test.parquet", index=False)

    metadata = {**stats, "split_summary": split_stats}
    with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return train_df, val_df, test_df, metadata

def load_processed_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Loads preprocessed datasets from disk, or generates them if not found."""
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    val_path = PROCESSED_DATA_DIR / "val.parquet"
    test_path = PROCESSED_DATA_DIR / "test.parquet"

    if train_path.exists() and val_path.exists() and test_path.exists() and METADATA_SAVE_PATH.exists():
        logger.info("⚡ Loading preprocessed parquet splits from disk...")
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)
        with open(METADATA_SAVE_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return train_df, val_df, test_df, metadata

    return preprocess_and_split()

if __name__ == "__main__":
    preprocess_and_split()
