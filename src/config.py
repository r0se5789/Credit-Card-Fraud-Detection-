"""Central configuration for Credit Card Fraud Detection ML Engine."""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models_saved"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure all directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Data paths & URL
RAW_CSV_NAME = "creditcard.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_CSV_NAME

# Public direct download mirror (Machine Learning Group - ULB European Cardholders dataset)
DATASET_DOWNLOAD_URLS = [
    "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv",
    "https://huggingface.co/datasets/JEFFREY-VERDIERE/Creditcard/resolve/main/creditcard.csv",
]

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Device Configuration
def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")

DEVICE = get_device()

# Dataset Schema Constants
TIME_COL = "Time"
AMOUNT_COL = "Amount"
TARGET_COL = "Class"
FEATURE_COLS = [TIME_COL] + [f"V{i}" for i in range(1, 29)] + [AMOUNT_COL]

# Split Configuration
# As in BigQuery ML specification: DATA_SPLIT_METHOD='seq', DATA_SPLIT_COL='Time'
SPLIT_METHOD = "seq"  # Options: 'seq' (chronological sequential split) or 'stratified'
TEST_SIZE = 0.20
VAL_SIZE = 0.10

# Cost-Sensitive Learning Parameters (Dal Pozzolo et al., CIDM 2015 / Trans. NNLS 2018)
# False Positive cost: administrative cost of investigating an alert or calling the customer ($10)
# False Negative cost: average or actual transaction amount stolen by the fraudster
ADMINISTRATIVE_INVESTIGATION_COST = 10.0

# Deep Learning Hyperparameters (FraudNet)
NN_CONFIG = {
    "batch_size": 2048,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 20,
    "early_stopping_patience": 5,
    "hidden_dims": [128, 64, 32],
    "dropout_rate": 0.25,
}

# Autoencoder Hyperparameters (Unsupervised Anomaly Detection)
AE_CONFIG = {
    "batch_size": 2048,
    "learning_rate": 2e-3,
    "epochs": 15,
    "hidden_dims": [64, 32, 16],
    "early_stopping_patience": 4,
}

# XGBoost GPU Hyperparameters
XGB_CONFIG = {
    "n_estimators": 350,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cuda",
    "eval_metric": "aucpr",
    "random_state": RANDOM_SEED,
}
