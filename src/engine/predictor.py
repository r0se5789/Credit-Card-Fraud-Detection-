"""Inference engine for real-time and batch fraud scoring."""

from typing import Dict, Any, Union, List, Optional
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path

from src.config import (
    MODELS_DIR,
    TIME_COL,
    AMOUNT_COL,
    TARGET_COL,
    DEVICE,
)
from src.data.preprocess import SCALER_SAVE_PATH
from src.models.baseline import BigQueryLogisticRegressionBaseline, BASELINE_MODEL_PATH
from src.models.xgboost_gpu import GPUXGBoostModel, XGB_MODEL_PATH
from src.models.neural_net import FraudNet, FRAUDNET_MODEL_PATH
from src.models.autoencoder import FraudAutoencoder, AUTOENCODER_MODEL_PATH
from src.utils.logger import logger

class FraudPredictor:
    """Production inference engine capable of scoring transactions with any or all trained models."""

    def __init__(self, model_name: str = "xgboost"):
        self.model_name = model_name
        self.scaler = None
        self.model = None
        self.device = DEVICE
        self.feature_cols = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
        self.load_artifacts()

    def load_artifacts(self):
        """Loads fitted scaler and the chosen model."""
        if not SCALER_SAVE_PATH.exists():
            raise FileNotFoundError(f"Scaler not found at {SCALER_SAVE_PATH}. Run pipeline training first.")
        self.scaler = joblib.load(SCALER_SAVE_PATH)

        if self.model_name == "baseline":
            self.model = BigQueryLogisticRegressionBaseline.load(BASELINE_MODEL_PATH)
        elif self.model_name == "xgboost":
            self.model = GPUXGBoostModel.load(XGB_MODEL_PATH)
        elif self.model_name == "fraudnet":
            self.model = FraudNet.load(FRAUDNET_MODEL_PATH, device=self.device)
        elif self.model_name == "autoencoder":
            self.model = FraudAutoencoder.load(AUTOENCODER_MODEL_PATH, device=self.device)
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

    def preprocess_raw_input(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """Transforms raw transaction dict or DataFrame to model-ready features:
        - SAFE.LOG(Amount) -> np.log1p(Amount)
        - RobustScaler on Time and log_amount
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Log transform amount
        log_amount = np.log1p(df[AMOUNT_COL].values)
        times = df[TIME_COL].values

        # Apply fitted scaler
        scaled = self.scaler.transform(np.column_stack([times, log_amount]))
        scaled_time = scaled[:, 0]
        scaled_amount = scaled[:, 1]

        # Extract V1..V28
        v_features = [f"V{i}" for i in range(1, 29)]
        X_v = df[v_features].values

        # Stack into feature matrix
        X = np.column_stack([X_v, scaled_amount, scaled_time])
        return X

    def predict_transaction(self, raw_tx: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
        """Scores a single transaction dictionary.
        
        Returns:
            Dict matching BigQuery ML PREDICT output structure.
        """
        X = self.preprocess_raw_input(raw_tx)
        
        if self.model_name in ["baseline", "xgboost"]:
            probs = self.model.predict_proba(X)[0]
            prob_normal = float(probs[0])
            prob_fraud = float(probs[1])
        elif self.model_name == "fraudnet":
            tensor_X = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                prob_fraud = float(self.model.predict_proba(tensor_X).cpu().numpy()[0][0])
                prob_normal = 1.0 - prob_fraud
        elif self.model_name == "autoencoder":
            tensor_X = torch.tensor(X, dtype=torch.float32).to(self.device)
            anomaly_score = float(self.model.compute_anomaly_scores(tensor_X)[0])
            prob_fraud = 1.0 if anomaly_score > self.model.threshold else 0.0
            prob_normal = 1.0 - prob_fraud
            return {
                "Amount": raw_tx.get(AMOUNT_COL),
                "anomaly_score": round(anomaly_score, 6),
                "threshold": round(self.model.threshold, 6),
                "predicted_class": int(anomaly_score > self.model.threshold),
                "Class": raw_tx.get(TARGET_COL, None),
            }

        predicted_class = 1 if prob_fraud >= threshold else 0

        # Exact structure from BigQuery ML output:
        # Amount, predicted_class_probs, Class
        return {
            "Amount": raw_tx.get(AMOUNT_COL),
            "predicted_class": predicted_class,
            "predicted_class_probs": [
                {"label": 0, "prob": round(prob_normal, 6)},
                {"label": 1, "prob": round(prob_fraud, 6)},
            ],
            "fraud_probability": round(prob_fraud, 6),
            "Class": raw_tx.get(TARGET_COL, None),
        }

    def predict_batch(self, df_raw: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Scores a batch of transactions and returns DataFrame with predictions."""
        X = self.preprocess_raw_input(df_raw)
        result_df = df_raw.copy()

        if self.model_name in ["baseline", "xgboost"]:
            probs = self.model.predict_proba(X)
            result_df["prob_normal"] = probs[:, 0]
            result_df["prob_fraud"] = probs[:, 1]
            result_df["predicted_class"] = (probs[:, 1] >= threshold).astype(int)
        elif self.model_name == "fraudnet":
            tensor_X = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                probs_fraud = self.model.predict_proba(tensor_X).cpu().numpy().flatten()
            result_df["prob_normal"] = 1.0 - probs_fraud
            result_df["prob_fraud"] = probs_fraud
            result_df["predicted_class"] = (probs_fraud >= threshold).astype(int)
        elif self.model_name == "autoencoder":
            tensor_X = torch.tensor(X, dtype=torch.float32).to(self.device)
            scores = self.model.compute_anomaly_scores(tensor_X)
            result_df["anomaly_score"] = scores
            result_df["predicted_class"] = (scores > self.model.threshold).astype(int)

        return result_df
