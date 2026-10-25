"""GPU-Accelerated XGBoost Classifier with automated class imbalance handling."""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from src.config import MODELS_DIR, XGB_CONFIG, RANDOM_SEED
from src.utils.logger import logger

XGB_MODEL_PATH = MODELS_DIR / "xgboost_fraud_model.joblib"

class GPUXGBoostModel:
    """XGBoost Classifier configured for high-speed GPU execution and extreme class imbalance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or XGB_CONFIG.copy()
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_cols = []
        self.is_fitted = False
        self.device_used = "cpu"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_cols: list = None,
    ):
        """Fits XGBoost model with GPU acceleration and early stopping."""
        self.feature_cols = feature_cols or []
        
        # Calculate scale_pos_weight = neg / pos
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = float(neg_count / max(1, pos_count))
        logger.info(
            f"⚖️ Computed scale_pos_weight: {scale_pos_weight:.2f} "
            f"(Negatives: {neg_count:,}, Positives: {pos_count:,})"
        )

        params = self.config.copy()
        params["scale_pos_weight"] = scale_pos_weight

        # Attempt GPU training first; if GPU driver lacks specific CC kernels, seamlessly fall back to multi-core CPU
        try:
            logger.info("🚀 Attempting GPU-accelerated XGBoost training (tree_method='hist', device='cuda')...")
            params["device"] = "cuda"
            self.model = xgb.XGBClassifier(**params)
            
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
            self.device_used = "cuda"
            logger.info("🎉 XGBoost trained successfully on NVIDIA GPU!")
        except Exception as e:
            logger.warning(f"⚠️ GPU training failed ({e}). Falling back to multi-threaded CPU hist...")
            params["device"] = "cpu"
            params["n_jobs"] = -1
            self.model = xgb.XGBClassifier(**params)
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
            self.device_used = "cpu"
            logger.info("✅ XGBoost trained successfully on multi-threaded CPU.")

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns [P(Normal), P(Fraud)]."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary label given a probability decision threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """Returns feature importances sorted descending."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        importances = self.model.feature_importances_
        cols = self.feature_cols if len(self.feature_cols) == len(importances) else [f"feat_{i}" for i in range(len(importances))]
        df_imp = pd.DataFrame({"feature": cols, "importance": importances})
        return df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)

    def save(self, path: Path = XGB_MODEL_PATH):
        """Saves model to disk."""
        joblib.dump(self, path)
        logger.info(f"💾 Saved XGBoost model to {path}")

    @classmethod
    def load(cls, path: Path = XGB_MODEL_PATH) -> "GPUXGBoostModel":
        """Loads model from disk."""
        instance = joblib.load(path)
        logger.info(f"⚡ Loaded XGBoost model from {path}")
        return instance
