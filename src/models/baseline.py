"""Baseline Logistic Regression model replicating BigQuery ML configuration."""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

from src.config import MODELS_DIR, RANDOM_SEED, TARGET_COL
from src.utils.logger import logger

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_logistic_reg.joblib"

class BigQueryLogisticRegressionBaseline:
    """Logistic Regression model configured identically to BigQuery ML:
    - MODEL_TYPE = 'logistic_reg'
    - AUTO_CLASS_WEIGHTS = TRUE (class_weight='balanced')
    - Max iterations = 1000
    - Solver = 'lbfgs'
    """
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.model = LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
            C=1.0,
        )
        self.feature_cols = []
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_cols: list = None):
        """Fits the model on balanced weights."""
        logger.info("🧠 Training BigQuery ML Baseline (Logistic Regression with Balanced Class Weights)...")
        self.model.fit(X_train, y_train)
        self.feature_cols = feature_cols or []
        self.is_fitted = True
        logger.info("✅ Baseline Logistic Regression training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted probability for fraud [P(Class=0), P(Class=1)]."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary class given a probability decision threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """Returns absolute feature coefficients as importance."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        coefs = self.model.coef_[0]
        cols = self.feature_cols if len(self.feature_cols) == len(coefs) else [f"feat_{i}" for i in range(len(coefs))]
        df_imp = pd.DataFrame({"feature": cols, "coefficient": coefs, "abs_importance": np.abs(coefs)})
        return df_imp.sort_values(by="abs_importance", ascending=False).reset_index(drop=True)

    def save(self, path: Path = BASELINE_MODEL_PATH):
        """Saves model to disk."""
        joblib.dump(self, path)
        logger.info(f"💾 Saved Baseline model to {path}")

    @classmethod
    def load(cls, path: Path = BASELINE_MODEL_PATH) -> "BigQueryLogisticRegressionBaseline":
        """Loads model from disk."""
        instance = joblib.load(path)
        logger.info(f"⚡ Loaded Baseline model from {path}")
        return instance
