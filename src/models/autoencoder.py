"""Semi-Supervised Deep Autoencoder for Anomaly Detection in Transaction Streams."""

from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from src.config import MODELS_DIR, AE_CONFIG
from src.utils.logger import logger

AUTOENCODER_MODEL_PATH = MODELS_DIR / "autoencoder_model.pt"

class FraudAutoencoder(nn.Module):
    """Deep Tabular Autoencoder for unsupervised/semi-supervised anomaly detection.
    Trained strictly on non-fraudulent normal transactions to learn standard behaviors.
    """

    def __init__(self, input_dim: int = 30, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        hidden_dims = hidden_dims or AE_CONFIG["hidden_dims"]

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[2]),  # Bottleneck latent space
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[0], input_dim),
        )

        self.threshold = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def compute_anomaly_scores(self, x: torch.Tensor) -> np.ndarray:
        """Computes Mean Squared Reconstruction Error per sample."""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse_errors = torch.mean((x - reconstructed) ** 2, dim=1)
            return mse_errors.cpu().numpy()

    def set_threshold(self, normal_scores: np.ndarray, percentile: float = 98.5):
        """Sets anomaly threshold based on normal training score distribution."""
        self.threshold = float(np.percentile(normal_scores, percentile))
        logger.info(f"🎯 Set Autoencoder Anomaly Threshold to {self.threshold:.5f} ({percentile}th percentile)")

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Flags transactions whose reconstruction error exceeds threshold."""
        scores = self.compute_anomaly_scores(x)
        return (scores > self.threshold).astype(int)

    def save(self, path: Path = AUTOENCODER_MODEL_PATH):
        """Saves model weights and threshold."""
        state = {
            "state_dict": self.state_dict(),
            "threshold": self.threshold,
        }
        torch.save(state, path)
        logger.info(f"💾 Saved Autoencoder checkpoint to {path}")

    @classmethod
    def load(cls, path: Path = AUTOENCODER_MODEL_PATH, input_dim: int = 30, device: torch.device = torch.device("cpu")) -> "FraudAutoencoder":
        """Loads model weights and threshold."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(input_dim=input_dim)
        model.load_state_dict(checkpoint["state_dict"])
        model.threshold = checkpoint.get("threshold", 0.0)
        model.to(device)
        model.eval()
        logger.info(f"⚡ Loaded Autoencoder checkpoint from {path} (threshold={model.threshold:.5f})")
        return model
