"""Deep Neural Network architecture (FraudNet) and Focal Loss for imbalanced tabular fraud detection."""

import os
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from src.config import MODELS_DIR, NN_CONFIG
from src.utils.logger import logger

FRAUDNET_MODEL_PATH = MODELS_DIR / "fraudnet_model.pt"

class FocalLoss(nn.Module):
    """Focal Loss for addressing extreme class imbalance:
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class FraudNet(nn.Module):
    """Multi-layer Tabular Perceptron with Batch Normalization, Dropout, and Skip Connections."""

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        hidden_dims = hidden_dims or NN_CONFIG["hidden_dims"]

        # First block
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.drop1 = nn.Dropout(dropout_rate)

        # Second block
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.drop2 = nn.Dropout(dropout_rate * 0.8)

        # Third block with projection for residual
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.drop3 = nn.Dropout(dropout_rate * 0.5)

        # Output head (logits)
        self.head = nn.Linear(hidden_dims[2], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.drop1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.drop2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.drop3(x)

        # Output logits
        logits = self.head(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns fraud probability via sigmoid."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def save(self, path: Path = FRAUDNET_MODEL_PATH):
        """Saves model weights."""
        torch.save(self.state_dict(), path)
        logger.info(f"💾 Saved FraudNet checkpoint to {path}")

    @classmethod
    def load(cls, path: Path = FRAUDNET_MODEL_PATH, input_dim: int = 30, device: torch.device = torch.device("cpu")) -> "FraudNet":
        """Loads model weights."""
        model = cls(input_dim=input_dim)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"⚡ Loaded FraudNet checkpoint from {path}")
        return model
