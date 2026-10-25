"""Training routines for deep neural networks and autoencoders."""

import time
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.neural_net import FraudNet, FocalLoss, FRAUDNET_MODEL_PATH
from src.models.autoencoder import FraudAutoencoder, AUTOENCODER_MODEL_PATH
from src.config import DEVICE, NN_CONFIG, AE_CONFIG
from src.utils.logger import logger
from src.utils.gpu_utils import clear_gpu_cache

def train_fraudnet(
    model: FraudNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device = DEVICE,
    epochs: int = NN_CONFIG["epochs"],
    lr: float = NN_CONFIG["learning_rate"],
    patience: int = NN_CONFIG["early_stopping_patience"],
) -> Dict[str, List[float]]:
    """Trains FraudNet with Focal Loss, learning rate scheduling, and early stopping.
    
    Returns:
        history dictionary containing loss and metric curves.
    """
    logger.info(f"🚀 Training FraudNet on device: {device} for {epochs} epochs...")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=NN_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    best_val_pr_auc = -1.0
    best_weights = None
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pr_auc": [],
        "val_roc_auc": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss_accum = 0.0
        train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss_accum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_accum / max(1, train_batches)

        # Validation Phase
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                probs = torch.sigmoid(logits)

                val_loss_accum += loss.item()
                val_batches += 1

                val_preds.extend(probs.cpu().numpy().flatten())
                val_targets.extend(y_batch.cpu().numpy().flatten())

        avg_val_loss = val_loss_accum / max(1, val_batches)
        val_preds_arr = np.array(val_preds)
        val_targets_arr = np.array(val_targets)

        # In case validation set has only 1 class or extreme split
        try:
            val_pr_auc = average_precision_score(val_targets_arr, val_preds_arr)
            val_roc_auc = roc_auc_score(val_targets_arr, val_preds_arr)
        except Exception:
            val_pr_auc = 0.0
            val_roc_auc = 0.5

        scheduler.step(val_pr_auc)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["val_roc_auc"].append(val_roc_auc)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.5f} | "
            f"Val Loss: {avg_val_loss:.5f} | "
            f"Val PR-AUC: {val_pr_auc:.4f} | "
            f"Val ROC-AUC: {val_roc_auc:.4f}"
        )

        # Early Stopping check based on PR-AUC
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_weights = model.state_dict().copy()
            epochs_no_improve = 0
            model.save(FRAUDNET_MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"⏹️ Early stopping triggered at epoch {epoch}. Restoring best checkpoint.")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    elapsed = time.time() - start_time
    logger.info(f"✨ FraudNet training finished in {elapsed:.1f}s. Best Val PR-AUC: {best_val_pr_auc:.4f}")
    clear_gpu_cache()
    return history

def train_autoencoder(
    model: FraudAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device = DEVICE,
    epochs: int = AE_CONFIG["epochs"],
    lr: float = AE_CONFIG["learning_rate"],
    patience: int = AE_CONFIG["early_stopping_patience"],
) -> Dict[str, List[float]]:
    """Trains Deep Autoencoder on normal non-fraud transactions for anomaly detection."""
    logger.info(f"🔍 Training Autoencoder on device: {device} for {epochs} epochs...")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_batches = 0

        for batch in train_loader:
            X_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            X_batch = X_batch.to(device)

            optimizer.zero_grad()
            reconstructed = model(X_batch)
            loss = criterion(reconstructed, X_batch)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_accum / max(1, train_batches)

        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                X_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
                X_batch = X_batch.to(device)
                reconstructed = model(X_batch)
                loss = criterion(reconstructed, X_batch)
                val_loss_accum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_accum / max(1, val_batches)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        logger.info(f"AE Epoch {epoch:02d}/{epochs:02d} | Train MSE: {avg_train_loss:.5f} | Val MSE: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = model.state_dict().copy()
            epochs_no_improve = 0
            model.save(AUTOENCODER_MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"⏹️ AE Early stopping at epoch {epoch}.")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    clear_gpu_cache()
    return history
