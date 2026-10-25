"""PyTorch Dataset and DataLoader wrappers for high-performance GPU tensor batching."""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import TARGET_COL, NN_CONFIG

class FraudDataset(Dataset):
    """Custom PyTorch Dataset for tabular credit card transactions."""
    
    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        self.features = torch.tensor(features, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    batch_size: int = NN_CONFIG["batch_size"],
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds optimized PyTorch DataLoaders for Train, Validation, and Test partitions."""
    
    # Train
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    train_dataset = FraudDataset(X_train, y_train)

    # Val
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values
    val_dataset = FraudDataset(X_val, y_val)

    # Test
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COL].values
    test_dataset = FraudDataset(X_test, y_test)

    # DataLoaders (on Windows, num_workers=0 avoids spawn overhead; pin_memory speeds up CUDA transfer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
