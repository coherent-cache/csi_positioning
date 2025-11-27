"""Data helpers for the Location CNN workflow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class CSIDataset(Dataset):
    """Simple wrapper for torch DataLoader consumption."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def load_mat_dataset(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a CSI dataset exported as .mat file (features + positions)."""

    with h5py.File(file_path, "r") as storage:
        features = np.array(storage["features"])
        positions = np.array(storage["labels"]["position"])

    return features.astype(np.float32), positions.astype(np.float32)


def normalize_features(
    features: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """Normalize CSI features and optionally reuse an existing mean / std."""

    features = features.astype(np.float32)

    if mean is None:
        mean = float(np.mean(features))

    if std is None:
        std = float(np.std(features))

    if std == 0.0:
        std = 1.0

    normalized = (features - mean) / (std + 1e-9)
    return normalized, mean, std


def save_feature_stats(path: Path, mean: float, std: float) -> None:
    """Persist the normalization stats so downstream scripts can reuse them."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, float] = {"mean": float(mean), "std": float(std)}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def load_feature_stats(path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Return the previously stored normalization statistics if available."""

    if not path.exists():
        return None, None

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    return float(payload.get("mean")), float(payload.get("std"))


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wrap sklearn's split logic for consistent train / test partitions."""

    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )


def make_data_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """Utility for constructing DataLoaders from raw numpy arrays."""

    dataset = CSIDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




