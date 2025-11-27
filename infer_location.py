"""Run inference with the trained Location CNN in clear or FHE modes."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

from location_cnn.data import (
    load_feature_stats,
    load_mat_dataset,
    normalize_features,
    split_dataset,
)
from location_cnn.models import LocationCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the Location CNN.")
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("dataset/dataset_SNR50_outdoor.mat"),
        help="The CSI dataset used for validation samples.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/location_cnn.pt"),
        help="Trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("artifacts/feature_stats.json"),
        help="JSON file with normalization stats.",
    )
    parser.add_argument(
        "--quantized-module-path",
        type=Path,
        default=Path("artifacts/location_quantized.pkl"),
        help="Optional compiled QuantizedModule for FHE runs.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of validation samples to display.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for sample selection.",
    )
    parser.add_argument(
        "--fhe-mode",
        choices=["disable", "simulate", "execute"],
        default="simulate",
        help="Mode passed to QuantizedModule.forward (if available).",
    )
    return parser.parse_args()


def load_quantized_module(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to load quantized module: {exc}")
        return None


def main() -> int:
    args = parse_args()

    if not args.checkpoint.exists():
        print("Checkpoint not found; run train_location_cnn.py first.")
        return 1

    if not args.dataset_file.exists():
        print("Dataset not found; provide --dataset-file.")
        return 1

    features, targets = load_mat_dataset(args.dataset_file)
    normalized_features, mean, std = normalize_features(
        features, *load_feature_stats(args.stats_path)
    )
    if normalized_features.size == 0:
        print("Dataset contains no samples after normalization.")
        return 1

    _, val_features, _, val_targets = split_dataset(
        normalized_features,
        targets,
        test_size=0.2,
        random_state=args.seed,
    )

    model = LocationCNN()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(
        len(val_features), size=min(args.num_samples, len(val_features)), replace=False
    )

    quantized_module = load_quantized_module(args.quantized_module_path)
    if quantized_module is not None:
        print(f"Loaded quantized module from {args.quantized_module_path}")

    for idx in indices:
        sample = val_features[idx : idx + 1]
        target = val_targets[idx]

        input_tensor = torch.from_numpy(sample).float()
        with torch.no_grad():
            prediction = model(input_tensor).numpy().reshape(3)

        print(
            f"Sample {idx}: target={target.tolist()} pred={prediction.tolist()}"
        )
        print(f"  ℓ1 error={mean_absolute_error(target, prediction):.4f}")

        if quantized_module is not None:
            fhe_pred = quantized_module.forward(sample, fhe=args.fhe_mode)
            if isinstance(fhe_pred, tuple):
                fhe_pred = np.array(fhe_pred[0])
            print(
                f"  {args.fhe_mode} (quantized) -> {np.array(fhe_pred).reshape(3).tolist()}"
            )

    print("Inference done.")
    print(f"Normalization stats: mean={mean:.5f}, std={std:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



