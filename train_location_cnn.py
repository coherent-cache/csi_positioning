"""Train a compact CNN on the 5G CSI dataset and save checkpoints + metrics."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import nn

from location_cnn.data import (
    load_mat_dataset,
    make_data_loader,
    normalize_features,
    save_feature_stats,
    split_dataset,
)
from location_cnn.models import LocationCNN


METRICS_FIELDS = [
    "epoch",
    "train_loss",
    "train_mae",
    "train_r2",
    "val_loss",
    "val_mae",
    "val_r2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CSI location CNN and persist checkpoints / metrics."
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("dataset/dataset_SNR50_outdoor.mat"),
        help="Path to the .mat dataset (defaults to SNR50 outdoor).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to store trained weights.",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to persist normalization stats.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("logs"),
        help="Directory to store training metrics.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (both train and validation).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples held out for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for shuffling / initialization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to train on (cpu / cuda).",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Ignore existing checkpoint and retrain from scratch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load the checkpoint and continue training (does not extend epoch count).",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="optimized",
        choices=["optimized", "original"],
        help="Model architecture to use ('optimized' or 'original').",
    )
    return parser.parse_args()


def write_metrics(metrics_path: Path, history: list[dict]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def main() -> int:
    args = parse_args()
    start_time = time.time()

    if not args.dataset_file.exists():
        print(
            "Dataset not found: use --dataset-file to point at a .mat file.",
            file=sys.stderr,
        )
        return 1

    # Construct architecture-specific paths
    checkpoint_path = args.checkpoint_dir / f"location_cnn_{args.architecture}.pt"
    stats_path = args.stats_dir / f"feature_stats_{args.architecture}.json"
    metrics_path = (
        args.metrics_dir / f"location_training_metrics_{args.architecture}.csv"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_exists = checkpoint_path.exists()
    if checkpoint_exists and not args.retrain and not args.resume:
        print(
            f"Checkpoint {checkpoint_path} already exists. "
            "Use --retrain to force a new run or --resume to continue training."
        )
        return 0

    features, targets = load_mat_dataset(args.dataset_file)
    normalized_features, mean, std = normalize_features(features)

    save_feature_stats(stats_path, mean, std)
    train_features, val_features, train_targets, val_targets = split_dataset(
        normalized_features,
        targets,
        test_size=args.test_size,
        random_state=args.seed,
    )

    train_loader = make_data_loader(
        train_features, train_targets, args.batch_size, shuffle=True
    )
    val_loader = make_data_loader(
        val_features, val_targets, args.batch_size, shuffle=False
    )

    preferred_device = args.device.lower()
    if preferred_device not in {"cpu", "cuda"}:
        preferred_device = "cpu"

    if preferred_device == "cuda" and not torch.cuda.is_available():
        print("GPU requested but not available; falling back to CPU.")
        preferred_device = "cpu"

    device = torch.device(preferred_device)

    print(f"Using architecture: {args.architecture}")
    model = LocationCNN(architecture=args.architecture)
    model.to(device)

    if args.resume and checkpoint_exists:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Verify architecture matches
        ckpt_arch = checkpoint.get("architecture", "optimized")
        if ckpt_arch != args.architecture:
            print(
                f"Error: Checkpoint architecture ({ckpt_arch}) does not match requested ({args.architecture})."
            )
            return 1

        model.load_state_dict(checkpoint["model_state_dict"])
        print("Resumed training from checkpoint.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        train_preds: list[torch.Tensor] = []
        train_targets_list: list[torch.Tensor] = []

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            train_preds.append(predictions.detach().cpu())
            train_targets_list.append(batch_targets.detach().cpu())

        model.eval()
        val_losses = []
        val_preds: list[torch.Tensor] = []
        val_targets_list: list[torch.Tensor] = []

        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                predictions = model(batch_inputs)
                val_losses.append(loss_fn(predictions, batch_targets).item())
                val_preds.append(predictions.cpu())
                val_targets_list.append(batch_targets.cpu())

        train_preds_np = torch.cat(train_preds).numpy()
        train_targets_np = torch.cat(train_targets_list).numpy()
        val_preds_np = torch.cat(val_preds).numpy()
        val_targets_np = torch.cat(val_targets_list).numpy()

        train_mae = float(
            mae_fn(torch.cat(train_preds), torch.cat(train_targets_list)).item()
        )
        val_mae = float(
            mae_fn(torch.cat(val_preds), torch.cat(val_targets_list)).item()
        )
        train_r2 = float(r2_score(train_targets_np, train_preds_np))
        val_r2 = float(r2_score(val_targets_np, val_preds_np))

        metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "train_mae": train_mae,
            "train_r2": train_r2,
            "val_loss": float(np.mean(val_losses)),
            "val_mae": val_mae,
            "val_r2": val_r2,
        }
        history.append(metrics)

        print(
            f"[Epoch {epoch}] train loss={metrics['train_loss']:.4f} "
            f"val loss={metrics['val_loss']:.4f} train r2={train_r2:.3f} val r2={val_r2:.3f}"
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "epochs": args.epochs,
            "architecture": args.architecture,
        },
        checkpoint_path,
    )

    write_metrics(metrics_path, history)

    duration = time.time() - start_time
    print(f"Training completed in {duration:.1f}s. Checkpoint: {checkpoint_path}")
    print(f"Metrics logged to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
