"""Visualize and save the training loss + accuracy curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics saved to CSV.")
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("logs/location_training_metrics.csv"),
        help="CSV file produced by train_location_cnn.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where PNGs are written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image resolution for the saved charts.",
    )
    return parser.parse_args()


def read_metrics(metrics_path: Path) -> list[dict[str, float]]:
    metrics = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metrics.append({key: float(value) for key, value in row.items()})
    return metrics


def plot_curve(
    epochs: list[float],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, values, marker="o", label=title)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def main() -> int:
    args = parse_args()

    if not args.metrics_path.exists():
        print("Metrics CSV not found. Run train_location_cnn.py first.")
        return 1

    history = read_metrics(args.metrics_path)
    if not history:
        print("Metrics CSV is empty.")
        return 1

    epochs = [entry["epoch"] for entry in history]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_output = args.output_dir / "training_loss.png"
    r2_output = args.output_dir / "training_r2.png"

    plot_curve(
        epochs,
        [entry["train_loss"] for entry in history],
        "Training Loss",
        "Loss",
        loss_output,
        args.dpi,
    )

    plot_curve(
        epochs,
        [entry["val_loss"] for entry in history],
        "Validation Loss",
        "Loss",
        args.output_dir / "validation_loss.png",
        args.dpi,
    )

    plot_curve(
        epochs,
        [entry["train_r2"] for entry in history],
        "Training R²",
        "R²",
        r2_output,
        args.dpi,
    )

    plot_curve(
        epochs,
        [entry["val_r2"] for entry in history],
        "Validation R²",
        "R²",
        args.output_dir / "validation_r2.png",
        args.dpi,
    )

    print(f"Saved loss charts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



