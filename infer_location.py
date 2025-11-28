"""Run inference with the trained Location CNN in clear or FHE modes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from concrete.fhe import Configuration
from concrete.ml.common.serialization.loaders import load
from concrete.ml.torch.compile import compile_torch_model
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from location_cnn.data import (
    load_feature_stats,
    load_mat_dataset,
    normalize_features,
    save_feature_stats,
    split_dataset,
)
from location_cnn.models import LocationCNN

DEFAULT_N_BITS = {
    "model_inputs": 8,
    "op_inputs": 7,
    "op_weights": 7,
    "model_outputs": 9,
}


def parse_n_bits(raw: str) -> Union[int, dict]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "n-bits must be an integer or JSON-serializable dict"
            ) from exc


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
        default=Path("artifacts/location_quantized.json"),
        help="Optional compiled QuantizedModule for FHE runs.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of validation samples to display in the 3D scatter plot.",
    )
    parser.add_argument(
        "--max-inference-samples",
        type=int,
        default=None,
        help="Optional cap on how many validation samples to run inference on (random subset).",
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
    # Compilation arguments
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=256,
        help="Number of calibration samples used by Concrete ML.",
    )
    parser.add_argument(
        "--n-bits",
        type=parse_n_bits,
        default=None,
        help="Quantization bitwidths as integer or JSON dict (default = %(default)s).",
    )
    parser.add_argument(
        "--rounding-threshold-bits",
        type=int,
        default=7,
        help="Rounding thresholds used during compilation (None disables).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("plots/inference_scatter.png"),
        help="Where to save the 3D scatter plot comparing targets/predictions/FHE outputs.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=150,
        help="Image resolution for the saved inference plot.",
    )
    parser.add_argument(
        "--p-error",
        type=float,
        default=0.05,
        help="Probability of error per programmable bootstrap.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used by the Concrete compiler (cpu or cuda).",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("artifacts/inference_results.csv"),
        help="Path where the inference results CSV will be written.",
    )
    return parser.parse_args()


def load_quantized_module(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r") as handle:
            return load(handle)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to load quantized module: {exc}")
        return None


def ensure_device(device_name: str) -> str:
    normalized = device_name.lower()
    if normalized not in {"cpu", "cuda"}:
        return "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return normalized


def load_or_compute_stats(
    dataset_path: Path, stats_path: Path
) -> tuple[np.ndarray, float, float]:
    features, _ = load_mat_dataset(dataset_path)
    mean, std = load_feature_stats(stats_path)
    stored_mean, stored_std = mean, std
    if mean is None or std is None:
        _, stored_mean, stored_std = normalize_features(features)
        save_feature_stats(stats_path, stored_mean, stored_std)
    normalized_features, _, _ = normalize_features(features, stored_mean, stored_std)
    return normalized_features, stored_mean, stored_std


def compile_model_on_the_fly(args):
    """Compile the model on the fly for FHE execution."""
    print("Compiling model on the fly for FHE execution...")

    calib_features, _, _ = load_or_compute_stats(args.dataset_file, args.stats_path)
    calib_features = calib_features[: args.calib_samples]
    if calib_features.shape[0] == 0:
        print("No calibration samples left after slicing.")
        return None

    calib_tensor = torch.from_numpy(calib_features).float()

    model = LocationCNN()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n_bits = args.n_bits if args.n_bits is not None else DEFAULT_N_BITS
    config = Configuration(enable_tlu_fusing=True, print_tlu_fusing=False)
    device = ensure_device(args.device)

    try:
        quantized_module = compile_torch_model(
            model,
            calib_tensor,
            configuration=config,
            n_bits=n_bits,
            rounding_threshold_bits=args.rounding_threshold_bits,
            p_error=args.p_error,
            device=device,
        )
        return quantized_module
    except Exception as exc:  # pragma: no cover
        print("Compilation failed:", exc)
        return None


def plot_coordinate_comparison(
    targets: np.ndarray,
    predictions: np.ndarray,
    fhe_predictions: np.ndarray | None,
    output_path: Path,
    dpi: int,
    fhe_mode: str,
) -> None:
    """Render a 3D scatter plot comparing clear and FHE coordinates."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        targets[:, 0],
        targets[:, 1],
        targets[:, 2],
        c="black",
        marker="x",
        s=60,
        label="Target",
    )

    ax.scatter(
        predictions[:, 0],
        predictions[:, 1],
        predictions[:, 2],
        c="tab:blue",
        marker="o",
        s=50,
        label="Prediction",
    )

    if fhe_predictions is not None and fhe_predictions.size > 0:
        ax.scatter(
            fhe_predictions[:, 0],
            fhe_predictions[:, 1],
            fhe_predictions[:, 2],
            c="tab:red",
            marker="^",
            s=50,
            label=f"FHE ({fhe_mode})",
        )

    for tgt, pred in zip(targets, predictions):
        ax.plot(
            [tgt[0], pred[0]],
            [tgt[1], pred[1]],
            [tgt[2], pred[2]],
            color="gray",
            linestyle="--",
            alpha=0.4,
        )

    if fhe_predictions is not None and fhe_predictions.size == len(targets):
        for tgt, fhe_pred in zip(targets, fhe_predictions):
            ax.plot(
                [tgt[0], fhe_pred[0]],
                [tgt[1], fhe_pred[1]],
                [tgt[2], fhe_pred[2]],
                color="tab:red",
                linestyle=":",
                alpha=0.5,
            )

    ax.set_title("UE Coordinate Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved inference plot to {output_path}")


def compute_error_metrics(
    ground_truth: np.ndarray, predictions: np.ndarray
) -> tuple[float, float]:
    distances = np.linalg.norm(predictions - ground_truth, axis=1)
    return float(distances.mean()), float(distances.max())


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

    # Load quantized module (if available) or compile on the fly if needed for FHE
    quantized_module = None
    if args.fhe_mode in ("simulate", "execute"):
        # Try compiling on the fly first because loaded module won't have circuit
        quantized_module = compile_model_on_the_fly(args)
        if quantized_module is None:
            print(
                "On-the-fly compilation failed. Falling back to loading from disk (circuit might be missing)."
            )
            quantized_module = load_quantized_module(args.quantized_module_path)
    else:
        quantized_module = load_quantized_module(args.quantized_module_path)

    if quantized_module is not None:
        if args.fhe_mode in ("simulate", "execute"):
            print("Using on-the-fly compiled module for FHE.")
        else:
            print(f"Loaded quantized module from {args.quantized_module_path}")

        # Check if FHE circuit is available for simulation/execution
    if (
        args.fhe_mode in ("simulate", "execute")
        and quantized_module.fhe_circuit is None
    ):
        print(
            f"Warning: FHE circuit not found in loaded module. Switching to 'disable' (clear quantized) mode."
        )
        args.fhe_mode = "disable"

    if args.fhe_mode == "execute" and quantized_module is not None:
        print("Generating FHE keys...")
        t_keygen = time.time()
        quantized_module.fhe_circuit.keygen()
        print(f"Key generation complete in {time.time() - t_keygen:.2f}s.")

    rng = np.random.default_rng(args.seed)
    if args.max_inference_samples is not None:
        subset_size = max(0, min(args.max_inference_samples, len(val_features)))
        indices = rng.choice(len(val_features), size=subset_size, replace=False)
    else:
        indices = np.arange(len(val_features))

    print(f"Running inference on {len(indices)} validation samples...")
    collected_targets: list[np.ndarray] = []
    collected_predictions: list[np.ndarray] = []
    collected_fhe_predictions: list[np.ndarray] = []
    plain_durations: list[float] = []
    fhe_durations: list[float] = []

    fhe_available = quantized_module is not None and args.fhe_mode != "disable"

    for idx in tqdm(indices, desc="Inference Progress", unit="sample"):
        sample = val_features[idx : idx + 1]
        target = val_targets[idx]

        input_tensor = torch.from_numpy(sample).float()
        t_plain_start = time.time()
        with torch.no_grad():
            prediction = model(input_tensor).numpy().reshape(3)
        t_plain_end = time.time()

        tqdm.write(f"Sample {idx}: target={target.tolist()} pred={prediction.tolist()}")
        tqdm.write(f"  ℓ1 error={mean_absolute_error(target, prediction):.4f}")

        collected_targets.append(target)
        collected_predictions.append(prediction)
        plain_durations.append(t_plain_end - t_plain_start)

        if fhe_available:
            t_start = time.time()
            fhe_pred = quantized_module.forward(sample, fhe=args.fhe_mode)
            t_end = time.time()
            if isinstance(fhe_pred, tuple):
                fhe_pred = np.array(fhe_pred[0])
            fhe_array = np.array(fhe_pred).reshape(3)
            tqdm.write(f"  {args.fhe_mode} (quantized) -> {fhe_array.tolist()}")
            if args.fhe_mode == "execute":
                tqdm.write(f"  FHE inference took {t_end - t_start:.2f}s")

            collected_fhe_predictions.append(fhe_array)
            fhe_durations.append(t_end - t_start)

    print("Inference done.")
    print(f"Normalization stats: mean={mean:.5f}, std={std:.5f}")

    if not collected_targets or not collected_predictions:
        return 1

    ground_truth = np.stack(collected_targets)
    pred_plain = np.stack(collected_predictions)
    if collected_fhe_predictions:
        pred_fhe = np.stack(collected_fhe_predictions)
    else:
        pred_fhe = np.full_like(pred_plain, np.nan)

    # Compute metrics
    plain_mpe, plain_max = compute_error_metrics(ground_truth, pred_plain)
    plain_time = float(np.mean(plain_durations)) if plain_durations else float("nan")

    if not np.isnan(pred_fhe).all():
        fhe_mpe, fhe_max = compute_error_metrics(ground_truth, pred_fhe)
        fhe_time = float(np.mean(fhe_durations)) if fhe_durations else float("nan")
    else:
        fhe_mpe = fhe_max = fhe_time = float("nan")

    header_fields = [
        "gt_x",
        "gt_y",
        "gt_z",
        "pred_plain_x",
        "pred_plain_y",
        "pred_plain_z",
        "pred_fhe_x",
        "pred_fhe_y",
        "pred_fhe_z",
    ]
    header = ",".join(header_fields)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    combined = np.hstack((ground_truth, pred_plain, pred_fhe))
    np.savetxt(args.results_csv, combined, delimiter=",", header=header, comments="")
    print(f"Saved inference results to {args.results_csv}")

    # Print comparison table
    print("\n=== Performance Comparison ===")
    print("Metric               | Plain          | FHE")
    print("---------------------+----------------+----------------")
    print(f"Mean Position Error  | {plain_mpe:14.4f} | {fhe_mpe:14.4f}")
    print(f"Max Error            | {plain_max:14.4f} | {fhe_max:14.4f}")
    print(f"Avg time per sample  | {plain_time:14.4f}s | {fhe_time:14.4f}s")

    # Limit scatter plot to requested number of samples to keep it readable
    scatter_count = min(args.num_samples, len(ground_truth))
    plot_coordinate_comparison(
        targets=ground_truth[:scatter_count],
        predictions=pred_plain[:scatter_count],
        fhe_predictions=pred_fhe[:scatter_count]
        if not np.isnan(pred_fhe).all()
        else None,
        output_path=args.plot_path,
        dpi=args.plot_dpi,
        fhe_mode=args.fhe_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
