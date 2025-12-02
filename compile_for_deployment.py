"""Compile the Location CNN model for deployment (FHE artifacts)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
from concrete.fhe import Configuration
from concrete.ml.deployment import FHEModelDev
from concrete.ml.torch.compile import compile_torch_model

from location_cnn.data import (
    load_feature_stats,
    load_mat_dataset,
    normalize_features,
    save_feature_stats,
)
from location_cnn.models import LocationCNN


DEFAULT_N_BITS = {
    "model_inputs": 6,
    "op_inputs": 6,
    "op_weights": 6,
    "model_outputs": 6,
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
    parser = argparse.ArgumentParser(
        description="Compile the CSI location CNN for FHE deployment."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing trained PyTorch checkpoints.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("dataset/dataset_SNR50_outdoor.mat"),
        help="Dataset used for calibration (default: SNR50 outdoor).",
    )
    parser.add_argument(
        "--deployment-dir",
        type=Path,
        default=Path("deployment_artifacts"),
        help="Directory to save deployment artifacts (client.zip, server.zip, etc.).",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing normalization stats.",
    )
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
        "--p-error",
        type=float,
        default=0.1,
        help="Probability of error per programmable bootstrap.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used by the Concrete compiler (cpu or cuda).",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="optimized",
        choices=["optimized", "original"],
        help="Model architecture to compile.",
    )
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()

    checkpoint_path = args.checkpoint_dir / f"location_cnn_{args.architecture}.pt"
    stats_path = args.stats_dir / f"feature_stats_{args.architecture}.json"
    deployment_path = args.deployment_dir / args.architecture

    if not checkpoint_path.exists():
        print(
            f"Checkpoint missing: {checkpoint_path}. Run train_location_cnn.py first."
        )
        return 1

    if not args.dataset_file.exists():
        print("Dataset file missing; provide --dataset-file before compiling.")
        return 1

    print("Loading calibration data...")
    calib_features, mean, std = load_or_compute_stats(args.dataset_file, stats_path)
    calib_features = calib_features[: args.calib_samples]
    if calib_features.shape[0] == 0:
        print("No calibration samples left after slicing.")
        return 1

    calib_tensor = torch.from_numpy(calib_features).float()

    print("Loading PyTorch model...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Determine architecture
    ckpt_arch = checkpoint.get("architecture", "optimized")
    if ckpt_arch != args.architecture:
        print(
            f"Warning: Checkpoint architecture ({ckpt_arch}) does not match requested ({args.architecture})."
        )

    print(f"Loading model with architecture: {args.architecture}")
    model = LocationCNN(architecture=args.architecture)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n_bits = args.n_bits if args.n_bits is not None else DEFAULT_N_BITS
    config = Configuration(enable_tlu_fusing=True, print_tlu_fusing=False)
    device = ensure_device(args.device)

    print("Compiling model for FHE...")
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
    except Exception as exc:
        print("Compilation failed:", exc)
        return 1

    print("Checking model compilation...")
    quantized_module.check_model_is_compiled()

    print(f"Saving deployment artifacts to {deployment_path}...")
    deployment_path.mkdir(parents=True, exist_ok=True)

    # Use FHEModelDev to save artifacts
    dev = FHEModelDev(path_dir=str(deployment_path), model=quantized_module)
    dev.save()

    print("FHE compilation and export succeeded.")
    print(f"Artifacts in: {deployment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
