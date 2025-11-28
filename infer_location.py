"""Run inference with the trained Location CNN in clear or FHE modes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Union

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

    print(f"Running inference on {len(indices)} samples...")
    for idx in tqdm(indices, desc="Inference Progress", unit="sample"):
        sample = val_features[idx : idx + 1]
        target = val_targets[idx]

        input_tensor = torch.from_numpy(sample).float()
        with torch.no_grad():
            prediction = model(input_tensor).numpy().reshape(3)

        # Print details only if not using tqdm to avoid messing up the bar, or use tqdm.write
        tqdm.write(f"Sample {idx}: target={target.tolist()} pred={prediction.tolist()}")
        tqdm.write(f"  ℓ1 error={mean_absolute_error(target, prediction):.4f}")

        if quantized_module is not None:
            t_start = time.time()
            fhe_pred = quantized_module.forward(sample, fhe=args.fhe_mode)
            t_end = time.time()
            if isinstance(fhe_pred, tuple):
                fhe_pred = np.array(fhe_pred[0])
            tqdm.write(
                f"  {args.fhe_mode} (quantized) -> {np.array(fhe_pred).reshape(3).tolist()}"
            )
            if args.fhe_mode == "execute":
                tqdm.write(f"  FHE inference took {t_end - t_start:.2f}s")

    print("Inference done.")
    print(f"Normalization stats: mean={mean:.5f}, std={std:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
