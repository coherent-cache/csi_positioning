"""Compile the trained Location CNN into an encrypted Concrete ML artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
from concrete.fhe import Configuration
from concrete.ml.common.serialization.dumpers import dump
from concrete.ml.torch.compile import compile_torch_model

from location_cnn.data import (
    load_feature_stats,
    load_mat_dataset,
    normalize_features,
    save_feature_stats,
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
    parser = argparse.ArgumentParser(
        description="Compile the CSI location CNN for FHE execution."
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
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to persist compiled artifacts and stats.",
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


def convert_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_keys_to_str(i) for i in obj]
    return obj


def main() -> int:
    args = parse_args()

    checkpoint_path = args.checkpoint_dir / f"location_cnn_{args.architecture}.pt"
    stats_path = args.artifacts_dir / f"feature_stats_{args.architecture}.json"
    quantized_module_path = (
        args.artifacts_dir / f"location_quantized_{args.architecture}.json"
    )
    debug_dir = args.artifacts_dir / f"location_quantized_debug_{args.architecture}"

    if not checkpoint_path.exists():
        print(
            f"Checkpoint missing: {checkpoint_path}. Run train_location_cnn.py first."
        )
        return 1

    if not args.dataset_file.exists():
        print("Dataset file missing; provide --dataset-file before compiling.")
        return 1

    calib_features, mean, std = load_or_compute_stats(args.dataset_file, stats_path)
    calib_features = calib_features[: args.calib_samples]
    if calib_features.shape[0] == 0:
        print("No calibration samples left after slicing.")
        return 1

    calib_tensor = torch.from_numpy(calib_features).float()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Determine architecture
    ckpt_arch = checkpoint.get("architecture", "optimized")
    if ckpt_arch != args.architecture:
        print(
            f"Warning: Checkpoint architecture {ckpt_arch} differs from requested {args.architecture}"
        )
        # We proceed with requested architecture, assuming the user knows what they are doing (or fail later)

    print(f"Loading model with architecture: {args.architecture}")
    model = LocationCNN(architecture=args.architecture)
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
    except Exception as exc:  # pragma: no cover
        print("Compilation failed:", exc)
        return 1

    quantized_module.check_model_is_compiled()

    quantized_module_path.parent.mkdir(parents=True, exist_ok=True)
    with quantized_module_path.open("w") as handle:
        dump(quantized_module, handle)

    debug_dir.mkdir(parents=True, exist_ok=True)
    graph_txt = quantized_module.fhe_circuit.graph.format(show_locations=True)
    (debug_dir / "circuit.graph.txt").write_text(graph_txt, encoding="utf-8")
    (debug_dir / "circuit.mlir.txt").write_text(
        quantized_module.fhe_circuit.mlir, encoding="utf-8"
    )

    with (debug_dir / "statistics.json").open("w", encoding="utf-8") as stats_handle:
        # Convert keys to strings recursively because JSON requires string keys
        statistics = convert_keys_to_str(quantized_module.fhe_circuit.statistics)
        json.dump(
            statistics,
            stats_handle,
            default=lambda value: str(value),
            indent=2,
        )

    print("FHE compilation succeeded.")
    print(f"Saved compiled module to {quantized_module_path}")
    print(f"Mean / std used: {mean:.5f}, {std:.5f}")
    print(f"Debug traces written to {debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
