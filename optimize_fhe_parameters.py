"""
Script to find optimal FHE parameters for the Location CNN model.
Explores the trade-off between FHE execution time (performance) and model accuracy.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from concrete.fhe import Configuration
from concrete.ml.torch.compile import compile_torch_model

from location_cnn.data import (
    load_mat_dataset,
    load_feature_stats,
    normalize_features,
    save_feature_stats,
)
from location_cnn.models import LocationCNN

# Define the search space for quantization bits
# We test a range of uniform bit-widths and the default mixed precision
N_BITS_CONFIGURATIONS = {
    "2-bit": 2,
    "4-bit": 4,
    "6-bit": 6,
    "8-bit": 8,
    "10-bit": 10,
    "12-bit": 12,
    "Default (Mixed)": {
        "model_inputs": 8,
        "op_inputs": 7,
        "op_weights": 7,
        "model_outputs": 9,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize FHE parameters for Location CNN."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/location_cnn.pt"),
        help="Path to the trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("dataset/dataset_SNR50_outdoor.mat"),
        help="Path to the dataset for calibration and evaluation.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("artifacts/feature_stats.json"),
        help="Path to feature normalization stats.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/fhe_optimization_results.json"),
        help="Where to save the optimization results.",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=256,
        help="Number of samples for calibration.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=100,
        help="Number of samples for accuracy evaluation (simulation).",
    )
    parser.add_argument(
        "--perf-samples",
        type=int,
        default=1,
        help="Number of samples for performance timing (actual FHE execution). Set to 0 to skip.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for compilation (cpu/cuda).",
    )
    return parser.parse_args()


def load_data(args):
    """Load and normalize data for calibration and testing."""
    print(f"Loading dataset from {args.dataset_file}...")
    features, targets = load_mat_dataset(args.dataset_file)

    # Load or compute stats
    mean, std = load_feature_stats(args.stats_path)
    if mean is None or std is None:
        _, mean, std = normalize_features(features)
        save_feature_stats(args.stats_path, mean, std)

    normalized_features, _, _ = normalize_features(features, mean, std)

    # Split into calibration set and evaluation set
    # We just take the first N for calibration, and the next M for evaluation
    # in a real scenario, ensure these don't overlap with training data if possible,
    # but for parameter optimization, using a hold-out set is fine.

    calib_data = normalized_features[: args.calib_samples]

    eval_start = args.calib_samples
    eval_end = eval_start + args.eval_samples
    eval_data = normalized_features[eval_start:eval_end]
    eval_targets = targets[eval_start:eval_end]

    return calib_data, eval_data, eval_targets


def evaluate_configuration(
    name: str,
    n_bits: Union[int, Dict],
    model: torch.nn.Module,
    calib_data: np.ndarray,
    eval_data: np.ndarray,
    eval_targets: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Compile and evaluate a single configuration."""
    print(f"\n--- Evaluating Configuration: {name} ---")
    print(f"n_bits: {n_bits}")

    calib_tensor = torch.from_numpy(calib_data).float()
    config = Configuration(enable_tlu_fusing=True, print_tlu_fusing=False)

    # 1. Compile
    start_compile = time.time()
    try:
        quantized_module = compile_torch_model(
            model,
            calib_tensor,
            configuration=config,
            n_bits=n_bits,
            rounding_threshold_bits=7,  # Consistent with compile_fhe_location.py
            p_error=0.05,  # Standard probability of error
            device=args.device,
        )
    except Exception as e:
        print(f"Compilation failed for {name}: {e}")
        return {"name": name, "error": str(e), "success": False}
    compile_time = time.time() - start_compile
    print(f"Compilation successful ({compile_time:.2f}s)")

    # 2. Evaluate Accuracy (Simulation)
    print(f"Running simulation on {len(eval_data)} samples...")
    start_sim = time.time()
    # simulate() handles a batch
    predicted_targets = quantized_module.fhe_circuit.simulate(eval_data)
    sim_time = time.time() - start_sim

    # Calculate metrics (MSE)
    mse = np.mean((predicted_targets - eval_targets) ** 2)
    mae = np.mean(np.abs(predicted_targets - eval_targets))
    print(f"Accuracy: MSE={mse:.6f}, MAE={mae:.6f}")

    # 3. Evaluate Performance (FHE Execution)
    fhe_exec_time = None
    if args.perf_samples > 0:
        print(f"Running actual FHE execution on {args.perf_samples} sample(s)...")
        # We run one by one for timing
        times = []
        # Use a subset of eval data for performance testing
        perf_data = eval_data[: args.perf_samples]

        for i in range(args.perf_samples):
            single_input = perf_data[i : i + 1]
            start_fhe = time.time()
            # encryption + execution + decryption
            quantized_module.fhe_circuit.encrypt_run_decrypt(single_input)
            duration = time.time() - start_fhe
            times.append(duration)
            print(f"  Sample {i+1}: {duration:.4f}s")

        fhe_exec_time = np.mean(times)
        print(f"Average FHE execution time: {fhe_exec_time:.4f}s")
    else:
        print("Skipping FHE execution timing.")

    # Circuit complexity stats
    circuit_stats = quantized_module.fhe_circuit.statistics

    return {
        "name": name,
        "n_bits": n_bits,
        "success": True,
        "compile_time": compile_time,
        "mse": mse,
        "mae": mae,
        "fhe_exec_time": fhe_exec_time,
        "circuit_size_of_inputs": circuit_stats.get("size_of_inputs"),
        "circuit_size_of_outputs": circuit_stats.get("size_of_outputs"),
        # "circuit_bootstrap_keys": circuit_stats.get("bootstrap_keys"), # Keys vary by version
    }


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return 1
    if not args.dataset_file.exists():
        print(f"Error: Dataset not found at {args.dataset_file}")
        return 1

    # Load Data
    calib_data, eval_data, eval_targets = load_data(args)

    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = LocationCNN()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = []

    # Loop through configurations
    for name, n_bits in N_BITS_CONFIGURATIONS.items():
        res = evaluate_configuration(
            name, n_bits, model, calib_data, eval_data, eval_targets, args
        )
        results.append(res)

    # Save results
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Helper to serialize results
    def json_default(obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)

    with args.output_path.open("w") as f:
        json.dump(results, f, indent=2, default=json_default)

    print(f"\nSaved optimization results to {args.output_path}")

    # Summary Table
    print("\n" + "=" * 80)
    print(
        f"{'Config':<20} | {'MSE':<10} | {'MAE':<10} | {'Time (s)':<10} | {'Success':<8}"
    )
    print("-" * 80)

    best_config = None
    min_score = float(
        "inf"
    )  # Simple score: MSE * Time (lower is better, assuming balanced weight)

    for r in results:
        if not r["success"]:
            print(f"{r['name']:<20} | {'FAILED':<10} | {'-':<10} | {'-':<10} | False")
            continue

        time_str = f"{r['fhe_exec_time']:.4f}" if r["fhe_exec_time"] else "N/A"
        print(
            f"{r['name']:<20} | {r['mse']:.6f} | {r['mae']:.6f} | {time_str:<10} | True"
        )

        # Determine 'best' (simple heuristic)
        # If we have time data, use MSE * Time. If not, just use MSE.
        if r["fhe_exec_time"]:
            score = r["mse"] * r["fhe_exec_time"]
            if score < min_score:
                min_score = score
                best_config = r["name"]
        else:
            # If no timing, just pick best MSE
            if r["mse"] < min_score:
                min_score = r["mse"]
                best_config = r["name"]

    print("=" * 80)
    if best_config:
        print(f"Best trade-off configuration found: {best_config}")


if __name__ == "__main__":
    main()

