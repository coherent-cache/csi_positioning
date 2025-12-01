"""Client script for Location CNN FHE/Cleartext Inference."""

import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
import torch
from concrete.ml.deployment import FHEModelClient

from location_cnn.data import (
    load_feature_stats,
    load_mat_dataset,
    normalize_features,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Client for Location CNN Inference")
    parser.add_argument(
        "--mode",
        choices=["fhe", "clear"],
        required=True,
        help="Inference mode: 'fhe' (encrypted) or 'clear' (plaintext)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Server URL",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("dataset/dataset_SNR50_outdoor.mat"),
        help="Dataset file",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("artifacts/feature_stats.json"),
        help="Feature stats file",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Index of the sample to test",
    )
    return parser.parse_args()


def load_sample(dataset_path, stats_path, idx):
    """Load and normalize a sample."""
    print(f"Loading dataset from {dataset_path}...")
    features, targets = load_mat_dataset(dataset_path)

    print(f"Loading stats from {stats_path}...")
    mean, std = load_feature_stats(stats_path)
    if mean is None or std is None:
        # Fallback: compute from dataset (should ideally exist)
        print("Stats not found, computing...")
        _, mean, std = normalize_features(features)

    normalized_features, _, _ = normalize_features(features, mean, std)

    if idx >= len(normalized_features):
        raise ValueError(
            f"Sample index {idx} out of range (0-{len(normalized_features)-1})"
        )

    sample = normalized_features[idx]
    target = targets[idx]
    return sample, target


def run_fhe_inference(url, sample, target):
    """Run FHE inference."""
    print("\n=== Running FHE Inference ===")

    # 1. Setup Client
    print("Setting up FHE client...")
    client_dir = Path("client_fhe_data")
    client_dir.mkdir(exist_ok=True)

    # Download client.zip if not present (or always to be safe)
    print(f"Downloading client.zip from {url}/get_client...")
    try:
        r = requests.get(f"{url}/get_client")
        r.raise_for_status()
        (client_dir / "client.zip").write_bytes(r.content)
    except Exception as e:
        print(f"Error downloading client.zip: {e}")
        return

    client = FHEModelClient(path_dir=str(client_dir), key_dir=str(client_dir / "keys"))

    # 2. Key Generation
    print("Generating/Loading keys...")
    client.get_serialized_evaluation_keys()

    # 3. Send Keys
    print("Sending evaluation keys to server...")
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    r = requests.post(
        f"{url}/add_key",
        files={"key": io.BytesIO(serialized_evaluation_keys)},
    )
    r.raise_for_status()
    uid = r.json()["uid"]
    print(f"Received UID: {uid}")

    # 4. Encrypt Data
    print("Encrypting data...")
    # Sample needs to be (1, 4, 16, 193) for model
    sample_batch = sample.reshape(1, 4, 16, 193)
    encrypted_input = client.quantize_encrypt_serialize(sample_batch)
    print(f"Encrypted payload size: {len(encrypted_input)} bytes")

    # 5. Send for Computation
    print("Sending encrypted data for computation...")
    start_time = time.time()
    r = requests.post(
        f"{url}/compute",
        files={"model_input": io.BytesIO(encrypted_input)},
        data={"uid": uid},
        stream=True,
    )
    r.raise_for_status()
    encrypted_result = r.content
    duration = time.time() - start_time
    print(f"Received encrypted response in {duration:.2f}s")
    print(f"Response payload size: {len(encrypted_result)} bytes")

    # 6. Decrypt Result
    print("Decrypting result...")
    decrypted_prediction = client.deserialize_decrypt_dequantize(encrypted_result)
    prediction = decrypted_prediction[0]

    print(f"\nGround Truth: {target}")
    print(f"Prediction:   {prediction}")
    print(f"Error:        {np.abs(target - prediction)}")


def run_clear_inference(url, sample, target):
    """Run Cleartext inference."""
    print("\n=== Running Cleartext Inference ===")

    # Sample needs to be flat list for JSON
    payload = {"features": sample.flatten().tolist()}

    print("Sending plaintext data...")
    start_time = time.time()
    try:
        r = requests.post(f"{url}/compute_clear", json=payload)
        r.raise_for_status()
        result = r.json()
        prediction = np.array(result["prediction"])
        duration = time.time() - start_time

        print(f"Received response in {duration:.2f}s")
        print(f"\nGround Truth: {target}")
        print(f"Prediction:   {prediction}")
        print(f"Error:        {np.abs(target - prediction)}")

    except Exception as e:
        print(f"Error during cleartext inference: {e}")


def main():
    args = parse_args()

    sample, target = load_sample(args.dataset_file, args.stats_path, args.sample_idx)

    if args.mode == "fhe":
        run_fhe_inference(args.url, sample, target)
    else:
        run_clear_inference(args.url, sample, target)


if __name__ == "__main__":
    main()
