# Location CNN + FHE Workflow

This project trains a compact CNN on the 5G channel-frequency response dataset and then compiles the model with Concrete ML so you can evaluate the trained predictor in both simulated and real Fully Homomorphic Encryption (FHE) modes.

## 1. Dataset snapshot

The project uses the **CSI Dataset towards 5G NR High-Precision Positioning**.

**Download Instructions:**

1. Visit [IEEE DataPort](https://ieee-dataport.org/open-access/csi-dataset-towards-5g-nr-high-precision-positioning).
2. Log in (free account required).
3. Download the `.mat` files (e.g., `dataset_SNR50_outdoor.mat`).
4. Place them in the `dataset/` directory.

- Expected files: `dataset/dataset_SNR10_indoor_21-11-17_03-07.mat`, `dataset/dataset_SNR10_outdoor.mat`, `dataset/dataset_SNR20_indoor_21-11-17_01-50.mat`, `dataset/dataset_SNR20_outdoor.mat`, `dataset/dataset_SNR50_indoor_21-11-16_23-11.mat`, `dataset/dataset_SNR50_outdoor.mat`
- Feature shape: `(3876, 4, 16, 193)` with `float64` values (samples, Rx antennas, subcarriers, frequency bins)
- Target shape: `(3876, 3)` giving ground-truth `[x, y, z]` coordinates per sample
- The training script defaults to `dataset_SNR50_outdoor.mat`, but you can switch to any file above via `--dataset-file`.

## 2. Environment setup (Python 3.12 via `uv`)

Concrete ML currently requires Python 3.12, so we recommend creating or using a `uv` venv that targets that interpreter.

```bash
cd csi_positioning
# Create a fresh virtualenv with Python 3.12
uv venv create python3.12
# If you already have one, ensure it's active or use 'uv run' as shown below.

uv run python --version         # confirm the active interpreter is Python 3.12.x
uv pip install --upgrade pip    # keep the managed pip current
uv pip install -r requirements/location_cnn.txt
uv pip install -r requirements/deployment.txt # install deployment dependencies
```

Use `uv run python ...` to launch the training pipeline so all dependencies stay inside the uv-managed environment.

## 3. Project Scripts

The repository ships standalone scripts that mirror the plan above. Each script can be run independently and shares the dataset / checkpoint folders explained earlier.

### Training (`train_location_cnn.py`)

Default command:

```bash
uv run python train_location_cnn.py --dataset-file dataset/dataset_SNR50_outdoor.mat
```

- Persists checkpoints under `checkpoints/location_cnn.pt`, writes normalization stats to `artifacts/feature_stats.json`, and logs loss + R² to `logs/location_training_metrics.csv`.
- The script will **skip** training when an existing checkpoint is found; pass `--retrain` to overwrite or `--resume` to load the weights and continue optimizing.

### FHE compilation (`compile_fhe_location.py`)

Point at the same dataset + checkpoint:

```bash
uv run python compile_fhe_location.py --checkpoint checkpoints/location_cnn.pt
```

- Generates a quantized module JSON at `artifacts/location_quantized.json` and saves debugging artifacts (MLIR, graph, statistics) under `artifacts/location_quantized_debug/`.
- Uses Concrete ML `compile_torch_model` with `AveragePool` layers, rounding thresholds, and basic packing settings; make sure the checkpoint exists so the conversion pipeline reuses it instead of triggering another training run.

### Visualization (`plot_training_curves.py`)

```bash
uv run python plot_training_curves.py
```

- Reads `logs/location_training_metrics.csv` and writes PNG plots in the `plots/` directory (`training_loss.png`, `validation_loss.png`, `training_r2.png`, `validation_r2.png`).
- Useful when troubleshooting divergence or verifying that the training loss is decreasing before compiling for FHE.

### Inference (`infer_location.py`)

Runs the trained CNN (via `torch`) on several validation samples and optionally loads the quantized module to show `fhe="simulate"` or `fhe="execute"` outputs.

Default usage:

```bash
uv run python infer_location.py --checkpoint checkpoints/location_cnn.pt --quantized-module-path artifacts/location_quantized.json
```

- If the quantized module file is missing or corrupted, the script still runs clear inference so you can compare predictions without rerunning training.

## 4. Client-Server Deployment Demo

This project includes a full client-server setup to demonstrate FHE inference over HTTP. This setup allows you to inspect encrypted traffic (e.g., using Wireshark).

### Prerequisites

Ensure you have installed the deployment requirements:

```bash
uv pip install -r requirements/deployment.txt
```

### Step 1: Compile for Deployment

Generate the necessary FHE artifacts (`client.zip`, `server.zip`) optimized for deployment:

```bash
uv run python compile_for_deployment.py --p-error 0.1
```

This creates a `deployment_artifacts/` directory containing the compiled FHE circuit and configuration files.

### Step 2: Start the Server

Launch the FastAPI server which handles both key exchange and encrypted inference:

```bash
uv run python server.py
```

The server will start on `http://0.0.0.0:8000`.

### Step 3: Run the Client

You can run the client in two modes to compare plaintext vs. FHE execution.

**Cleartext Mode (Baseline):**

```bash
uv run python client.py --mode clear
```

_Sends raw floating point numbers. Response is instantaneous._

**FHE Mode (Encrypted):**

```bash
uv run python client.py --mode fhe
```

_Downloads `client.zip`, generates keys, encrypts data, sends ciphertext, and decrypts the result. This is computationally intensive._

### Step 4: Inspect Traffic (Wireshark)

To visually verify encryption:

1. Open **Wireshark**.
2. Capture traffic on your loopback interface (`Lo0` or `Adapter for loopback traffic capture`).
3. Filter for port 8000: `tcp.port == 8000`.
4. Run the client in `clear` mode → Observe readable JSON payloads.
5. Run the client in `fhe` mode → Observe binary ciphertext blobs (unreadable).

## 5. References

- **Paper**: [Toward 5G NR High-Precision Indoor Positioning via Channel Frequency Response](assets/Toward_5G_NR_High-Precision_Indoor_Positioning_via_Channel_Frequency_Response_A_New_Paradigm_and_Dataset_Generation_Method.pdf) (included in `assets/`)
- **Dataset**: [IEEE DataPort - CSI Dataset towards 5G NR High-Precision Positioning](https://ieee-dataport.org/open-access/csi-dataset-towards-5g-nr-high-precision-positioning)
