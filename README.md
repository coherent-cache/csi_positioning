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

Concrete ML currently requires Python 3.12, so we recommend creating or using a `uv` venv that targets that interpreter. If you already have one, rerun the commands inside it with `uv run`.

```bash
cd csi_positioning
uv run python --version         # confirm the active interpreter is Python 3.12.x
uv pip install --upgrade pip    # keep the managed pip current
uv pip install -r requirements/location_cnn.txt
```

Use `uv run python ...` to launch the training pipeline so all dependencies stay inside the uv-managed environment. If you need to create a fresh venv, `uv venv create python3.12` is the recommended approach before repeating the commands above.

## 3. Segmented workflow

The repository ships four standalone scripts that mirror the plan above. Each script can be run independently and shares the dataset / checkpoint folders explained earlier.

1. **Training (`train_location_cnn.py`)**

   - Default command: `uv run python train_location_cnn.py --dataset-file dataset/dataset_SNR50_outdoor.mat`
   - Persists checkpoints under `checkpoints/location_cnn.pt`, writes normalization stats to `artifacts/feature_stats.json`, and logs loss + R² to `logs/location_training_metrics.csv`.
   - The script will **skip** training when an existing checkpoint is found; pass `--retrain` to overwrite or `--resume` to load the weights and continue optimizing.

2. **FHE compilation (`compile_fhe_location.py`)**

   - Point at the same dataset + checkpoint: `uv run python compile_fhe_location.py --checkpoint checkpoints/location_cnn.pt`
   - Generates a quantized module pickle at `artifacts/location_quantized.pkl` and saves debugging artifacts (MLIR, graph, statistics) under `artifacts/location_quantized_debug/`.
   - Uses Concrete ML `compile_torch_model` with `AveragePool` layers, rounding thresholds, and basic packing settings; make sure the checkpoint exists so the conversion pipeline reuses it instead of triggering another training run.

3. **Visualization (`plot_training_curves.py`)**

   - Reads `logs/location_training_metrics.csv` and writes PNG plots in the `plots/` directory (`training_loss.png`, `validation_loss.png`, `training_r2.png`, `validation_r2.png`).
   - Useful when troubleshooting divergence or verifying that the training loss is decreasing before compiling for FHE.

4. **Inference (`infer_location.py`)**
   - Runs the trained CNN (via `torch`) on several validation samples and optionally loads the quantized module to show `fhe="simulate"` or `fhe="execute"` outputs.
   - Default usage: `uv run python infer_location.py --checkpoint checkpoints/location_cnn.pt --quantized-module-path artifacts/location_quantized.pkl`.
   - If the quantized module file is missing or corrupted, the script still runs clear inference so you can compare predictions without rerunning training.
