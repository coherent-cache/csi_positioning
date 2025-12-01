"""Deployment server for Location CNN.

Routes:
    - /get_client: Get client.zip (FHE config)
    - /add_key: Add a client's public key (FHE)
    - /compute: Compute on encrypted data (FHE)
    - /compute_clear: Compute on plaintext data
"""

import io
import os
import uuid
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

# Concrete ML deployment helper
from concrete.ml.deployment import FHEModelServer

# Application specific imports
from location_cnn.models import LocationCNN
from location_cnn.data import load_feature_stats, normalize_features


# Define request model for cleartext
class CleartextInput(BaseModel):
    features: List[float]


app = FastAPI(title="Location CNN FHE Server")

# Configuration
DEPLOYMENT_DIR = Path(os.environ.get("DEPLOYMENT_DIR", "deployment_artifacts"))
CHECKPOINT_PATH = Path(os.environ.get("CHECKPOINT_PATH", "checkpoints/location_cnn.pt"))
STATS_PATH = Path(os.environ.get("STATS_PATH", "artifacts/feature_stats.json"))
PORT = int(os.environ.get("PORT", "8000"))

# Global state
KEYS: Dict[str, bytes] = {}
fhe_server = None
clear_model = None
feature_mean = None
feature_std = None


def load_clear_model():
    """Load the PyTorch model for cleartext inference."""
    global clear_model, feature_mean, feature_std

    print(f"Loading cleartext model from {CHECKPOINT_PATH}...")
    if not CHECKPOINT_PATH.exists():
        print("Warning: Checkpoint not found. Cleartext inference will fail.")
        return

    model = LocationCNN()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    clear_model = model

    if STATS_PATH.exists():
        feature_mean, feature_std = load_feature_stats(STATS_PATH)
        print(
            f"Loaded normalization stats: mean={feature_mean:.4f}, std={feature_std:.4f}"
        )
    else:
        print("Warning: Stats file not found. Normalization might be incorrect.")


@app.on_event("startup")
async def startup_event():
    """Initialize FHE server and load clear model."""
    global fhe_server

    # Initialize FHE Server
    if DEPLOYMENT_DIR.exists() and (DEPLOYMENT_DIR / "server.zip").exists():
        print(f"Initializing FHE Server from {DEPLOYMENT_DIR}...")
        fhe_server = FHEModelServer(str(DEPLOYMENT_DIR.resolve()))
    else:
        print(f"Warning: Deployment directory {DEPLOYMENT_DIR} invalid. FHE will fail.")

    # Load Clear Model
    load_clear_model()


@app.get("/get_client")
def get_client():
    """Get client.zip for FHE configuration."""
    path_to_client = DEPLOYMENT_DIR / "client.zip"
    if not path_to_client.exists():
        raise HTTPException(status_code=500, detail="Could not find client.zip.")
    return FileResponse(path_to_client, media_type="application/zip")


@app.post("/add_key")
async def add_key(key: UploadFile):
    """Add public key for FHE session."""
    uid = str(uuid.uuid4())
    KEYS[uid] = await key.read()
    return {"uid": uid}


@app.post("/compute")
async def compute(model_input: UploadFile, uid: str = Form()):
    """Compute on encrypted data."""
    if fhe_server is None:
        raise HTTPException(status_code=500, detail="FHE Server not initialized.")

    if uid not in KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown UID: {uid}")

    key = KEYS[uid]
    try:
        encrypted_results = fhe_server.run(
            serialized_encrypted_quantized_data=await model_input.read(),
            serialized_evaluation_keys=key,
        )
        return StreamingResponse(
            io.BytesIO(encrypted_results), media_type="application/octet-stream"
        )
    except Exception as e:
        print(f"FHE Computation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compute_clear")
async def compute_clear(input_data: CleartextInput):
    """Compute on cleartext data."""
    if clear_model is None:
        raise HTTPException(status_code=500, detail="Cleartext model not loaded.")

    try:
        # Prepare input
        features = np.array(input_data.features, dtype=np.float32).reshape(4, 16, 193)

        # Normalize if stats are available
        # Note: The input is expected to be raw features?
        # Or normalized features?
        # Let's assume raw features to match the 'client' loading raw data.
        # But we need to be consistent with how client sends data.
        # If client normalizes before sending, we shouldn't normalize here.
        # Standard flow: Client prepares input.
        # In FHE flow: Client normalizes -> quantizes -> encrypts.
        # So in clear flow: Client should normalize -> send.
        # I will assume the client handles normalization to be consistent with FHE flow where client owns preprocessing.

        input_tensor = torch.from_numpy(features).unsqueeze(0)  # Add batch dim

        with torch.no_grad():
            prediction = clear_model(input_tensor).numpy().flatten()

        return {"prediction": prediction.tolist()}

    except Exception as e:
        print(f"Cleartext Computation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
