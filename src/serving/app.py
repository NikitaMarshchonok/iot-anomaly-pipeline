import json
from pathlib import Path

import numpy as np
import joblib
import redis
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent.parent
BUNDLE_DIR = BASE_DIR / "model" / "bundle"

MODEL = tf.saved_model.load(str(BUNDLE_DIR / "saved_model"))
SCALER = joblib.load(str(BUNDLE_DIR / "scaler.pkl"))
with open(BUNDLE_DIR / "meta.json") as f:
    META = json.load(f)

THRESHOLD = META["threshold"]
SEQ_LEN = META["seq_len"]

REDIS_CLIENT = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)

app = FastAPI(title="IoT Anomaly Detection API")

class PredictRequest(BaseModel):
    series: list[float]

def make_windows(arr: list[float], window: int) -> list[list[float]]:
    return [arr[i : i + window] for i in range(len(arr) - window + 1)]

@app.post("/predict")
async def predict(req: PredictRequest):
    data = req.series
    if len(data) < SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {SEQ_LEN} points, got {len(data)}"
        )

    windows = make_windows(data, SEQ_LEN)
    X = np.array(windows).reshape(-1, SEQ_LEN, 1)
    flat = X.reshape(-1, 1)
    X_scaled = SCALER.transform(flat).reshape(X.shape)

    recon = MODEL(X_scaled)
    errors = np.mean((X_scaled - recon.numpy())**2, axis=(1,2))

    result = []
    for err in errors:
        result.append({
            "error": float(err),
            "is_anomaly": err > THRESHOLD
        })

    REDIS_CLIENT.lpush("anomaly_results", json.dumps(result[-1]))
    REDIS_CLIENT.ltrim("anomaly_results", 0, 99)

    return {"predictions": result}

@app.get("/latest")
async def latest(count: int = 10):
    raw = REDIS_CLIENT.lrange("anomaly_results", 0, count - 1)
    return [json.loads(r) for r in raw]
