import json
from pathlib import Path
from typing import List

import numpy as np
import joblib
import redis
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Pydantic-схемы
class Prediction(BaseModel):
    index: int
    value: float
    anomaly: bool

class PredictRequest(BaseModel):
    series: List[float]

class PredictResponse(BaseModel):
    predictions: List[Prediction]

class HistoryResponse(BaseModel):
    history: List[List[Prediction]]

# Определяем корень проекта и путь до бандла
BASE_DIR = Path(__file__).resolve().parent
if not (BASE_DIR / "model" / "bundle").exists():
    BASE_DIR = BASE_DIR.parent.parent
BUNDLE_DIR = BASE_DIR / "model" / "bundle"

# Загружаем артефакты
MODEL  = tf.saved_model.load(str(BUNDLE_DIR / "saved_model"))
SCALER = joblib.load(str(BUNDLE_DIR / "scaler.pkl"))
with open(BUNDLE_DIR / "meta.json") as f:
    META = json.load(f)
THRESHOLD = META["threshold"]
SEQ_LEN   = META["seq_len"]

REDIS_CLIENT = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)

app = FastAPI(title="IoT Anomaly Detection API")

def make_windows(arr: List[float], window: int) -> List[List[float]]:
    return [arr[i : i + window] for i in range(len(arr) - window + 1)]

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run anomaly prediction on a time series",
)
async def predict(req: PredictRequest):
    if len(req.series) < SEQ_LEN:
        raise HTTPException(400, f"Need at least {SEQ_LEN} points")
    windows = np.array(make_windows(req.series, SEQ_LEN))
    scaled  = SCALER.transform(windows)
    preds   = MODEL(scaled).numpy().squeeze()
    mses    = np.mean((preds - scaled) ** 2, axis=1)
    result  = [
        Prediction(index=i, value=float(mses[i]), anomaly=mses[i] > THRESHOLD)
        for i in range(len(mses))
    ]
    REDIS_CLIENT.rpush("results", json.dumps([r.dict() for r in result]))
    return PredictResponse(predictions=result)

@app.get(
    "/latest",
    response_model=HistoryResponse,
    summary="Fetch the last N prediction results",
)
async def latest(n: int = 1):
    raw     = REDIS_CLIENT.lrange("results", -n, -1) or []
    history = [[Prediction(**item) for item in json.loads(r)] for r in raw]
    return HistoryResponse(history=history)
