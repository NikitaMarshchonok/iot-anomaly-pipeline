import json
from pathlib import Path
from typing import List

import numpy as np
import joblib
import redis
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ────────────────────────────────────────────────────────────────────────────────
# 1) схемы Pydantic
# ────────────────────────────────────────────────────────────────────────────────

class Prediction(BaseModel):
    index:   int
    value:   float
    anomaly: bool

class PredictRequest(BaseModel):
    series: List[float] = Field(
        ...,
        description="Одномерный временной ряд; длина ≥ seq_len",
        example=[i * 0.01 for i in range(60)],
    )

class PredictResponse(BaseModel):
    predictions: List[Prediction]

class HistoryResponse(BaseModel):
    history: List[List[Prediction]]


# ────────────────────────────────────────────────────────────────────────────────
# 2) загрузка артефактов
# ────────────────────────────────────────────────────────────────────────────────

# ищем корень так, чтобы работало и локально, и из Docker-контейнера
BASE_DIR = Path(__file__).resolve().parent
if not (BASE_DIR / "model" / "bundle").exists():
    BASE_DIR = BASE_DIR.parent.parent
BUNDLE_DIR = BASE_DIR / "model" / "bundle"

MODEL     = tf.saved_model.load(str(BUNDLE_DIR / "saved_model"))
SCALER    = joblib.load(str(BUNDLE_DIR / "scaler.pkl"))
META      = json.loads((BUNDLE_DIR / "meta.json").read_text())
THRESHOLD = META["threshold"]
SEQ_LEN    = META["seq_len"]

REDIS_CLIENT = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)


# ────────────────────────────────────────────────────────────────────────────────
# 3) приложение
# ────────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="IoT Anomaly Detection API", version="0.1.0")


def make_windows(arr: List[float], window: int) -> np.ndarray:
    """
    Разбивает список в перекрывающиеся окна формы (n_windows, window).
    """
    return np.array([arr[i : i + window] for i in range(len(arr) - window + 1)])


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Сделать предсказание аномалий",
)
async def predict(req: PredictRequest):
    if len(req.series) < SEQ_LEN:
        raise HTTPException(400, f"Need at least {SEQ_LEN} points, got {len(req.series)}")

    # 1) формируем окна
    windows = make_windows(req.series, SEQ_LEN)            # (n_windows, SEQ_LEN)
    n_win    = windows.shape[0]

    # 2) Flatten → scale → Restore
    flat        = windows.reshape(-1, 1)                   # (n_windows*SEQ_LEN, 1)
    flat_scaled = SCALER.transform(flat)                   # (n_windows*SEQ_LEN, 1)
    scaled      = flat_scaled.reshape(n_win, SEQ_LEN)      # (n_windows, SEQ_LEN)

    # 3) LSTM ждёт третий измеритель (features=1)
    model_input = scaled[..., np.newaxis]                  # (n_windows, SEQ_LEN, 1)

    # 4) инференс
    preds = MODEL(model_input).numpy().squeeze()            # (n_windows, SEQ_LEN)

    # 5) считаем MSE
    mses = np.mean((preds - scaled) ** 2, axis=1)           # (n_windows,)

    # 6) собираем ответ
    result = [
        Prediction(index=i, value=float(mses[i]), anomaly=bool(mses[i] > THRESHOLD))
        for i in range(n_win)
    ]

    # 7) пушим в Redis
    REDIS_CLIENT.rpush("results", json.dumps([r.dict() for r in result]))

    return PredictResponse(predictions=result)


@app.get(
    "/latest",
    response_model=HistoryResponse,
    summary="Получить последние N предсказаний",
)
async def latest(n: int = 1):
    raw     = REDIS_CLIENT.lrange("results", -n, -1) or []
    history = [
        [Prediction(**item) for item in json.loads(entry)]
        for entry in raw
    ]
    return HistoryResponse(history=history)
