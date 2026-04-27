from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import os

REQUEST_COUNT = Counter(
    "app_requests_total",
    "Total number of requests"
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency in seconds"
)

CONFIDENCE_GAUGE = Gauge(
    "model_confidence",
    "Prediction confidence value"
)

app = FastAPI()


MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model.pkl")

model = joblib.load(MODEL_PATH)

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):

    start = time.time()

    X = np.array(data.features).reshape(1, -1)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    confidence = float(np.max(prob))

    # Prometheus updates
    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(time.time() - start)
    CONFIDENCE_GAUGE.set(confidence)

    return {
        "prediction": int(pred),
        "confidence": confidence
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
