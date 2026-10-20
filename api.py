import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path
from functools import lru_cache
import json
import logging

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EV Battery Prediction API")

BASE_DIR = Path(__file__).resolve().parent


class BatteryInput(BaseModel):
    cycle_count: int
    avg_ambient_temp_c: float
    voltage_avg_v: float
    battery_age_days: int


@lru_cache(maxsize=1)
def load_models():
    reg_path = BASE_DIR / "soh_regression_model_compressed.joblib"
    clf_path = BASE_DIR / "degradation_classifier_compressed.joblib"
    config_path = BASE_DIR / "model_config.json"

    for path in (reg_path, clf_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path.name}")

        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("%s: %.2f MB on disk", path.name, size_mb)

    logger.info("Loading regression model")
    reg_model = joblib.load(reg_path)
    logger.info("Regression model loaded")

    logger.info("Loading classifier model")
    clf_model = joblib.load(clf_path)
    logger.info("Classifier model loaded")

    with open(config_path, "r") as file:
        config = json.load(file)

    return reg_model, clf_model, config


@app.get("/")
def home():
    return {"message": "EV Battery Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: BatteryInput):
    try:
        reg_model, clf_model, config = load_models()

        values = {
            "cycle_count": data.cycle_count,
            "avg_ambient_temp_c": data.avg_ambient_temp_c,
            "voltage_avg_v": data.voltage_avg_v,
            "battery_age_days": data.battery_age_days,
        }

        features = config["features"]
        input_data = np.asarray(
            [[values[name] for name in features]],
            dtype=np.float64,
        )

        health_value = float(
            reg_model.predict(input_data)[0]
        )
        category = str(
            clf_model.predict(input_data)[0]
        )

        return {
            "state_of_health": round(health_value, 2),
            "degradation_category": category,
        }

    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the API logs.",
        ) from error
