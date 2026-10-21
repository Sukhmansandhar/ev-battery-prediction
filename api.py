import os

# Limit numerical-library threads to reduce memory overhead.
# Set these before importing NumPy or scikit-learn.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("battery_api")

app = FastAPI(title="EV Battery Prediction API")

BASE_DIR = Path(__file__).resolve().parent


class BatteryInput(BaseModel):
    cycle_count: int
    avg_ambient_temp_c: float
    voltage_avg_v: float
    battery_age_days: int


def load_config():
    config_path = BASE_DIR / "model_config.json"

    if not config_path.is_file():
        raise FileNotFoundError("model_config.json not found")

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    features = config.get("features")

    expected_features = {
        "cycle_count",
        "avg_ambient_temp_c",
        "voltage_avg_v",
        "battery_age_days",
    }

    if not isinstance(features, list) or not features:
        raise ValueError(
            "model_config.json must contain a non-empty features list"
        )

    if set(features) != expected_features:
        raise ValueError(
            f"Unexpected model features: {features}"
        )

    return config


@app.get("/")
@app.head("/")
def home():
    return {
        "message": "EV Battery Prediction API is running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: BatteryInput):
    try:
        reg_path = (
    BASE_DIR / "soh_regression_model_small.joblib"
)
        
        clf_path = (
            BASE_DIR / "degradation_classifier_compressed.joblib"
        )

        config = load_config()

        for path in (reg_path, clf_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required model file not found: {path.name}"
                )

            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info("%s: %.2f MB on disk", path.name, size_mb)

        values = {
            "cycle_count": data.cycle_count,
            "avg_ambient_temp_c": data.avg_ambient_temp_c,
            "voltage_avg_v": data.voltage_avg_v,
            "battery_age_days": data.battery_age_days,
        }

        input_data = np.asarray(
            [[values[name] for name in config["features"]]],
            dtype=np.float64,
        )

        # Load and run the regression model first.
        logger.info("Loading regression model")
        reg_model = joblib.load(reg_path)

        logger.info("Running regression prediction")
        health_value = float(
            reg_model.predict(input_data)[0]
        )

        del reg_model
        gc.collect()

        # Load and run the classifier separately.
        logger.info("Loading classifier model")
        clf_model = joblib.load(clf_path)

        logger.info("Running degradation classification")
        category = str(
            clf_model.predict(input_data)[0]
        )

        del clf_model
        gc.collect()

        logger.info("Prediction completed successfully")

        return {
            "state_of_health": round(health_value, 2),
            "degradation_category": category,
        }

    except Exception as error:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {type(error).__name__}. "
                "Check Render logs for details."
            ),
        ) from error
