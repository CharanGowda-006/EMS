"""Machine learning prediction logic for the EMS backend."""

from __future__ import annotations

import logging
import math
from numbers import Real
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import (
    DAY_ENCODER_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    SCALER_PATH,
    SCORE_MAX_PATH,
    SCORE_MIN_PATH,
)

logger = logging.getLogger("ems.predictor")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ENCODER_MAPPING = {day: index for index, day in enumerate(DAY_NAMES)}


class PredictorService:
    """Loads and serves the trained anomaly detection model."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.scaler: Any | None = None
        self.day_encoder: Any | None = None
        self.score_min: float | None = None
        self.score_max: float | None = None
        self.is_ready = False

    def initialize(self) -> None:
        """Load model artifacts once during startup."""
        if self.is_ready:
            return

        logger.info(
            "Loading model artifacts from %s, %s, %s, %s, %s",
            MODEL_PATH,
            SCALER_PATH,
            SCORE_MIN_PATH,
            SCORE_MAX_PATH,
            DAY_ENCODER_PATH,
        )
        for artifact_path, artifact_name in (
            (MODEL_PATH, "model"),
            (SCALER_PATH, "scaler"),
            (SCORE_MIN_PATH, "score_min"),
            (SCORE_MAX_PATH, "score_max"),
            (DAY_ENCODER_PATH, "day_encoder"),
        ):
            if not artifact_path.exists():
                raise FileNotFoundError(f"Required artifact not found: {artifact_name} at {artifact_path}")

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.score_min = float(joblib.load(SCORE_MIN_PATH))
        self.score_max = float(joblib.load(SCORE_MAX_PATH))

        try:
            self.day_encoder = joblib.load(DAY_ENCODER_PATH)
            logger.info("Loaded day encoder from %s", DAY_ENCODER_PATH)
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError(f"Failed to load day encoder from {DAY_ENCODER_PATH}") from exc

        if self.score_max <= self.score_min:
            raise ValueError("score_max must be greater than score_min")

        self._validate_scaler_features()
        self.is_ready = True
        logger.info("Model artifacts loaded successfully")

    def _validate_scaler_features(self) -> None:
        """Ensure the scaler was trained with the same feature layout."""
        if self.scaler is None:
            raise RuntimeError("Scaler is not loaded")

        feature_names = getattr(self.scaler, "feature_names_in_", None)
        if feature_names is None:
            return

        names = [str(name) for name in feature_names]
        if names != FEATURE_COLUMNS:
            raise ValueError(
                f"Scaler feature mismatch: expected {FEATURE_COLUMNS}, got {names}"
            )

    def get_status(self) -> str:
        """Return the readiness state of the predictor."""
        return "loaded" if self.is_ready else "not_loaded"

    def _normalize_day(self, day: Any) -> str:
        """Validate and normalize the day input."""
        if not isinstance(day, str):
            raise ValueError("day must be a string")

        normalized = day.strip().lower().title()
        if normalized not in DAY_NAMES:
            raise ValueError("day must be one of: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday")
        return normalized

    def _encode_day(self, day: Any) -> int:
        """Encode the day using the same mapping as training."""
        if self.day_encoder is None:
            raise RuntimeError("Day encoder is not loaded")

        normalized_day = self._normalize_day(day)
        if hasattr(self.day_encoder, "transform"):
            return int(self.day_encoder.transform([normalized_day])[0])
        if isinstance(self.day_encoder, dict):
            if normalized_day not in self.day_encoder:
                raise ValueError("day encoder does not contain the requested day")
            return int(self.day_encoder[normalized_day])
        raise RuntimeError("Unsupported day encoder format")

    def validate_inputs(self, day: Any, living: Any, kitchen: Any, bathroom: Any, hour: Any, weekend: Any) -> dict[str, float | int]:
        """Validate and normalize incoming feature values."""
        values = {
            "day": day,
            "living": living,
            "kitchen": kitchen,
            "bathroom": bathroom,
            "hour": hour,
            "weekend": weekend,
        }

        for name, value in values.items():
            if name == "day":
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a numeric value")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")

        if not 0 <= int(hour) <= 23:
            raise ValueError("hour must be between 0 and 23")
        if int(weekend) not in {0, 1}:
            raise ValueError("weekend must be 0 or 1")
        if float(living) < 0 or float(kitchen) < 0 or float(bathroom) < 0:
            raise ValueError("living, kitchen, and bathroom must be non-negative")

        encoded_day = self._encode_day(day)
        normalized = {
            "day": encoded_day,
            "hour": int(hour),
            "Living": float(living),
            "Kitchen": float(kitchen),
            "Bathroom": float(bathroom),
            "is_weekend": int(weekend),
        }
        return normalized

    def predict_activity(self, day: Any, living: Any, kitchen: Any, bathroom: Any, hour: Any, weekend: Any) -> dict[str, Any]:
        """Run the ML pipeline and return a prediction dictionary."""
        if not self.is_ready:
            raise RuntimeError("Model artifacts are not loaded")

        validated = self.validate_inputs(day, living, kitchen, bathroom, hour, weekend)
        logger.info("Validated features: %s", validated)

        features = pd.DataFrame([validated], columns=FEATURE_COLUMNS)
        logger.info("Feature DataFrame: %s", features.to_dict(orient="records"))

        try:
            scaled_features = self.scaler.transform(features)
        except ValueError as exc:
            logger.exception("Scaler feature mismatch")
            raise ValueError("Scaler feature mismatch") from exc
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Scaling failed")
            raise RuntimeError("Feature scaling failed") from exc

        logger.info("Scaled feature vector: %s", scaled_features.tolist())

        try:
            prediction = int(self.model.predict(scaled_features)[0])
            score = float(self.model.decision_function(scaled_features)[0])
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Prediction failed")
            raise RuntimeError("Prediction could not be completed") from exc

        if self.score_min is None or self.score_max is None:
            raise RuntimeError("Score bounds are not loaded")

        normalized = (score - self.score_min) / (self.score_max - self.score_min)
        normalized = float(np.clip(normalized, 0.0, 1.0))
        confidence = round((1.0 - normalized) * 100.0, 1)

        anomaly = prediction == -1
        result = {
            "status": "Anomaly" if anomaly else "Normal",
            "anomaly": anomaly,
            "confidence": confidence,
            "score": round(score, 2),
        }
        logger.info("Prediction: %s", prediction)
        logger.info("Anomaly score: %.6f", score)
        logger.info("Confidence: %.1f", confidence)
        return result


predictor_service = PredictorService()


def initialize_predictor() -> None:
    """Initialize the predictor service once at startup."""
    predictor_service.initialize()


def is_predictor_ready() -> bool:
    """Expose the readiness state of the predictor."""
    return predictor_service.is_ready


def predict_activity(day: Any, living: Any, kitchen: Any, bathroom: Any, hour: Any, weekend: Any) -> dict[str, Any]:
    """Expose the prediction entry point for the Flask app."""
    return predictor_service.predict_activity(day, living, kitchen, bathroom, hour, weekend)
