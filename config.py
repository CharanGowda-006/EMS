"""Configuration settings for the EMS Flask backend."""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent


def resolve_artifact_path(env_name: str, default_name: str) -> Path:
    """Resolve an artifact path from an environment variable, the current working directory, or the workspace root."""
    configured = os.getenv(env_name)
    if configured:
        return Path(configured).expanduser().resolve()

    cwd_candidate = (Path.cwd() / default_name).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    base_candidate = (BASE_DIR / default_name).resolve()
    if base_candidate.exists():
        return base_candidate

    return cwd_candidate


MODEL_PATH = resolve_artifact_path("EMS_MODEL_PATH", "model.pkl")
SCALER_PATH = resolve_artifact_path("EMS_SCALER_PATH", "scaler.pkl")
SCORE_MIN_PATH = resolve_artifact_path("EMS_SCORE_MIN_PATH", "score_min.pkl")
SCORE_MAX_PATH = resolve_artifact_path("EMS_SCORE_MAX_PATH", "score_max.pkl")
DAY_ENCODER_PATH = resolve_artifact_path("EMS_DAY_ENCODER_PATH", "day_encoder.pkl")

FEATURE_COLUMNS = ["day", "hour", "Living", "Kitchen", "Bathroom", "is_weekend"]
INPUT_FIELDS = ["day", "hour", "living", "kitchen", "bathroom", "weekend"]
REQUIRED_FIELDS = INPUT_FIELDS
LOG_LEVEL = os.getenv("EMS_LOG_LEVEL", "INFO")
HOST = os.getenv("EMS_HOST", "0.0.0.0")
PORT = int(os.getenv("EMS_PORT", "5000"))
