"""Flask application for the EMS anomaly detection backend."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, HTTPException

from config import HOST, LOG_LEVEL, PORT
from predictor import initialize_predictor, is_predictor_ready, predict_activity


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    app.logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    startup_time = time.time()

    try:
        initialize_predictor()
    except FileNotFoundError as exc:
        app.logger.warning("Model files are missing: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive logging
        app.logger.exception("Failed to initialize predictor at startup")
        app.logger.warning("Prediction endpoint will be unavailable until the issue is resolved: %s", exc)

    @app.before_request
    def log_incoming_request() -> None:
        app.logger.info("Incoming %s %s", request.method, request.path)

    @app.errorhandler(BadRequest)
    def handle_bad_request(error: BadRequest) -> Any:
        return jsonify({"error": "Invalid JSON payload"}), 400

    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException) -> Any:
        return jsonify({"error": error.description or "Request failed"}), error.code or 500

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception) -> Any:
        app.logger.exception("Unhandled exception", exc_info=error)
        return jsonify({"error": "Internal server error"}), 500

    @app.get("/")
    def root() -> Any:
        return jsonify({"status": "running", "model": "loaded"})

    @app.get("/health")
    def health() -> Any:
        uptime_seconds = round(time.time() - startup_time, 2)
        return jsonify(
            {
                "status": "ok",
                "server_status": "ok",
                "model_status": "loaded" if is_predictor_ready() else "not_loaded",
                "uptime_seconds": uptime_seconds,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    @app.post("/predict")
    def predict_endpoint() -> Any:
        try:
            payload = request.get_json(force=False, silent=False)
        except BadRequest:
            return jsonify({"error": "Request body must be valid JSON"}), 400
        except Exception as exc:  # pragma: no cover - defensive logging
            app.logger.exception("Failed to parse JSON")
            return jsonify({"error": "Request body must be valid JSON"}), 400

        if not isinstance(payload, dict):
            return jsonify({"error": "JSON body must be an object"}), 400

        app.logger.info("Received JSON: %s", payload)

        required = ["day", "living", "kitchen", "bathroom", "hour", "weekend"]
        missing = [field for field in required if field not in payload]
        if missing:
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

        try:
            result = predict_activity(
                day=payload["day"],
                living=payload["living"],
                kitchen=payload["kitchen"],
                bathroom=payload["bathroom"],
                hour=payload["hour"],
                weekend=payload["weekend"],
            )
        except ValueError as exc:
            app.logger.warning("Validation error: %s", exc)
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            app.logger.warning("Prediction service error: %s", exc)
            return jsonify({"error": str(exc)}), 503
        except Exception as exc:  # pragma: no cover - defensive logging
            app.logger.exception("Prediction failed")
            return jsonify({"error": "Prediction could not be completed"}), 500

        return jsonify(result), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
