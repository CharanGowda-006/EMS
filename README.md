# EMS Backend

A production-ready Flask backend for an IoT-based elderly monitoring system that uses an Isolation Forest model for anomaly detection.

## Project structure

- app.py: Flask application and API routes
- predictor.py: ML loading and prediction logic
- config.py: configuration and path settings
- train.py: example training script that saves the required model artifacts
- requirements.txt: Python dependencies
- model.pkl: trained Isolation Forest model (place this file in the project root)
- scaler.pkl: trained StandardScaler (place this file in the project root)
- score_min.pkl: minimum decision-function score from training
- score_max.pkl: maximum decision-function score from training

## Installation

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your provided model artifact files in the project root:

   - model.pkl
   - scaler.pkl
   - score_min.pkl
   - score_max.pkl
   - day_encoder.pkl (optional if you want to use a custom encoder)

The backend will use these files directly for inference, and it requires day_encoder.pkl to be present for day-based prediction.

## Running the server

```bash
python app.py
```

The server will start on http://0.0.0.0:5000.

## Model inference pipeline

The backend uses the same feature layout as the training pipeline:

- Input JSON fields: day, hour, living, kitchen, bathroom, weekend
- Internal feature columns: day, hour, Living, Kitchen, Bathroom, is_weekend

During inference:

1. The incoming payload is validated.
2. The day string is mapped to the same encoded integer used during training.
3. Features are converted to the training column order.
4. The scaler transforms the input features.
5. The Isolation Forest predicts whether the activity is anomalous.
6. The decision function score is normalized using score_min.pkl and score_max.pkl.
7. Confidence is computed as $(1 - normalized) \times 100$.

The day encoding follows this mapping:

- Monday = 0
- Tuesday = 1
- Wednesday = 2
- Thursday = 3
- Friday = 4
- Saturday = 5
- Sunday = 6

This ensures the confidence score and feature ordering are statistically consistent with the training data.

## API endpoints

### GET /

Returns basic service status.

Example response:

```json
{
  "status": "running",
  "model": "loaded"
}
```

### GET /health

Returns server health and model readiness.

Example response:

```json
{
  "status": "ok",
  "model_status": "loaded",
  "uptime_seconds": 12.34,
  "timestamp": "2026-07-13T00:00:00+00:00"
}
```

### POST /predict

Accepts a JSON object with the following structure:

```json
{
  "day": "Monday",
  "hour": 14,
  "living": 15,
  "kitchen": 8,
  "bathroom": 2,
  "weekend": 0
}
```

Example response for normal activity:

```json
{
  "status": "Normal",
  "anomaly": false,
  "confidence": 12.5,
  "score": 0.28
}
```

Example response for anomalous activity:

```json
{
  "status": "Anomaly",
  "anomaly": true,
  "confidence": 96.8,
  "score": -0.42
}
```

## Example requests

### Curl

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"day": "Monday", "hour": 14, "living": 15, "kitchen": 8, "bathroom": 2, "weekend": 0}'
```

## Troubleshooting

- If the server reports that the model is not loaded, verify that model.pkl, scaler.pkl, score_min.pkl, score_max.pkl, and day_encoder.pkl exist in the project root.
- If you see a 400 response, check that all required fields are present and that values are numeric.
- If the prediction endpoint returns 503, the model files may be missing, corrupted, or incompatible with the current scaler.
- If you see a scaler mismatch error, retrain the model and regenerate the artifacts with the same feature order used in the inference pipeline.
- If you need to change the model file location, set the EMS_MODEL_PATH, EMS_SCALER_PATH, EMS_SCORE_MIN_PATH, and EMS_SCORE_MAX_PATH environment variables.
