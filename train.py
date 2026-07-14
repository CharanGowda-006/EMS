"""Training script for the EMS anomaly detection model."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ENCODER = {day: index for index, day in enumerate(DAYS)}

# Example training data placeholder. Replace with your real dataset.
training_data = pd.DataFrame(
    [
        {"day": "Monday", "hour": 14, "Living": 12, "Kitchen": 5, "Bathroom": 2, "is_weekend": 0},
        {"day": "Tuesday", "hour": 8, "Living": 10, "Kitchen": 3, "Bathroom": 1, "is_weekend": 0},
        {"day": "Saturday", "hour": 20, "Living": 15, "Kitchen": 8, "Bathroom": 2, "is_weekend": 1},
    ]
)

training_data["day"] = training_data["day"].map(DAY_ENCODER)
feature_frame = training_data[["day", "hour", "Living", "Kitchen", "Bathroom", "is_weekend"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_frame)

model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_scaled)

scores = model.decision_function(X_scaled)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(scores.min(), "score_min.pkl")
joblib.dump(scores.max(), "score_max.pkl")
joblib.dump(DAY_ENCODER, "day_encoder.pkl")

print("Training complete. Artifacts saved to disk.")
