import joblib
import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent

paths = [
    base / 'model.pkl',
    base / 'scaler.pkl',
    base / 'score_min.pkl',
    base / 'score_max.pkl',
    base / 'day_encoder.pkl',
]
print([(p.name, p.exists()) for p in paths])

enc = joblib.load(base / 'day_encoder.pkl')
print('encoder type:', type(enc))
print('encoder object:', enc)
if hasattr(enc, 'transform'):
    print('encoder classes:', getattr(enc, 'classes_', None))

scaler = joblib.load(base / 'scaler.pkl')
print('scaler feature_names_in_', getattr(scaler, 'feature_names_in_', None))
print('scaler mean', getattr(scaler, 'mean_', None))
print('scaler scale', getattr(scaler, 'scale_', None))

model = joblib.load(base / 'model.pkl')
print('model type:', type(model))

sample = {'day': 'Monday', 'hour': 9, 'Living': 0, 'Kitchen': 0, 'Bathroom': 0, 'is_weekend': 0}
if hasattr(enc, 'transform'):
    day_val = enc.transform([sample['day']])[0]
elif isinstance(enc, dict):
    day_val = enc[sample['day']]
else:
    raise RuntimeError('Unsupported encoder type')
print('day_val', day_val)

feat = pd.DataFrame([{
    'day': day_val,
    'hour': sample['hour'],
    'Living': sample['Living'],
    'Kitchen': sample['Kitchen'],
    'Bathroom': sample['Bathroom'],
    'is_weekend': sample['is_weekend'],
}])
print('feature row:', feat.to_dict(orient='records'))
scaled = scaler.transform(feat)
print('scaled', scaled.tolist())
pred = model.predict(scaled)
score = model.decision_function(scaled)
print('pred', pred.tolist())
print('score', score.tolist())
print('score_min', joblib.load(base / 'score_min.pkl'))
print('score_max', joblib.load(base / 'score_max.pkl'))
