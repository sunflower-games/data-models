# ROLLING MODEL USAGE
import pickle, json, pandas as pd

# Load config
config = json.load(open('config.json'))
features = config['features']

# Load model
with open('models/APP_IOS_NUVEI_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/APP_IOS_NUVEI_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Predict (data must have all 12 rolling features)
X = data[features].fillna(1.0)
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)  # -1 = anomaly
scores = model.decision_function(X_scaled)

# Check results
anomalies = (predictions == -1)
print(f"Anomalies: {anomalies.sum()}/{len(predictions)}")
