#!/usr/bin/env python3

import os
import sys
import subprocess
import json
from google.cloud import aiplatform, storage
from google.cloud.aiplatform.prediction import LocalModel

# %%
# Create deployment directory
os.makedirs("anomaly_deployment", exist_ok=True)

# %%
# Change to deployment directory
try:
    os.chdir("anomaly_deployment")
    print(f"Changed to directory: {os.getcwd()}")
except Exception as e:
    print(f"Error changing directory: {e}")

# %%
# Configuration variables
PROJECT_ID = "crowncoins-prod-lake-server"
REGION = "us-central1"

# Updated for ML anomaly detection models
MODEL_ARTIFACT_DIR = "gs://ds-models-bucket/anomaly-detection/ml_anomaly_model/"
REPOSITORY = "ds-models-repo"
IMAGE = "ml-anomaly-detection-predictor"  # Updated for ML models
MODEL_DISPLAY_NAME = "ml-anomaly-detection-model"  # Updated for ML models

# Bucket configuration
BUCKET_NAME = "ds-models-bucket"
BUCKET_URI = f"gs://{BUCKET_NAME}"
MODEL_DIRECTORY = "anomaly-detection/ml_anomaly_model"  # Updated to match bucket structure

# Local development directories
USER_SRC_DIR = "src_dir_sdk"  # Updated for ML models
LOCAL_MODEL_ARTIFACTS_DIR = "model_artifacts"  # Updated for ML models

# %%
# Create necessary directories
os.makedirs("src_dir_sdk", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("model_artifacts", exist_ok=True)

# %%
# Create requirements.txt
requirements_content = """
# Core ML and Data Processing
pandas>=1.3.0,<2.0.0
numpy~=1.20
joblib~=1.0
scikit-learn==1.4.0

# API Framework
fastapi
uvicorn==0.17.6
pydantic>=1.8.0,<2.0.0

# Google Cloud Services
google-cloud-storage>=1.26.0,<2.0.0
google-cloud-aiplatform[prediction]>=1.16.0
google-cloud-logging>=2.0.0,<4.0.0

# Monitoring
prometheus-client>=0.13.0

# Data Serialization
pickle-mixin
"""

with open("src_dir_sdk/requirements.txt", "w") as f:
    f.write(requirements_content.strip())

print("Created requirements.txt")

# %%
# Create predictor.py
predictor_content = '''import numpy as np
import pickle
import json
import pandas as pd
import os
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils

class MLAnomalyPredictor(Predictor):

    def __init__(self):
        self._models = {}
        self._scalers = {}
        self._config = None

    def load(self, artifacts_uri: str):
        """Load ML models with contamination rates."""
        prediction_utils.download_model_artifacts(artifacts_uri)

        downloaded_files = os.listdir(".")
        print(f"Downloaded files: {downloaded_files}")

        # Load config
        if "config.json" not in downloaded_files:
            raise Exception(f"config.json not found. Available: {downloaded_files}")

        with open("config.json", "r") as f:
            self._config = json.load(f)

        # Load models and scalers
        models_dir = "models"
        if models_dir not in downloaded_files:
            raise Exception(f"models/ directory not found")

        model_files = os.listdir(models_dir)
        platforms_loaded = 0

        for platform in self._config["platforms"]:
            model_file = f"{platform}_model.pkl"
            scaler_file = f"{platform}_scaler.pkl"

            if model_file in model_files and scaler_file in model_files:
                with open(f"{models_dir}/{model_file}", "rb") as f:
                    self._models[platform] = pickle.load(f)
                with open(f"{models_dir}/{scaler_file}", "rb") as f:
                    self._scalers[platform] = pickle.load(f)
                platforms_loaded += 1
            else:
                print(f"⚠️ Missing files for {platform}")

        print(f"✅ Loaded {platforms_loaded} platforms with {len(self._config['ml_features'])} features")

    def _engineer_features(self, df):
        """Engineer 14 ML features - ALIGNED with training code."""
        df = df.copy()

        # Sort by hour first
        df['hour'] = pd.to_datetime(df['hour'])
        df = df.sort_values('hour').reset_index(drop=True)

        # CORE METRICS
        df['success_rate'] = df['approved'] / df['total'].replace(0, 1)

        # TIME CONTEXT - FIXED: Use correct business hours (15-3)
        df['hour_of_day'] = df['hour'].dt.hour
        df['is_business_hours'] = ((df['hour_of_day'] >= 15) | (df['hour_of_day'] <= 3)).astype(int)

        # CYCLICAL TIME ENCODING
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

        # DAY OF WEEK PATTERNS
        df['day_of_week'] = df['hour'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # ROLLING FEATURES - FIXED: Use correct window sizes
        # 3-hour volatility (12 periods of 15min data)
        df['success_volatility'] = df['success_rate'].rolling(12, min_periods=1).std().fillna(0)
        df['volume_volatility'] = df['total'].rolling(12, min_periods=1).std().fillna(0)

        # Create volume moving averages first
        df['volume_ma_24h'] = df['total'].rolling(96, min_periods=1).mean().fillna(df['total'].mean())
        df['volume_ma_3h'] = df['total'].rolling(12, min_periods=1).mean().fillna(df['total'].mean())

        # Performance trends (6-hour slopes = 24 periods)
        df['success_trend'] = df['success_rate'].rolling(24, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=False
        ).fillna(0)

        # ML EARLY WARNING INDICATORS
        df['success_warning'] = ((df['success_rate'] < 0.75) & (df['total'] >= 100)).astype(int)
        df['volume_warning'] = (df['total'] < (df['volume_ma_24h'] * 0.4)).fillna(False).astype(int)

        # RELATIVE PERFORMANCE
        df['success_relative'] = (df['success_rate'] / df['success_rate'].rolling(96, min_periods=1).mean().replace(0, 1)).fillna(1)
        df['volume_relative'] = (df['total'] / df['volume_ma_24h'].replace(0, 1)).fillna(1)

        # Clean data - handle inf/nan
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['hour', 'platform_provider', 'day_of_week']
        for col in numeric_cols:
            if col not in exclude_cols:
                df[col] = df[col].fillna(0)
                df[col] = df[col].replace([np.inf, -np.inf], 0)

        return df

    def predict(self, instances):
        """Predict anomalies - returns only most recent (excluding max hour)."""
        platform_provider = instances.get("platform_provider")

        if not platform_provider or platform_provider not in self._models:
            return {"error": f"Platform {platform_provider} not available. Available: {list(self._models.keys())}"}

        # Convert to DataFrame
        raw_data = instances["instances"]
        df = pd.DataFrame(raw_data)

        # Validate required fields
        required_fields = ["approved", "total", "hour"]
        if not all(field in df.columns for field in required_fields):
            return {"error": f"Missing required fields: {required_fields}"}

        # Sort by hour and remove max hour (most recent timestamp)
        df['hour'] = pd.to_datetime(df['hour'])
        df = df.sort_values('hour')

        # Remove the max hour (most recent record)
        max_hour = df['hour'].max()
        df_filtered = df[df['hour'] < max_hour].copy()

        if len(df_filtered) == 0:
            return {"error": "No data available after excluding max hour"}

        # Engineer features on ALL filtered data (needed for rolling calculations)
        df_features = self._engineer_features(df_filtered)

        # Get most recent record AFTER feature engineering
        latest_record = df_features.iloc[-1:].copy()

        # Extract ML features in correct order
        ml_features = self._config["ml_features"]
        X = latest_record[ml_features].values

        # Handle inf/nan (double check)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale and predict
        X_scaled = self._scalers[platform_provider].transform(X)
        model = self._models[platform_provider]
        prediction = model.predict(X_scaled)[0]
        score = model.decision_function(X_scaled)[0]

        # Get contamination rate for context
        contamination_rate = self._config["contamination_rates"].get(platform_provider, 0.005)

        # Extract feature values for debugging (optional)
        feature_values = {feature: float(latest_record[feature].iloc[0]) for feature in ml_features}

        return {
            "platform": platform_provider,
            "is_anomaly": bool(prediction == -1),
            "anomaly_score": float(score),
            "confidence": float(abs(score)),
            "contamination_rate": float(contamination_rate),
            "prediction_timestamp": str(latest_record['hour'].iloc[0]),
            "records_analyzed": len(df_filtered),
            "max_hour_excluded": str(max_hour),
            "feature_values": feature_values  # For debugging
        }
'''

with open("src_dir_sdk/predictor.py", "w") as f:
    f.write(predictor_content)

print("Created predictor.py")

# %%
# Create test instances.json
instances_data = {
    "platform_provider": "PAYSAFE_SKRILL",
    "instances": [
        {"approved": 105, "total": 145, "hour": "2025-07-09T09:30:00"},
        {"approved": 87, "total": 123, "hour": "2025-07-09T09:45:00"},
        {"approved": 71, "total": 115, "hour": "2025-07-09T10:00:00"},
        {"approved": 50, "total": 78, "hour": "2025-07-09T10:15:00"},
        {"approved": 89, "total": 125, "hour": "2025-07-09T10:30:00"},
        {"approved": 76, "total": 116, "hour": "2025-07-09T10:45:00"},
        {"approved": 83, "total": 124, "hour": "2025-07-09T11:00:00"},
        {"approved": 78, "total": 113, "hour": "2025-07-09T11:15:00"},
        {"approved": 84, "total": 120, "hour": "2025-07-09T11:30:00"},
        {"approved": 63, "total": 89, "hour": "2025-07-09T11:45:00"},
        {"approved": 78, "total": 121, "hour": "2025-07-09T12:00:00"},
        {"approved": 82, "total": 120, "hour": "2025-07-09T12:15:00"},
        {"approved": 108, "total": 150, "hour": "2025-07-09T12:30:00"},
        {"approved": 89, "total": 133, "hour": "2025-07-09T12:45:00"},
        {"approved": 97, "total": 140, "hour": "2025-07-09T13:00:00"},
        {"approved": 98, "total": 133, "hour": "2025-07-09T13:15:00"},
        {"approved": 84, "total": 116, "hour": "2025-07-09T13:30:00"},
        {"approved": 89, "total": 124, "hour": "2025-07-09T13:45:00"},
        {"approved": 111, "total": 149, "hour": "2025-07-09T14:00:00"},
        {"approved": 79, "total": 119, "hour": "2025-07-09T14:15:00"},
        {"approved": 85, "total": 122, "hour": "2025-07-09T14:30:00"},
        {"approved": 102, "total": 134, "hour": "2025-07-09T14:45:00"},
        {"approved": 106, "total": 141, "hour": "2025-07-09T15:00:00"},
        {"approved": 70, "total": 105, "hour": "2025-07-09T15:15:00"},
        {"approved": 68, "total": 99, "hour": "2025-07-09T15:30:00"},
        {"approved": 83, "total": 103, "hour": "2025-07-09T15:45:00"},
        {"approved": 67, "total": 102, "hour": "2025-07-09T16:00:00"},
        {"approved": 74, "total": 114, "hour": "2025-07-09T16:15:00"},
        {"approved": 73, "total": 115, "hour": "2025-07-09T16:30:00"},
        {"approved": 60, "total": 95, "hour": "2025-07-09T16:45:00"},
        {"approved": 55, "total": 88, "hour": "2025-07-09T17:00:00"},
        {"approved": 59, "total": 90, "hour": "2025-07-09T17:15:00"},
        {"approved": 68, "total": 87, "hour": "2025-07-09T17:30:00"},
        {"approved": 49, "total": 92, "hour": "2025-07-09T17:45:00"},
        {"approved": 43, "total": 78, "hour": "2025-07-09T18:00:00"},
        {"approved": 37, "total": 60, "hour": "2025-07-09T18:15:00"},
        {"approved": 54, "total": 77, "hour": "2025-07-09T18:30:00"},
        {"approved": 96, "total": 130, "hour": "2025-07-09T18:45:00"},
        {"approved": 77, "total": 101, "hour": "2025-07-09T19:00:00"},
        {"approved": 67, "total": 84, "hour": "2025-07-09T19:15:00"},
        {"approved": 65, "total": 97, "hour": "2025-07-09T19:30:00"},
        {"approved": 65, "total": 94, "hour": "2025-07-09T19:45:00"},
        {"approved": 57, "total": 81, "hour": "2025-07-09T20:00:00"},
        {"approved": 69, "total": 87, "hour": "2025-07-09T20:15:00"},
        {"approved": 56, "total": 82, "hour": "2025-07-09T20:30:00"},
        {"approved": 60, "total": 84, "hour": "2025-07-09T20:45:00"},
        {"approved": 71, "total": 91, "hour": "2025-07-09T21:00:00"},
        {"approved": 71, "total": 89, "hour": "2025-07-09T21:15:00"},
        {"approved": 62, "total": 94, "hour": "2025-07-09T21:30:00"},
        {"approved": 53, "total": 84, "hour": "2025-07-09T21:45:00"},
        {"approved": 58, "total": 92, "hour": "2025-07-09T22:00:00"},
        {"approved": 55, "total": 86, "hour": "2025-07-09T22:15:00"},
        {"approved": 76, "total": 96, "hour": "2025-07-09T22:30:00"},
        {"approved": 72, "total": 92, "hour": "2025-07-09T22:45:00"},
        {"approved": 70, "total": 110, "hour": "2025-07-09T23:00:00"},
        {"approved": 63, "total": 101, "hour": "2025-07-09T23:15:00"},
        {"approved": 68, "total": 96, "hour": "2025-07-09T23:30:00"},
        {"approved": 83, "total": 116, "hour": "2025-07-09T23:45:00"},
        {"approved": 73, "total": 99, "hour": "2025-07-10T00:00:00"},
        {"approved": 81, "total": 103, "hour": "2025-07-10T00:15:00"},
        {"approved": 61, "total": 98, "hour": "2025-07-10T00:30:00"},
        {"approved": 93, "total": 130, "hour": "2025-07-10T00:45:00"},
        {"approved": 78, "total": 114, "hour": "2025-07-10T01:00:00"},
        {"approved": 55, "total": 85, "hour": "2025-07-10T01:15:00"},
        {"approved": 75, "total": 102, "hour": "2025-07-10T01:30:00"},
        {"approved": 61, "total": 98, "hour": "2025-07-10T01:45:00"},
        {"approved": 78, "total": 107, "hour": "2025-07-10T02:00:00"},
        {"approved": 60, "total": 86, "hour": "2025-07-10T02:15:00"},
        {"approved": 91, "total": 120, "hour": "2025-07-10T02:30:00"},
        {"approved": 108, "total": 141, "hour": "2025-07-10T02:45:00"},
        {"approved": 102, "total": 144, "hour": "2025-07-10T03:00:00"},
        {"approved": 94, "total": 125, "hour": "2025-07-10T03:15:00"},
        {"approved": 88, "total": 127, "hour": "2025-07-10T03:30:00"},
        {"approved": 85, "total": 131, "hour": "2025-07-10T03:45:00"},
        {"approved": 93, "total": 134, "hour": "2025-07-10T04:00:00"},
        {"approved": 88, "total": 138, "hour": "2025-07-10T04:15:00"},
        {"approved": 113, "total": 166, "hour": "2025-07-10T04:30:00"},
        {"approved": 92, "total": 130, "hour": "2025-07-10T04:45:00"},
        {"approved": 109, "total": 144, "hour": "2025-07-10T05:00:00"},
        {"approved": 93, "total": 135, "hour": "2025-07-10T05:15:00"},
        {"approved": 92, "total": 129, "hour": "2025-07-10T05:30:00"},
        {"approved": 100, "total": 134, "hour": "2025-07-10T05:45:00"},
        {"approved": 78, "total": 118, "hour": "2025-07-10T06:00:00"},
        {"approved": 104, "total": 148, "hour": "2025-07-10T06:15:00"},
        {"approved": 101, "total": 131, "hour": "2025-07-10T06:30:00"},
        {"approved": 86, "total": 131, "hour": "2025-07-10T06:45:00"},
        {"approved": 82, "total": 126, "hour": "2025-07-10T07:00:00"},
        {"approved": 76, "total": 118, "hour": "2025-07-10T07:15:00"},
        {"approved": 67, "total": 95, "hour": "2025-07-10T07:30:00"},
        {"approved": 70, "total": 107, "hour": "2025-07-10T07:45:00"},
        {"approved": 72, "total": 108, "hour": "2025-07-10T08:00:00"},
        {"approved": 76, "total": 102, "hour": "2025-07-10T08:15:00"},
        {"approved": 72, "total": 111, "hour": "2025-07-10T08:30:00"},
        {"approved": 63, "total": 103, "hour": "2025-07-10T08:45:00"},
        {"approved": 70, "total": 104, "hour": "2025-07-10T09:00:00"},
        {"approved": 75, "total": 94, "hour": "2025-07-10T09:15:00"},
        {"approved": 72, "total": 106, "hour": "2025-07-10T09:30:00"},
        {"approved": 90, "total": 135, "hour": "2025-07-10T09:45:00"},
        {"approved": 79, "total": 112, "hour": "2025-07-10T10:00:00"},
        {"approved": 80, "total": 130, "hour": "2025-07-10T10:15:00"},
        {"approved": 79, "total": 111, "hour": "2025-07-10T10:30:00"},
        {"approved": 73, "total": 108, "hour": "2025-07-10T10:45:00"},
        {"approved": 69, "total": 114, "hour": "2025-07-10T11:00:00"},
        {"approved": 82, "total": 117, "hour": "2025-07-10T11:15:00"},
        {"approved": 78, "total": 126, "hour": "2025-07-10T11:30:00"}
    ]
}

with open("instances.json", "w") as f:
    json.dump(instances_data, f, indent=2)

print("Created instances.json")


# %%
# Build local model and test
def build_and_test_local_model():
    """Build and test the local model container"""
    # Use absolute path to be sure
    current_dir = os.getcwd()
    src_dir = os.path.join(current_dir, "src_dir_sdk")

    print(f"Looking for predictor in: {src_dir}")
    print(f"predictor.py exists: {os.path.exists(os.path.join(src_dir, 'predictor.py'))}")

    # Add absolute path to Python path
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Now import
    from predictor import MLAnomalyPredictor

    # Build with absolute path
    local_model = LocalModel.build_cpr_model(
        src_dir,  # Use absolute path here too
        f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE}",
        predictor=MLAnomalyPredictor,
        requirements_path=os.path.join(src_dir, "requirements.txt"),
    )

    print("✅ Container built successfully!")
    return local_model


# Execute the function
local_model = build_and_test_local_model()


# %%
# Test with downloaded model artifacts
def test_with_model_artifacts(local_model):
    """Test the model with downloaded artifacts from GCS"""
    print("🚀 Testing with corrected GCS structure...")

    # Download the reorganized files first
    MODEL_ARTIFACT_DIR_LOCAL = "./downloaded_model_artifacts_fixed"
    os.makedirs(MODEL_ARTIFACT_DIR_LOCAL, exist_ok=True)

    # Download from GCS with new structure
    client = storage.Client()
    bucket = client.bucket("ds-models-bucket")
    blobs = bucket.list_blobs(prefix="anomaly-detection/ml_anomaly_model/")

    for blob in blobs:
        if not blob.name.endswith('/'):
            # Preserve the folder structure (models/ subfolder)
            relative_path = blob.name.replace("anomaly-detection/ml_anomaly_model/", "")
            local_path = os.path.join(MODEL_ARTIFACT_DIR_LOCAL, relative_path)

            # Create subdirectories
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            blob.download_to_filename(local_path)
            print(f"✅ Downloaded: {relative_path}")

    # Now test the container
    with local_model.deploy_to_local_endpoint(
            artifact_uri=MODEL_ARTIFACT_DIR_LOCAL,
    ) as local_endpoint:

        predict_response = local_endpoint.predict(
            request_file="instances.json",
            headers={"Content-Type": "application/json"},
        )

        result = json.loads(predict_response.content)
        print("🎯 SUCCESS!")
        print(json.dumps(result, indent=2))

    return MODEL_ARTIFACT_DIR_LOCAL


# Execute the function
test_result = test_with_model_artifacts(local_model)


# %%
# Push image to Google Container Registry
def push_image_to_gcr(local_model):
    """Push the container image to Google Container Registry"""
    # 1. Fix Docker auth (this is usually the only issue)
    subprocess.run([
        "gcloud", "auth", "configure-docker",
        "us-central1-docker.pkg.dev", "--quiet"
    ], check=False)

    # 2. Create repo if needed (ignore errors if it exists)
    result = subprocess.run([
        "gcloud", "artifacts", "repositories", "create", "ds-models-repo",
        "--repository-format=docker", "--location=us-central1",
        "--project=crowncoins-prod-lake-server"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("Repo might already exist")

    # 3. Push the image
    local_model.push_image()
    print("✅ Image pushed successfully!")


# Execute the function
push_image_to_gcr(local_model)

# %%
# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)
print("✅ Vertex AI initialized")


# %%
# Upload model to Vertex AI
def upload_model_to_vertex(local_model):
    """Upload the model to Vertex AI"""
    # Ensure we use the correct GCS path
    model_artifact_dir = MODEL_ARTIFACT_DIR
    if not model_artifact_dir.startswith("gs://"):
        model_artifact_dir = "gs://ds-models-bucket/anomaly-detection/ml_anomaly_model/"
        print(f"✅ Corrected to GCS path: {model_artifact_dir}")

    # Upload model
    model = aiplatform.Model.upload(
        local_model=local_model,
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=model_artifact_dir,
    )

    print(f"✅ Model uploaded: {model.resource_name}")
    return model


# Execute the function
model = upload_model_to_vertex(local_model)


# %%
# Deploy model to endpoint
def deploy_model_to_endpoint(model):
    """Deploy the model to a Vertex AI endpoint"""
    # Step 1: Create empty endpoint
    endpoint = aiplatform.Endpoint.create(display_name="ml-anomaly-endpoint")

    # Step 2: Deploy model to endpoint
    deployed_model = model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-8",
        min_replica_count=1,
        max_replica_count=3,
    )

    print(f"✅ Model deployed to endpoint: {endpoint.resource_name}")
    return endpoint


# Execute the function
endpoint = deploy_model_to_endpoint(model)

