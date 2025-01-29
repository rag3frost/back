from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from pathlib import Path

app = FastAPI()

# Get port from environment variable (for Railway)
port = int(os.environ.get("PORT", 8000))

# Function to safely load model files
def load_model_files(model_prefix):
    try:
        base_path = Path("models")
        files = {
            "model": joblib.load(base_path / f"{model_prefix}_model.pkl"),
            "scaler": joblib.load(base_path / f"{model_prefix}_scaler.pkl"),
            "le_gender": joblib.load(base_path / f"{model_prefix}_le_gender.pkl"),
            "le_smoking": joblib.load(base_path / f"{model_prefix}_le_smoking.pkl"),
            "features": joblib.load(base_path / f"{model_prefix}_features.pkl")
        }
        if model_prefix == "cancer":
            files.update({
                "le_genetic_risk": joblib.load(base_path / "cancer_le_genetic_risk.pkl"),
                "le_activity": joblib.load(base_path / "cancer_le_activity.pkl"),
                "le_alcohol": joblib.load(base_path / "cancer_le_alcohol.pkl"),
                "le_cancer_history": joblib.load(base_path / "cancer_le_cancer_history.pkl")
            })
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading {model_prefix} model files: {str(e)}")

# Load model files
diabetes_files = load_model_files("diabetes")
cancer_files = load_model_files("cancer")

# Extract models and preprocessing tools
diabetes_model = diabetes_files["model"]
diabetes_scaler = diabetes_files["scaler"]
diabetes_le_gender = diabetes_files["le_gender"]
diabetes_le_smoking = diabetes_files["le_smoking"]
diabetes_features = diabetes_files["features"]

cancer_model = cancer_files["model"]
cancer_scaler = cancer_files["scaler"]
cancer_le_gender = cancer_files["le_gender"]
cancer_le_smoking = cancer_files["le_smoking"]
cancer_le_genetic_risk = cancer_files["le_genetic_risk"]
cancer_le_activity = cancer_files["le_activity"]
cancer_le_alcohol = cancer_files["le_alcohol"]
cancer_le_cancer_history = cancer_files["le_cancer_history"]
cancer_features = cancer_files["features"]


# Request Model for Diabetes Prediction
class DiabetesRequest(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float


@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesRequest):
    # Encode categorical variables
    gender_encoded = diabetes_le_gender.transform([data.gender])[0]
    smoking_encoded = diabetes_le_smoking.transform([data.smoking_history])[0]

    # Create input array
    input_data = np.array([[gender_encoded, data.age, data.hypertension, data.heart_disease, 
                             smoking_encoded, data.bmi, data.HbA1c_level, data.blood_glucose_level]])

    # Scale numerical features
    input_scaled = diabetes_scaler.transform(input_data)

    # Predict
    prediction = diabetes_model.predict(input_scaled)[0]

    return {"diabetes_prediction": int(prediction)}


# Request Model for Cancer Prediction
class CancerRequest(BaseModel):
    Gender: str
    Age: float
    BMI: float
    Smoking: str
    GeneticRisk: str
    PhysicalActivity: str
    AlcoholIntake: str
    CancerHistory: str


@app.post("/predict/cancer")
def predict_cancer(data: CancerRequest):
    # Encode categorical variables
    gender_encoded = cancer_le_gender.transform([data.Gender])[0]
    smoking_encoded = cancer_le_smoking.transform([data.Smoking])[0]
    genetic_risk_encoded = cancer_le_genetic_risk.transform([data.GeneticRisk])[0]
    activity_encoded = cancer_le_activity.transform([data.PhysicalActivity])[0]
    alcohol_encoded = cancer_le_alcohol.transform([data.AlcoholIntake])[0]
    cancer_history_encoded = cancer_le_cancer_history.transform([data.CancerHistory])[0]

    # Create input array
    input_data = np.array([[gender_encoded, data.Age, data.BMI, smoking_encoded, 
                             genetic_risk_encoded, activity_encoded, alcohol_encoded, cancer_history_encoded]])

    # Scale numerical features
    input_scaled = cancer_scaler.transform(input_data)

    # Predict
    prediction = cancer_model.predict(input_scaled)[0]

    return {"cancer_prediction": int(prediction)}


# Root Endpoint
@app.get("/")
def home():
    return {"message": "API is running. Use /predict/diabetes or /predict/cancer"}
