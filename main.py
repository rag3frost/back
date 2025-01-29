from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from pathlib import Path

app = FastAPI()

# Get port from environment variable (for Railway)
port = int(os.environ.get("PORT", 8000))

# Function to safely encode categorical variables
def safe_transform(encoder, value, feature_name):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        valid_values = encoder.classes_.tolist()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value for {feature_name}. Valid values are: {valid_values}"
        )

# Function to safely load model files
def load_model_files(model_path: Path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model file {model_path.name}: {str(e)}"
        )

# Load all model files
models_path = Path("models")

# Load diabetes model files
diabetes_model = load_model_files(models_path / "diabetes_model.pkl")
diabetes_scaler = load_model_files(models_path / "diabetes_scaler.pkl")
diabetes_le_gender = load_model_files(models_path / "diabetes_le_gender.pkl")
diabetes_le_smoking = load_model_files(models_path / "diabetes_le_smoking.pkl")
diabetes_features = load_model_files(models_path / "diabetes_features.pkl")

# Load cancer model files
cancer_model = load_model_files(models_path / "cancer_model.pkl")
cancer_scaler = load_model_files(models_path / "cancer_scaler.pkl")
cancer_le_gender = load_model_files(models_path / "cancer_le_gender.pkl")
cancer_le_smoking = load_model_files(models_path / "cancer_le_smoking.pkl")
cancer_le_genetic_risk = load_model_files(models_path / "cancer_le_genetic_risk.pkl")
cancer_le_activity = load_model_files(models_path / "cancer_le_activity.pkl")
cancer_le_alcohol = load_model_files(models_path / "cancer_le_alcohol.pkl")
cancer_le_cancer_history = load_model_files(models_path / "cancer_le_cancer_history.pkl")
cancer_features = load_model_files(models_path / "cancer_features.pkl")

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
    try:
        # Encode categorical variables with validation
        gender_encoded = safe_transform(diabetes_le_gender, data.gender, "gender")
        smoking_encoded = safe_transform(diabetes_le_smoking, data.smoking_history, "smoking_history")

        # Create input array
        input_data = np.array([[gender_encoded, data.age, data.hypertension, data.heart_disease, 
                               smoking_encoded, data.bmi, data.HbA1c_level, data.blood_glucose_level]])

        # Scale numerical features
        input_scaled = diabetes_scaler.transform(input_data)

        # Predict
        prediction = diabetes_model.predict(input_scaled)[0]

        return {"diabetes_prediction": int(prediction)}
    except Exception as e:
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        raise

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
    try:
        # Encode categorical variables with validation
        gender_encoded = safe_transform(cancer_le_gender, data.Gender, "Gender")
        smoking_encoded = safe_transform(cancer_le_smoking, data.Smoking, "Smoking")
        genetic_risk_encoded = safe_transform(cancer_le_genetic_risk, data.GeneticRisk, "GeneticRisk")
        activity_encoded = safe_transform(cancer_le_activity, data.PhysicalActivity, "PhysicalActivity")
        alcohol_encoded = safe_transform(cancer_le_alcohol, data.AlcoholIntake, "AlcoholIntake")
        cancer_history_encoded = safe_transform(cancer_le_cancer_history, data.CancerHistory, "CancerHistory")

        # Create input array
        input_data = np.array([[gender_encoded, data.Age, data.BMI, smoking_encoded, 
                               genetic_risk_encoded, activity_encoded, alcohol_encoded, cancer_history_encoded]])

        # Scale numerical features
        input_scaled = cancer_scaler.transform(input_data)

        # Predict
        prediction = cancer_model.predict(input_scaled)[0]

        return {"cancer_prediction": int(prediction)}
    except Exception as e:
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        raise

# Root Endpoint
@app.get("/")
def home():
    return {"message": "API is running. Use /predict/diabetes or /predict/cancer"}
