from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load Diabetes Model & Preprocessing Tools
diabetes_model = joblib.load("models\diabetes_model.pkl")
diabetes_scaler = joblib.load("models\diabetes_scaler.pkl")
diabetes_le_gender = joblib.load("models\diabetes_le_gender.pkl")
diabetes_le_smoking = joblib.load("models\diabetes_le_smoking.pkl")
diabetes_features = joblib.load("models\diabetes_features.pkl")

# Load Cancer Model & Preprocessing Tools
cancer_model = joblib.load("models\cancer_model.pkl")
cancer_scaler = joblib.load("models\cancer_scaler.pkl")
cancer_le_gender = joblib.load("models\cancer_le_gender.pkl")
cancer_le_smoking = joblib.load("models\cancer_le_smoking.pkl")
cancer_le_genetic_risk = joblib.load("models\cancer_le_genetic_risk.pkl")
cancer_le_activity = joblib.load("models\cancer_le_activity.pkl")
cancer_le_alcohol = joblib.load("models\cancer_le_alcohol.pkl")
cancer_le_cancer_history = joblib.load("models\cancer_le_cancer_history.pkl")
cancer_features = joblib.load("models\cancer_features.pkl")


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
