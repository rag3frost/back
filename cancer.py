import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("C:\Projects\JHU\dataset\The_Cancer_data_1500_V2.csv")

# Define feature columns
feature_columns = [
    "Gender", "Age", "BMI", "Smoking", "GeneticRisk", 
    "PhysicalActivity", "AlcoholIntake", "CancerHistory"
]

# Initialize label encoders for categorical variables
le_gender = LabelEncoder()
le_smoking = LabelEncoder()
le_genetic_risk = LabelEncoder()
le_activity = LabelEncoder()
le_alcohol = LabelEncoder()
le_cancer_history = LabelEncoder()

# Transform categorical variables
df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Smoking"] = le_smoking.fit_transform(df["Smoking"])
df["GeneticRisk"] = le_genetic_risk.fit_transform(df["GeneticRisk"])
df["PhysicalActivity"] = le_activity.fit_transform(df["PhysicalActivity"])
df["AlcoholIntake"] = le_alcohol.fit_transform(df["AlcoholIntake"])
df["CancerHistory"] = le_cancer_history.fit_transform(df["CancerHistory"])

# Split features and target
X = df[feature_columns]
y = df["Diagnosis"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and preprocessing tools
joblib.dump(model, "models/cancer_model.pkl")
joblib.dump(scaler, "models/cancer_scaler.pkl")
joblib.dump(le_gender, "models/cancer_le_gender.pkl")
joblib.dump(le_smoking, "models/cancer_le_smoking.pkl")
joblib.dump(le_genetic_risk, "models/cancer_le_genetic_risk.pkl")
joblib.dump(le_activity, "models/cancer_le_activity.pkl")
joblib.dump(le_alcohol, "models/cancer_le_alcohol.pkl")
joblib.dump(le_cancer_history, "models/cancer_le_cancer_history.pkl")
joblib.dump(feature_columns, "models/cancer_features.pkl")

# Print model accuracy
print(f"Cancer Model Accuracy: {model.score(X_test_scaled, y_test):.4f}")
