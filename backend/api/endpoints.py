from fastapi import APIRouter, HTTPException
from backend.api.schemas import Features, Prediction # Assuming these are correctly defined
from backend.db.connection import get_db_connection
from backend.model.fetch import fetch_global_model
import pandas as pd
import numpy as np

router = APIRouter()

# --- Prediction Endpoint (Assuming this is correct and functional) ---
try:
    model = fetch_global_model(model_id=1)
except Exception as e:
    raise RuntimeError(f"Error loading global model from DB: {e}")

COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

@router.post("/", response_model=Prediction)
def predict_diabetes(data: Features):
    if len(data.features) != len(COLUMNS):
        raise HTTPException(status_code=400, detail=f"Expected {len(COLUMNS)} features, but got {len(data.features)}")

    # Create DataFrame
    df = pd.DataFrame([data.features], columns=COLUMNS)

    # Add engineered features
    df["Age_BMI"] = df["Age"] * df["BMI"]
    df["BP_Chol"] = df["HighBP"] + df["HighChol"]
    df["ChronicRiskScore"] = df["Stroke"] + df["HeartDiseaseorAttack"] + df["DiffWalk"]
    df["HealthCombo"] = df["GenHlth"] + df["PhysHlth"] + df["MentHlth"]

    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Prediction service unavailable: Global model not loaded.")
        prediction = int(model.predict(df)[0])
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# --- Metrics Endpoint (Crucially Corrected) ---
@router.get("/metrics/")
def get_metrics():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT client_id, round_num AS iteration, model_name, fit_status,
                   accuracy, loss, macro_f1, recall_minority,
                   f1_minority, f1_majority
            FROM client_updates
            ORDER BY iteration
        """)


        records = cursor.fetchall()
        df = pd.DataFrame(records)

        if df.empty:
            return {"global": [], "clients": []}

        # Identify numeric columns for global aggregation
        numeric_cols = [
            col for col in df.columns
            if col not in ['client_id', 'iteration', 'model_name', 'fit_status'] and pd.api.types.is_numeric_dtype(df[col])
        ]

        global_metrics = df.groupby("iteration")[numeric_cols].mean().reset_index()
        client_metrics = df.copy()

        return {
            "global": global_metrics.to_dict(orient="records"),
            "clients": client_metrics.to_dict(orient="records")
        }

    except Exception as e:
        import traceback
        print(f"Error fetching metrics in FastAPI: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {e}")