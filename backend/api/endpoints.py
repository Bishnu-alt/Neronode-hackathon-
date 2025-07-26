from fastapi import APIRouter, HTTPException
from backend.api.schemas import Features, Prediction
from backend.db.connection import get_db_connection
from backend.model.fetch import fetch_global_model
import pandas as pd

router = APIRouter()

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
        raise HTTPException(status_code=400, detail=f"Expected {len(COLUMNS)} features")

    df = pd.DataFrame([data.features], columns=COLUMNS)
    try:
        prediction = int(model.predict(df)[0])
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@router.get("/metrics/")
def get_metrics():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT client_id, round_num AS iteration, 
                   accuracy, macro_f1, recall_minority, 
                   f1_minority, f1_majority
            FROM client_updates 
            ORDER BY iteration
        """)
        records = cursor.fetchall()
        df = pd.DataFrame(records)

        if df.empty:
            return {"global": [], "clients": []}

        global_metrics = df.groupby("iteration").mean(numeric_only=True).reset_index()
        client_metrics = df.copy()

        return {
            "global": global_metrics.to_dict(orient="records"),
            "clients": client_metrics.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {e}")

@router.get("/accuracy/clients")
def get_client_accuracy():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT client_id, round_num AS iteration, accuracy 
            FROM client_updates 
            ORDER BY round_num
        """)
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching client accuracy: {e}")
