# backend/api/endpoints.py

from fastapi import APIRouter, HTTPException
# Import your Pydantic schemas from the schemas.py file
from backend.api.schemas import Features, Prediction, SingleFeatureInput, MetricsResponse, ClientMetric, GlobalMetric # Make sure to import all used schemas

# Assuming backend.db.connection and backend.model.fetch are correctly set up
from backend.db.connection import get_db_connection
from backend.model.fetch import fetch_global_model
import pandas as pd
import numpy as np
from typing import List
import sys

router = APIRouter()

# --- Model Loading ---
try:
    # Ensure this model ID and version correctly point to a model trained ONLY on raw features
    model = fetch_global_model(model_id=1, version=3) # Confirmed version 3 is the one in use

    # --- DEBUGGING START: Inspect the loaded model's expected features ---
    # This block will now likely confirm only the 21 raw features are expected.
    print("\n--- DEBUG: Inspecting Loaded Model's Expected Feature Names ---", file=sys.stderr)
    expected_features_from_model = None
    try:
        if hasattr(model, 'named_steps'):
            for name, step in model.named_steps.items():
                if hasattr(step, 'get_feature_names_out'):
                    expected_features_from_model = step.get_feature_names_out().tolist()
                    print(f"DEBUG: Found preprocessor '{name}' with get_feature_names_out:", file=sys.stderr)
                    break
                elif hasattr(step, 'feature_names_in_'):
                     expected_features_from_model = step.feature_names_in_.tolist()
                     print(f"DEBUG: Found step '{name}' with feature_names_in_:", file=sys.stderr)
                     break
        if expected_features_from_model is None and hasattr(model, 'feature_names_in_'):
            expected_features_from_model = model.feature_names_in_.tolist()
            print("DEBUG: Model (estimator) has feature_names_in_:", file=sys.stderr)

        if expected_features_from_model:
            print("DEBUG: Model expects these final features (post-preprocessing/direct input):", file=sys.stderr)
            print(expected_features_from_model, file=sys.stderr)
            print(f"DEBUG: Total expected features by model: {len(expected_features_from_model)}", file=sys.stderr)
        else:
            print("DEBUG: Could not automatically determine model's expected feature names. Proceeding with hardcoded list.", file=sys.stderr)
    except Exception as debug_e_model_inspect:
        print(f"DEBUG: Error during model feature inspection: {debug_e_model_inspect}", file=sys.stderr)
    print("----------------------------------------------------------\n", file=sys.stderr)
    # --- DEBUGGING END: Inspect the loaded model's expected features ---

except Exception as e:
    raise RuntimeError(f"Error loading global model from DB: {e}")

# --- Define the raw input columns, matching the order in SingleFeatureInput ---
RAW_INPUT_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

# --- Define the FINAL list of features that the model was trained ON ---
# <<< IMPORTANT CHANGE HERE: ONLY RAW FEATURES >>>
MODEL_TRAINING_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]
# Total expected features for the model = 21 (raw features).


# --- Prediction Endpoint ---
@router.post("/", response_model=Prediction)
def predict_diabetes(data: Features):
    if not data.features:
        raise HTTPException(status_code=400, detail="No features provided for prediction.")

    list_of_input_dicts = [item.model_dump() for item in data.features]

    df_raw = pd.DataFrame(list_of_input_dicts)[RAW_INPUT_COLUMNS]

    # <<< IMPORTANT CHANGE HERE: REMOVE FEATURE ENGINEERING >>>
    # df_processed = df_raw.copy() # No need for df_processed if no new features are created
    # df_processed["Age_BMI"] = df_processed["Age"] * df_processed["Age"] # Original typo was here!
    # df_processed["BP_Chol"] = df_processed["HighBP"] + df_processed["HighChol"]
    # df_processed["ChronicRiskScore"] = df_processed["Stroke"] + df_processed["HeartDiseaseorAttack"] + df_processed["DiffWalk"]
    # df_processed["HealthCombo"] = df_processed["GenHlth"] + df_processed["PhysHlth"] + df_processed["MentHlth"]

    # 3. CRUCIAL: Reindex the DataFrame to match the model's exact training features and order
    try:
        # Since no new features are created, df_raw is now the source of df_final_input
        df_final_input = df_raw[MODEL_TRAINING_FEATURES] # This will select raw columns in the correct order
    except KeyError as ke:
        import traceback
        print(f"--- Feature Missing/Mismatch Error Traceback ---\n{traceback.format_exc()}", file=sys.stderr)
        missing_columns = set(MODEL_TRAINING_FEATURES) - set(df_raw.columns) # Changed df_processed to df_raw
        extra_columns = set(df_raw.columns) - set(MODEL_TRAINING_FEATURES)   # Changed df_processed to df_raw
        raise HTTPException(status_code=500,
                            detail=f"Prediction error: Feature mismatch or incorrect order. "
                                   f"Missing in input for model: {list(missing_columns)}. "
                                   f"Extra in input from client: {list(extra_columns)}. "
                                   f"DataFrame columns before reindexing: {df_raw.columns.tolist()}." # Changed df_processed to df_raw
                                   f"Model expected columns (MODEL_TRAINING_FEATURES): {MODEL_TRAINING_FEATURES}. "
                                   f"Check server logs for details.")
    except Exception as e_reindex:
        import traceback
        print(f"--- Reindexing Error Traceback ---\n{traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Prediction error during reindexing: {e_reindex}. "
                                                    f"DataFrame columns before reindexing: {df_raw.columns.tolist()}." # Changed df_processed to df_raw
                                                    f"Model expected columns (MODEL_TRAINING_FEATURES): {MODEL_TRAINING_FEATURES}. "
                                                    f"Check server logs for details.")

    # --- DEBUGGING START: Print DataFrame columns being sent to model ---
    print(f"\n--- DEBUG: DataFrame columns being sent to model ---", file=sys.stderr)
    print(df_final_input.columns.tolist(), file=sys.stderr)
    print(f"DEBUG: Total DataFrame columns: {len(df_final_input.columns.tolist())}", file=sys.stderr)
    # Added the data row print back, as it's useful for debugging the model's actual output
    print("DEBUG: Data row being sent to model:", file=sys.stderr)
    print(df_final_input.iloc[0].to_dict(), file=sys.stderr)
    print("-----------------------------------------------------\n", file=sys.stderr)
    # --- DEBUGGING END ---

    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Prediction service unavailable: Global model not loaded.")

        prediction = int(model.predict(df_final_input)[0])
        return {"prediction": prediction}
    except Exception as e:
        import traceback
        print(f"--- Prediction Error Traceback ---\n{traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(status_code=500,
                            detail=f"Prediction error: {e}. "
                                   f"Final DataFrame columns sent: {df_final_input.columns.tolist()}. "
                                   f"Check server logs for full traceback.")


# --- Metrics Endpoint (No changes needed here for this issue) ---
@router.get("/metrics/", response_model=MetricsResponse)
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
            return {"global_metrics": [], "client_metrics": []}

        metric_columns = ['accuracy', 'loss', 'macro_f1', 'recall_minority', 'f1_minority', 'f1_majority']

        global_metrics = df.groupby("iteration")[metric_columns].mean().reset_index()
        client_metrics = df.copy()

        return {
            "global_metrics": global_metrics.to_dict(orient="records"),
            "client_metrics": client_metrics.to_dict(orient="records")
        }

    except Exception as e:
        import traceback
        print(f"Error fetching metrics in FastAPI: {e}\n{traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {e}")