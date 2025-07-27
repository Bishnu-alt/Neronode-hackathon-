# backend/api/schemas.py - CORRECTED BMI TYPE

from pydantic import BaseModel, Field
from typing import List

class SingleFeatureInput(BaseModel):
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI: float  # <--- CORRECTED TO FLOAT
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int
    AnyHealthcare: int
    NoDocbcCost: int
    GenHlth: int
    MentHlth: float
    PhysHlth: float
    DiffWalk: int
    Sex: int
    Age: int
    Education: int
    Income: int

class Features(BaseModel):
    features: List[SingleFeatureInput]

class Prediction(BaseModel):
    prediction: int



# Pydantic models for metrics (already seem fine, just including for completeness)
class ClientMetric(BaseModel):
    client_id: int
    iteration: int
    model_name: str
    fit_status: str
    accuracy: float
    loss: float
    macro_f1: float
    recall_minority: float
    f1_minority: float
    f1_majority: float

class GlobalMetric(BaseModel):
    iteration: int
    accuracy: float
    loss: float
    macro_f1: float
    recall_minority: float
    f1_minority: float
    f1_majority: float

class MetricsResponse(BaseModel):
    global_metrics: List[GlobalMetric]
    client_metrics: List[ClientMetric]