from pydantic import BaseModel
from typing import List

class Features(BaseModel):
    features: List[float]

class Prediction(BaseModel):
    prediction: int

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