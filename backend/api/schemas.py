from pydantic import BaseModel
from typing import List

class Features(BaseModel):
    features: List[float]

class Prediction(BaseModel):
    prediction: int