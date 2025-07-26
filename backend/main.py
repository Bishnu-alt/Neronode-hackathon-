from fastapi import FastAPI
from backend.api.endpoints import router as predict_router

app = FastAPI(title="Diabetes Prediction API")

app.include_router(predict_router)