# backend/main.py (Example structure)

from fastapi import FastAPI
from backend.api import endpoints # Import your endpoints router

app = FastAPI(
    title="Neronode Federated Learning API",
    description="API for federated learning model predictions and metrics.",
    version="0.1.0"
)

# Include the router
app.include_router(endpoints.router)

# Optional: Add a root endpoint for health check
@app.get("/")
async def root():
    return {"message": "Neronode API is running!"}