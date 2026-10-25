from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"

# Load the model once when the API starts.
model = joblib.load(MODEL_PATH)
FEATURES = ["Open", "High", "Low", "Adj Close", "Volume"]

app = FastAPI(
    title="Random Forest Prediction API",
    description="Serve predictions from the supplied trained Random Forest regression model.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    Open: float = Field(..., description="Opening price")
    High: float = Field(..., description="Highest price")
    Low: float = Field(..., description="Lowest price")
    adj_close: float = Field(..., alias="Adj Close", description="Adjusted closing price")
    Volume: float = Field(..., ge=0, description="Trading volume")

    model_config = {"populate_by_name": True}


@app.get("/")
def root():
    return {"message": "Random Forest prediction API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        row = pd.DataFrame(
            [[request.Open, request.High, request.Low, request.adj_close, request.Volume]],
            columns=FEATURES,
        )
        prediction = float(model.predict(row)[0])
        return {"prediction": prediction}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
