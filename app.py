from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"

# Load the trained model once when the application starts.
model = joblib.load(MODEL_PATH)
FEATURES = ["Open", "High", "Low", "Adj Close", "Volume"]

app = FastAPI(
    title="Random Forest Prediction App",
    description="Web form and API for a trained Random Forest regression model.",
    version="1.1.0",
)

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Random Forest Prediction</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; padding: 32px 16px; background: #f4f7fb; color: #172033; font-family: Arial, sans-serif; }
    main { max-width: 680px; margin: 0 auto; background: white; padding: 28px; border-radius: 18px; box-shadow: 0 10px 35px rgba(20,35,70,.10); }
    h1 { margin-top: 0; font-size: 28px; }
    p { color: #526078; line-height: 1.5; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    label { display: block; font-size: 14px; font-weight: 700; margin-bottom: 7px; }
    input { width: 100%; border: 1px solid #cbd3e1; border-radius: 9px; padding: 12px; font-size: 16px; }
    input:focus { outline: 2px solid #9dbbff; border-color: #4779e6; }
    button { margin-top: 22px; width: 100%; border: 0; border-radius: 10px; padding: 14px; background: #245fe8; color: white; font-size: 16px; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .65; cursor: wait; }
    #result { margin-top: 20px; padding: 16px; border-radius: 10px; background: #edf4ff; display: none; overflow-wrap: anywhere; }
    #result.error { background: #fff0f0; color: #9c2020; }
    .note { font-size: 12px; margin-top: 18px; }
    @media (max-width: 520px) { .grid { grid-template-columns: 1fr; } main { padding: 22px; } }
  </style>
</head>
<body>
<main>
  <h1>Random Forest Prediction</h1>
  <p>Enter the five feature values below, then select <b>Predict</b> to get a result from the trained model.</p>
  <form id="prediction-form">
    <div class="grid">
      <div><label for="open">Open</label><input id="open" name="Open" type="number" step="any" value="100" required></div>
      <div><label for="high">High</label><input id="high" name="High" type="number" step="any" value="105" required></div>
      <div><label for="low">Low</label><input id="low" name="Low" type="number" step="any" value="98" required></div>
      <div><label for="adj-close">Adj Close</label><input id="adj-close" name="Adj Close" type="number" step="any" value="102" required></div>
      <div><label for="volume">Volume</label><input id="volume" name="Volume" type="number" step="any" min="0" value="1000000" required></div>
    </div>
    <button id="predict-button" type="submit">Predict</button>
  </form>
  <div id="result" role="status" aria-live="polite"></div>
  <p class="note">Example values are for demonstration only. Model predictions depend on the data and units used during training.</p>
</main>
<script>
const form = document.getElementById('prediction-form');
const button = document.getElementById('predict-button');
const result = document.getElementById('result');
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  button.disabled = true;
  button.textContent = 'Predicting…';
  result.style.display = 'none';
  result.className = '';
  const data = {
    'Open': Number(form.elements['Open'].value),
    'High': Number(form.elements['High'].value),
    'Low': Number(form.elements['Low'].value),
    'Adj Close': Number(form.elements['Adj Close'].value),
    'Volume': Number(form.elements['Volume'].value)
  };
  try {
    const response = await fetch('/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Prediction failed');
    result.textContent = 'Predicted value: ' + Number(payload.prediction).toLocaleString(undefined, {maximumFractionDigits: 6});
    result.style.display = 'block';
  } catch (error) {
    result.textContent = 'Error: ' + error.message;
    result.className = 'error';
    result.style.display = 'block';
  } finally {
    button.disabled = false;
    button.textContent = 'Predict';
  }
});
</script>
</body>
</html>"""


class PredictionRequest(BaseModel):
    Open: float = Field(..., description="Opening price")
    High: float = Field(..., description="Highest price")
    Low: float = Field(..., description="Lowest price")
    adj_close: float = Field(..., alias="Adj Close", description="Adjusted closing price")
    Volume: float = Field(..., ge=0, description="Trading volume")

    model_config = {"populate_by_name": True}


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


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
