# Random Forest Model API — Harness CI starter

This project wraps the existing `random_forest_model.pkl` in a small FastAPI application. It does not retrain the model.

## Model inputs

The model expects five numeric features, in this order:

- `Open`
- `High`
- `Low`
- `Adj Close`
- `Volume`

The target's meaning and units depend on the dataset used to train the model.

## Run locally

Use Python 3.11 or 3.12, then run:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` to try the API, or check `http://localhost:8000/health`.

Example POST body for `/predict`:

```json
{
  "Open": 100.0,
  "High": 105.0,
  "Low": 98.0,
  "Adj Close": 102.0,
  "Volume": 1000000
}
```

## Run tests / Harness CI

```bash
pip install -r requirements.txt
pytest -q
```

In the Harness CI build step, run those same commands. Ensure the pipeline checks out this repository before running the commands.

## Deployment note

CI verifies the code; it does not by itself make the API publicly available. To serve live predictions, deploy this app to a supported runtime/host. A container is one common option, but the runtime depends on the deployment target you choose.

## Important

Only load pickle/joblib files you trust. The model was saved using scikit-learn 1.6.1, so the requirements pin that version for compatibility.
