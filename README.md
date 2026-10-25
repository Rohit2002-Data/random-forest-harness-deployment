# Random Forest Prediction App

A small FastAPI application that loads `random_forest_model.pkl`, provides a browser form for entering five features, and exposes a JSON prediction API. The model is not retrained.

## Features

- Browser form at `/` for `Open`, `High`, `Low`, `Adj Close`, and `Volume`
- JSON prediction endpoint at `POST /predict`
- Health check at `GET /health`
- Automated tests with pytest

## Run locally

Use Python 3.11 or 3.12 and install the pinned dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 in a browser. Enter the five values and click **Predict**. API documentation is at http://127.0.0.1:8000/docs.

## Run tests

```bash
pytest -q
```

## Harness CI

The CI pipeline can install `requirements.txt` and run `pytest -q`. This validates the application; it does not publish a live service by itself. To make the page accessible to other users, deploy the app to a supported runtime environment and configure Harness CD for that target.

## Notes

- Keep `random_forest_model.pkl` beside `app.py`.
- Use only trusted `.pkl` files because deserializing pickle/joblib files can execute code.
- The example form values are illustrative. Use values and units consistent with the model's training data.
