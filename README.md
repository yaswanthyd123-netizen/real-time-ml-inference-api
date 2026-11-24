# RabTech Task 06 – Real-Time ML Inference REST API & Capstone

## Project overview
This project packages the trained champion model from the supervised classification task as a REST API using FastAPI. The API accepts borrower information as JSON and returns a loan-default prediction with class probabilities.

**Champion model:** Logistic Regression with the preprocessing pipeline saved in `outputs/champion_model.joblib`.

## Model and dataset context
- Dataset: `credit-default-risk.csv` from the previous modeling task (synthetic educational data)
- Target: `loan_default` (`0` = No Default, `1` = Default)
- Champion selection: highest mean cross-validation ROC-AUC
- Champion CV ROC-AUC: `0.666875`
- Best parameters: `classifier__C = 0.1`

The saved artifact includes numeric imputation/scaling, categorical imputation/one-hot encoding, and the fitted Logistic Regression classifier.

> Responsible-use note: This dataset is synthetic and the model has modest evaluation performance. This API is for learning and demonstration only. It is not validated for real credit decisions and must not be used to approve, reject, or price real loans.

## Repository structure
```text
rabtech-task06-ml-inference-api/
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
├── implementation_note.md
├── outputs/
│   └── champion_model.joblib
└── tests/
    └── test_api.py
```

## Run locally (macOS / VS Code)

1. Open this repository folder in VS Code.
2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```
3. Activate it:
   ```bash
   source .venv/bin/activate
   ```
4. Install pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the API:
   ```bash
   uvicorn app:app --reload
   ```
6. Open the interactive API documentation:
   - `http://127.0.0.1:8000/docs`
   - Health check: `http://127.0.0.1:8000/health`

## API endpoints

### `GET /`
Returns API information and documentation paths.

### `GET /health`
Returns service status and confirms that the model is loaded.

### `POST /predict`
Accepts one borrower record as JSON.

Example request:
```json
{
  "age": 35,
  "annual_income": 65000,
  "loan_amount": 18000,
  "credit_score": 690,
  "employment_years": 8,
  "debt_to_income_ratio": 0.32,
  "loan_term_months": 36,
  "education": "Graduate",
  "employment_type": "Salaried",
  "home_ownership": "Rent",
  "loan_purpose": "Vehicle"
}
```

Example response:
```json
{
  "prediction": 0,
  "prediction_label": "No Default",
  "probabilities": {
    "0": 0.72,
    "1": 0.28
  }
}
```
The probability values above illustrate the response format; actual values are calculated by the loaded model.

Input fields:
- Numeric: `age`, `annual_income`, `loan_amount`, `credit_score`, `employment_years`, `debt_to_income_ratio`, `loan_term_months`
- `education`: `High School`, `Graduate`, `Postgraduate`
- `employment_type`: `Contract`, `Salaried`, `Self-employed`
- `home_ownership`: `Mortgage`, `Own`, `Rent`
- `loan_purpose`: `Education`, `Home Improvement`, `Medical`, `Personal`, `Vehicle`

Invalid or missing fields return HTTP `422`.

## Run unit tests
```bash
pytest -v
```

Tests cover the root endpoint, health endpoint, successful prediction response schema and probabilities, invalid categories, and missing required fields.

## Docker

Build the image:
```bash
docker build -t rabtech-ml-api .
```

Run the container:
```bash
docker run --rm -p 8000:8000 rabtech-ml-api
```

Then visit `http://127.0.0.1:8000/docs`.

## System architecture
1. Client sends borrower features as a JSON request.
2. FastAPI validates the request using the Pydantic schema.
3. The request is converted into a pandas DataFrame with the model's expected feature names.
4. The serialized preprocessing and Logistic Regression pipeline transforms the features and predicts the class.
5. The API returns the predicted class, readable label, and probabilities for both classes.

## Reproducibility
The API loads the same fitted champion pipeline produced in the previous supervised-classification task. Dependencies are pinned in `requirements.txt`. The model artifact must remain at `outputs/champion_model.joblib`.
