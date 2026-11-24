from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_PAYLOAD = {
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
    "loan_purpose": "Vehicle",
}

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}

def test_predict_endpoint_returns_probabilities():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["prediction_label"] in ["Default", "No Default"]
    assert set(body["probabilities"].keys()) == {"0", "1"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6

def test_predict_rejects_invalid_category():
    payload = VALID_PAYLOAD.copy()
    payload["education"] = "Unknown"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_rejects_missing_field():
    payload = VALID_PAYLOAD.copy()
    payload.pop("age")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
