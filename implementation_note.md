# Implementation Note – Task 06

## Objective
Deploy the previously trained champion classification pipeline through a REST API using FastAPI, add unit tests, and prepare Docker-based deployment.

## Implementation
1. Reused the saved `outputs/champion_model.joblib` artifact from the supervised classification task.
2. Loaded the fitted scikit-learn pipeline once when the API application starts.
3. Created a `POST /predict` endpoint that accepts borrower details as a JSON payload.
4. Defined a Pydantic request schema to validate required numeric and categorical fields and reject unexpected fields.
5. Converted the validated payload into a pandas DataFrame using the feature names expected by the trained pipeline.
6. Returned the predicted class, readable prediction label, and probabilities for classes 0 and 1.
7. Added `GET /health` and root endpoints for service status and API discovery.
8. Added unit tests for successful predictions, response structure, health status, invalid categories, and missing fields.
9. Added a Dockerfile, `.dockerignore`, pinned dependencies, and run instructions in README.md.

## Model details
- Champion: Logistic Regression
- Selection metric: mean cross-validation ROC-AUC
- Best CV ROC-AUC: 0.666875
- Best parameter: `classifier__C = 0.1`

## Validation
Run `pytest -v` to execute the API tests. Run the service with `uvicorn app:app --reload` and use `/docs` to send a sample prediction request.

## Limitations and responsible use
The underlying dataset is synthetic and the model's measured performance is modest. This implementation demonstrates model serving and API engineering only. It is not a production-validated credit scoring service and must not be used for real lending decisions.
