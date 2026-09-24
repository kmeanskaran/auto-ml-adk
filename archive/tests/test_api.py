"""The HTTP service exposes the same runtime the agents run."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import create_app


def test_http_trains_and_serves(tmp_path) -> None:
  client = TestClient(create_app(root=tmp_path, model="scripted"))
  created = client.post(
      "/experiments",
      json={
          "objective": "Predict customer churn. Optimize for F1.",
          "target_column": "churn",
          "target_value": 0.75,
          "approve_training": False,
      },
  )
  assert created.status_code == 200, created.text
  body = created.json()
  assert body["status"] == "completed"
  experiment_id = body["id"]

  design = client.get(f"/experiments/{experiment_id}/design")
  assert design.status_code == 200
  assert "CandidateTrainer" in design.text

  scored = client.post(
      f"/experiments/{experiment_id}/predict",
      json={
          "rows": [
              {
                  "tenure_months": 2,
                  "monthly_charges": 110,
                  "total_charges": 220,
                  "contract": "month-to-month",
                  "internet": "fiber",
                  "support_calls": 5,
                  "senior": 0,
                  "paperless": 1,
              }
          ]
      },
  )
  assert scored.status_code == 200, scored.text
  assert scored.json()["predictions"][0]["label"] in (0, 1)

  listed = client.get("/experiments")
  assert listed.json()[0]["id"] == experiment_id

  missing = client.post("/experiments/does-not-exist/predict", json={"rows": [{"a": 1}]})
  assert missing.status_code == 404
  assert missing.headers["x-request-id"]


def test_health_ready_and_request_id(tmp_path) -> None:
  client = TestClient(create_app(root=tmp_path, model="scripted"))
  health = client.get("/health", headers={"X-Request-ID": "req-health"})
  assert health.status_code == 200
  assert health.json()["status"] == "ok"
  assert health.headers["x-request-id"] == "req-health"

  ready = client.get("/ready")
  assert ready.status_code == 200
  assert ready.json()["status"] == "ready"
  assert ready.headers["x-request-id"]

  missing = client.get("/not-a-route")
  assert missing.status_code == 404


def test_upload_over_the_size_limit_is_rejected(tmp_path) -> None:
  from dataclasses import replace

  from api.settings import load_settings

  settings = replace(load_settings(), max_upload_bytes=8, root=tmp_path, model="scripted")
  client = TestClient(create_app(settings=settings))
  response = client.post(
      "/experiments/upload",
      files={"file": ("customers.csv", b"customer_id,churn\n" + b"x" * 40)},
      data={"target_column": "churn", "target_value": "0.0", "approve_training": "false"},
  )
  assert response.status_code == 413
  assert "size" in response.json()["error"].lower()
