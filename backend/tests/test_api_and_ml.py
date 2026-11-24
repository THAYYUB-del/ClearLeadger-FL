"""Full FastAPI + scikit-learn + FedAvg + SHAP + CSV pytest suite."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_full_federated_security_ledger_shap_and_csv_workflow():
    # Reset first
    assert client.post("/api/experiment/reset").status_code == 200

    # Initialize
    init_resp = client.post(
        "/api/experiment/initialize",
        json={"distribution": "balanced", "seed": 42, "local_epochs": 10, "learning_rate": 0.12},
    )
    assert init_resp.status_code == 200
    init_data = init_resp.json()
    assert init_data["initialized"] is True
    assert init_data["total_train_samples"] + init_data["total_test_samples"] == 569

    # Train one hospital locally
    local_resp = client.post("/api/hospitals/hospital_a/train", json={"local_epochs": 8})
    assert local_resp.status_code == 200
    assert local_resp.json()["hospital"]["local_metrics"]["accuracy"] > 0.7

    # Run 2 federated rounds
    fed_resp = client.post("/api/federated/start", json={"rounds": 2})
    assert fed_resp.status_code == 200
    assert fed_resp.json()["current_round"] == 2
    assert fed_resp.json()["global_metrics"]["accuracy"] > 0.85

    # Invalid HMAC rejected
    sec_resp = client.post(
        "/api/security/simulate",
        json={"scenario": "invalid_hmac", "hospital_id": "hospital_b"},
    )
    assert sec_resp.status_code == 200
    assert sec_resp.json()["event"]["decision"] == "REJECTED"
    assert sec_resp.json()["event"]["hmac_valid"] is False

    # Exaggerated update passes HMAC but fails malicious validation
    exag_resp = client.post(
        "/api/security/simulate",
        json={"scenario": "exaggerated_update", "hospital_id": "hospital_b"},
    )
    assert exag_resp.status_code == 200
    assert exag_resp.json()["event"]["hmac_valid"] is True
    assert exag_resp.json()["event"]["decision"] == "REJECTED"

    # Ledger verification and controlled tampering test
    ver_resp = client.post("/api/ledger/verify")
    assert ver_resp.status_code == 200
    assert ver_resp.json()["valid"] is True

    tamper_resp = client.post("/api/ledger/tamper-test", json={})
    assert tamper_resp.status_code == 200
    assert tamper_resp.json()["verification_after_tamper"]["valid"] is False
    assert tamper_resp.json()["live_ledger_valid"] is True

    # Global & individual SHAP
    shap_global = client.get("/api/explainability/global")
    assert shap_global.status_code == 200
    assert shap_global.json()["available"] is True
    assert len(shap_global.json()["features"]) == 30

    shap_pred = client.post("/api/explainability/predict", json={"sample_index": 0})
    assert shap_pred.status_code == 200
    pred_data = shap_pred.json()
    assert abs(pred_data["shap_sum_check"] - pred_data["predicted_logit"]) < 1e-2

    # CSV export
    csv_resp = client.get("/api/analytics/export")
    assert csv_resp.status_code == 200
    assert "Federated Global Model (FedAvg)" in csv_resp.text
