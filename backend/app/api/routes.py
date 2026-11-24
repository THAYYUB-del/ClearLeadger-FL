"""FastAPI route definitions for ClearLedger-FL."""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.models.schemas import (
    FederatedStartRequest,
    HospitalId,
    InitializeExperimentRequest,
    LocalTrainRequest,
    PredictExplainRequest,
    SecuritySimulateRequest,
    TamperTestRequest,
)
from app.services.coordinator import FederatedCoordinator

router = APIRouter(prefix="/api")
coordinator = FederatedCoordinator()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ClearLedger-FL Python FastAPI Backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/dashboard")
def dashboard():
    return coordinator.get_dashboard_summary()


@router.get("/hospitals")
def hospitals():
    return {"hospitals": coordinator.get_hospitals()}


@router.get("/hospitals/{hospital_id}")
def hospital_detail(hospital_id: HospitalId):
    return coordinator.get_hospital_detail(hospital_id)


@router.post("/experiment/initialize")
def initialize_experiment(req: InitializeExperimentRequest):
    return coordinator.initialize_experiment(
        distribution=req.distribution,
        seed=req.seed,
        local_epochs=req.local_epochs,
        learning_rate=req.learning_rate,
    )


@router.post("/hospitals/{hospital_id}/train")
def train_hospital(hospital_id: HospitalId, req: LocalTrainRequest = LocalTrainRequest()):
    try:
        return coordinator.train_single_hospital(hospital_id, req.local_epochs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/federated/start")
def start_federated(req: FederatedStartRequest):
    try:
        return coordinator.run_federated_training(req.rounds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/federated/status")
def federated_status():
    return coordinator.get_federated_status()


@router.get("/federated/rounds")
def federated_rounds():
    return {"rounds": coordinator.db.get_rounds()}


@router.post("/experiment/reset")
def reset_experiment():
    return coordinator.reset_experiment()


@router.get("/security/events")
def security_events():
    events = coordinator.db.get_security_events(200)
    accepted = sum(1 for e in events if e["decision"] == "ACCEPTED")
    rejected_auth = sum(1 for e in events if e["layer_failed"] == "AUTHENTICATION")
    rejected_val = sum(1 for e in events if e["layer_failed"] == "MALICIOUS_VALIDATION")
    return {
        "summary": {
            "total_events": len(events),
            "accepted": accepted,
            "rejected_total": rejected_auth + rejected_val,
            "rejected_authentication": rejected_auth,
            "rejected_malicious_validation": rejected_val,
        },
        "events": events,
    }


@router.post("/security/simulate")
def simulate_security(req: SecuritySimulateRequest):
    try:
        return coordinator.simulate_security_experiment(req.scenario, req.hospital_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/ledger")
def get_ledger():
    return coordinator.get_ledger()


@router.post("/ledger/verify")
def verify_ledger():
    return coordinator.verify_ledger()


@router.post("/ledger/tamper-test")
def tamper_test(req: TamperTestRequest = TamperTestRequest()):
    try:
        return coordinator.run_tamper_test(req.target_record_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/explainability/global")
def explainability_global():
    return coordinator.get_global_explainability()


@router.post("/explainability/predict")
def explainability_predict(req: PredictExplainRequest):
    try:
        return coordinator.explain_prediction(req.sample_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/analytics")
def analytics():
    return coordinator.get_analytics()


@router.get("/analytics/export")
def analytics_export():
    try:
        csv_body = coordinator.export_analytics_csv()
        return PlainTextResponse(
            content=csv_body,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="clearledger-fl-analytics.csv"'
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
