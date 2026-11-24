"""Pydantic v2 request and response schemas for ClearLedger-FL API."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


HospitalId = Literal["hospital_a", "hospital_b", "hospital_c"]
DistributionMode = Literal["balanced", "non_iid"]
SecurityScenario = Literal[
    "valid_update",
    "invalid_hmac",
    "forged_sender",
    "exaggerated_update",
    "label_flipping",
]


class InitializeExperimentRequest(BaseModel):
    distribution: DistributionMode = "balanced"
    seed: int = Field(default=42, ge=1, le=999999)
    local_epochs: int = Field(default=12, ge=1, le=50)
    learning_rate: float = Field(default=0.12, gt=0.0, le=2.0)


class LocalTrainRequest(BaseModel):
    local_epochs: Optional[int] = Field(default=None, ge=1, le=50)


class FederatedStartRequest(BaseModel):
    rounds: int = Field(default=1, ge=1, le=20)


class SecuritySimulateRequest(BaseModel):
    scenario: SecurityScenario
    hospital_id: HospitalId = "hospital_b"


class TamperTestRequest(BaseModel):
    target_record_id: Optional[int] = Field(default=None, ge=1)


class PredictExplainRequest(BaseModel):
    sample_index: int = Field(default=0, ge=0)
