from typing import Any

from pydantic import BaseModel, Field


class FraudAnalysisResponse(BaseModel):
    invoice_id: str
    case_id: str | None = None
    ensemble_score: float
    fraud_decision: str
    fraud_patterns: list[str]
    individual_scores: dict[str, float]
    top_shap_features: list[dict[str, Any]]
    cascade_depth: int
    cascade_exposure: float
    alert_narrative: str
    regulation_citations: list[dict[str, Any]]
    processing_time_ms: int
    processing_errors: list[str] = Field(default_factory=list)


class FraudCaseResponse(BaseModel):
    id: str
    case_number: str
    invoice_id: str
    fraud_score: float
    decision: str
    severity: str
    fraud_patterns: list[str]
    alert_narrative: str
    ensemble_scores: dict[str, float] = Field(default_factory=dict)
    shap_values: dict[str, float] = Field(default_factory=dict)
    cascade_path: list[dict[str, Any]] = Field(default_factory=list)
    regulation_citations: list[dict[str, Any]] = Field(default_factory=list)
    analyst_decision: str | None = None
    analyst_notes: str | None = None
    sar_draft: str | None = None
    created_at: str


class AnalystDecisionUpdate(BaseModel):
    analyst_decision: str
    analyst_notes: str | None = None
