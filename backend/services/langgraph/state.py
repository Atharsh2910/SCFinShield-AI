from __future__ import annotations

from operator import add
from typing import Annotated, Any, Dict, List, TypedDict


class FraudState(TypedDict, total=False):
    # Input
    invoice: Dict[str, Any]
    invoice_id: str

    # Stage outputs
    dedup_result: Dict[str, Any]
    match_result: Dict[str, Any]
    graph_findings: Dict[str, Any]
    ml_result: Dict[str, Any]
    rag_result: Dict[str, Any]
    retrieved_documents: List[Dict[str, Any]]

    # Final outputs
    ensemble_score: float
    fraud_decision: str
    fraud_patterns: Annotated[List[str], add]
    alert_narrative: str
    regulation_citations: Annotated[List[Dict[str, Any]], add]
    shap_values: Dict[str, float]
    cascade_exposure: float
    processing_errors: Annotated[List[str], add]
    audit_trail: Annotated[List[str], add]

