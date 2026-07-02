from __future__ import annotations

from typing import Any, Dict, List, TypedDict


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

    # Final outputs
    ensemble_score: float
    fraud_decision: str
    fraud_patterns: List[str]
    alert_narrative: str
    regulation_citations: List[Dict[str, Any]]
    shap_values: Dict[str, float]
    cascade_exposure: float
    processing_errors: List[str]

