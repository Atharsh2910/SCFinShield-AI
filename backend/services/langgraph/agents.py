from __future__ import annotations

from typing import Any, Dict

from backend.core.constants import FraudPattern
from backend.db.supabase import get_supabase_client
from backend.services.fingerprinting.dedup_service import run_dedup_pipeline
from backend.services.graph.analytics import detect_carousel_trades, trace_cascade
from backend.services.langgraph.state import FraudState
from backend.services.ml.inference import run_ensemble_inference
from backend.services.rag.retrieval_chain import generate_fraud_narrative
from backend.services.verification.three_way_match import run_three_way_match


async def dedup_agent(state: FraudState) -> dict[str, Any]:
    """Agent 1: Run fingerprinting and duplicate detection."""
    try:
        db = get_supabase_client()
        result = await run_dedup_pipeline(state["invoice"], state["invoice_id"], db)
        patterns: list[str] = []
        if result.get("is_exact_duplicate") or float(result.get("duplicate_risk_score", 0) or 0) >= 0.85:
            patterns.append(FraudPattern.DUPLICATE_FINANCING)
        return {
            "dedup_result": result,
            "fraud_patterns": patterns,
            "audit_trail": ["dedup_agent completed"],
        }
    except Exception as exc:
        return {
            "dedup_result": {},
            "processing_errors": [f"dedup_agent: {exc}"],
            "audit_trail": ["dedup_agent failed"],
        }


async def match_agent(state: FraudState) -> dict[str, Any]:
    """Agent 2: Run 3-way PO/GRN/Invoice match."""
    try:
        invoice = state["invoice"]
        po_record = invoice.get("_po_record")
        grn_record = invoice.get("_grn_record")
        result = run_three_way_match(invoice, po_record, grn_record)
        patterns: list[str] = []
        if not result.get("pass", False) and not invoice.get("grn_number"):
            patterns.append(FraudPattern.PHANTOM_INVOICE)
        return {
            "match_result": result,
            "fraud_patterns": patterns,
            "audit_trail": ["match_agent completed"],
        }
    except Exception as exc:
        return {
            "match_result": {"overall_match_score": 0.5},
            "processing_errors": [f"match_agent: {exc}"],
            "audit_trail": ["match_agent failed"],
        }


async def graph_agent(state: FraudState) -> dict[str, Any]:
    """Agent 3: Run graph analytics — carousel detection + cascade tracing."""
    try:
        invoice = state["invoice"]
        supplier_id = invoice.get("_supplier_id", state["invoice_id"])

        carousel = await detect_carousel_trades(supplier_id)
        cascade = await trace_cascade(state["invoice_id"])

        patterns: list[str] = []
        if carousel.get("has_carousel"):
            patterns.append(FraudPattern.CAROUSEL_TRADE)
        if cascade.get("has_cascade"):
            patterns.append(FraudPattern.CASCADE_AMPLIFICATION)

        return {
            "graph_findings": {
                **carousel,
                **cascade,
                "detected_patterns": patterns,
            },
            "fraud_patterns": patterns,
            "cascade_exposure": float(cascade.get("total_cascade_exposure", 0.0) or 0.0),
            "audit_trail": ["graph_agent completed"],
        }
    except Exception as exc:
        return {
            "graph_findings": {},
            "processing_errors": [f"graph_agent: {exc}"],
            "audit_trail": ["graph_agent failed"],
        }


async def ml_agent(state: FraudState) -> dict[str, Any]:
    """Agent 4: Run ML ensemble inference."""
    try:
        entity_history = state.get("invoice", {}).get("_entity_history", {}) or {}
        result = await run_ensemble_inference(
            state["invoice"],
            state.get("dedup_result") or {},
            state.get("match_result") or {},
            state.get("graph_findings") or {},
            entity_history,
        )

        patterns: list[str] = []
        if result.get("individual_scores", {}).get("siamese", 0) > 0.85:
            patterns.append(FraudPattern.DUPLICATE_FINANCING)
        if result.get("individual_scores", {}).get("isolation_forest", 0) > 0.7:
            patterns.append(FraudPattern.VELOCITY_ANOMALY)

        return {
            "ml_result": result,
            "ensemble_score": float(result.get("ensemble_score", 0.0) or 0.0),
            "fraud_decision": str(result.get("fraud_decision", "PASS")),
            "fraud_patterns": patterns,
            "shap_values": result.get("shap_values", {}),
            "audit_trail": ["ml_agent completed"],
        }
    except Exception as exc:
        return {
            "ml_result": {},
            "ensemble_score": 0.0,
            "fraud_decision": "REVIEW",
            "processing_errors": [f"ml_agent: {exc}"],
            "audit_trail": ["ml_agent failed"],
        }


async def rag_agent(state: FraudState) -> dict[str, Any]:
    """Agent 5: Generate RAG-grounded fraud narrative (only for REVIEW/HOLD decisions)."""
    if state.get("fraud_decision") == "PASS":
        return {
            "rag_result": {},
            "alert_narrative": "",
            "regulation_citations": [],
            "audit_trail": ["rag_agent skipped for PASS"],
        }
    try:
        result = await generate_fraud_narrative(
            state["invoice"],
            {
                "ensemble_score": state.get("ensemble_score", 0),
                "fraud_decision": state.get("fraud_decision"),
                "individual_scores": state.get("ml_result", {}).get("individual_scores", {}),
                "top_shap_features": state.get("ml_result", {}).get("top_shap_features", []),
                "match_score": state.get("match_result", {}).get("overall_match_score", 0),
            },
            state.get("graph_findings") or {},
        )
        return {
            "rag_result": result,
            "alert_narrative": result.get("narrative", ""),
            "regulation_citations": result.get("regulation_citations", []),
            "audit_trail": ["rag_agent completed"],
        }
    except Exception as exc:
        return {
            "rag_result": {},
            "alert_narrative": "Narrative generation failed.",
            "regulation_citations": [],
            "processing_errors": [f"rag_agent: {exc}"],
            "audit_trail": ["rag_agent failed"],
        }


async def finalize_agent(state: FraudState) -> dict[str, Any]:
    """
    Finalize LangGraph outputs by deduplicating patterns/citations and ensuring
    state-level summary fields are populated consistently.
    """
    patterns = list(dict.fromkeys(state.get("fraud_patterns", []) or []))
    citations = state.get("regulation_citations", []) or []
    unique_citations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for citation in citations:
        key = (
            str(citation.get("title", "")),
            str(citation.get("source", "")),
            str(citation.get("category", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_citations.append(citation)

    graph_findings = state.get("graph_findings", {}) or {}
    return {
        "fraud_patterns": patterns,
        "regulation_citations": unique_citations,
        "cascade_exposure": float(graph_findings.get("total_cascade_exposure", state.get("cascade_exposure", 0.0)) or 0.0),
        "audit_trail": ["finalize_agent completed"],
    }

