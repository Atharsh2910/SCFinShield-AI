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
        return {"dedup_result": result}
    except Exception as exc:
        return {"dedup_result": {}, "processing_errors": [f"dedup_agent: {exc}"]}


async def match_agent(state: FraudState) -> dict[str, Any]:
    """Agent 2: Run 3-way PO/GRN/Invoice match."""
    try:
        invoice = state["invoice"]
        po_record = invoice.get("_po_record")
        grn_record = invoice.get("_grn_record")
        result = run_three_way_match(invoice, po_record, grn_record)
        return {"match_result": result}
    except Exception as exc:
        return {
            "match_result": {"overall_match_score": 0.5},
            "processing_errors": [f"match_agent: {exc}"],
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
            }
        }
    except Exception as exc:
        return {"graph_findings": {}, "processing_errors": [f"graph_agent: {exc}"]}


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

        patterns: list[str] = list(state.get("graph_findings", {}).get("detected_patterns", []))
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
        }
    except Exception as exc:
        return {
            "ml_result": {},
            "ensemble_score": 0.0,
            "fraud_decision": "REVIEW",
            "processing_errors": [f"ml_agent: {exc}"],
        }


async def rag_agent(state: FraudState) -> dict[str, Any]:
    """Agent 5: Generate RAG-grounded fraud narrative (only for REVIEW/HOLD decisions)."""
    if state.get("fraud_decision") == "PASS":
        return {"rag_result": {}, "alert_narrative": "", "regulation_citations": []}
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
        }
    except Exception as exc:
        return {
            "rag_result": {},
            "alert_narrative": "Narrative generation failed.",
            "regulation_citations": [],
            "processing_errors": [f"rag_agent: {exc}"],
        }

