from __future__ import annotations

from datetime import datetime
from typing import Any

from backend.core.config import get_settings
from backend.core.constants import AlertSeverity, FraudDecision
from backend.services.rag.knowledge_base import upsert_fraud_cases_to_pinecone


def _build_case_summary_document(case_record: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case_record.get("id") or case_record.get("case_number") or _generate_case_number())
    narrative = str(case_record.get("alert_narrative") or "")
    patterns = case_record.get("fraud_patterns") or []
    analyst_notes = case_record.get("analyst_notes") or ""
    citations = case_record.get("regulation_citations") or []
    content = (
        f"Case Number: {case_record.get('case_number', '')}\n"
        f"Invoice ID: {case_record.get('invoice_id', '')}\n"
        f"Decision: {case_record.get('decision', '')}\n"
        f"Severity: {case_record.get('severity', '')}\n"
        f"Fraud Score: {case_record.get('fraud_score', 0)}\n"
        f"Primary Signal: {case_record.get('primary_signal', '')}\n"
        f"Fraud Patterns: {patterns}\n"
        f"Alert Narrative: {narrative}\n"
        f"Analyst Decision: {case_record.get('analyst_decision', '')}\n"
        f"Analyst Notes: {analyst_notes}\n"
        f"Regulation Citations: {citations}\n"
    )
    return {
        "id": case_id,
        "title": f"Fraud Case {case_record.get('case_number', case_id)}",
        "source": f"fraud_case:{case_record.get('id', case_id)}",
        "category": "fraud_case",
        "content": content,
    }


def _best_effort_sync_case_to_vector_db(case_record: dict[str, Any]) -> None:
    try:
        upsert_fraud_cases_to_pinecone([_build_case_summary_document(case_record)])
    except Exception:
        # Safe no-op when Pinecone is not configured yet.
        pass


def _best_effort_audit_log(db_client, event_type: str, payload: dict[str, Any]) -> None:
    try:
        db_client.table("audit_log").insert(
            {
                "event_type": event_type,
                "entity_type": "fraud_case",
                "entity_id": payload.get("id"),
                "invoice_id": payload.get("invoice_id"),
                "case_id": payload.get("id"),
                "actor": "system",
                "payload": payload,
            }
        ).execute()
    except Exception:
        pass


def _severity_from_score(score: float) -> str:
    if score >= 0.9:
        return AlertSeverity.CRITICAL
    if score >= 0.75:
        return AlertSeverity.HIGH
    if score >= 0.5:
        return AlertSeverity.MEDIUM
    return AlertSeverity.LOW


def _recommended_action(decision: str) -> str:
    return {
        FraudDecision.HOLD: "HOLD financing pending enhanced due diligence",
        FraudDecision.REVIEW: "REVIEW evidence and request supporting documents",
        FraudDecision.PASS: "PASS financing request without additional action",
    }.get(decision, "REVIEW evidence")


def _generate_case_number() -> str:
    year = datetime.utcnow().year
    # Best-effort case number uniqueness: use timestamp-derived suffix.
    # (In production we'd use DB sequence/lock.)
    suffix = int(datetime.utcnow().timestamp() * 1000) % 1_000_000
    return f"SCF-{year}-{suffix:06d}"


async def create_fraud_case(
    invoice_id: str,
    fraud_state: dict[str, Any],
    db_client,
) -> dict[str, Any]:
    """
    Create or update a `fraud_cases` record from the completed LangGraph results.
    """
    score = float(fraud_state.get("ensemble_score", 0.0) or 0.0)
    decision = str(fraud_state.get("fraud_decision", FraudDecision.REVIEW) or FraudDecision.REVIEW)
    severity = _severity_from_score(score)

    patterns = fraud_state.get("fraud_patterns", []) or []
    primary_signal = patterns[0] if patterns else "UNKNOWN"

    rag_citations = fraud_state.get("regulation_citations", None)
    if rag_citations is None:
        rag_citations = fraud_state.get("rag_result", {}).get("regulation_citations", []) or []

    payload = {
        "invoice_id": invoice_id,
        "case_number": fraud_state.get("case_number") or _generate_case_number(),
        "fraud_patterns": patterns,
        "fraud_score": score,
        "decision": decision,
        "severity": severity,
        "primary_signal": primary_signal,
        "ensemble_scores": fraud_state.get("ml_result", {}).get("individual_scores", {}),
        "shap_values": fraud_state.get("shap_values", {}) or {},
        "cascade_path": fraud_state.get("graph_findings", {}).get("downstream_invoices", []) or [],
        "rag_context": fraud_state.get("rag_result", {}).get("narrative") or fraud_state.get("alert_narrative", ""),
        "alert_narrative": fraud_state.get("alert_narrative") or fraud_state.get("rag_result", {}).get("narrative", ""),
        "regulation_citations": rag_citations,
        "analyst_decision": None,
        "analyst_notes": None,
    }

    existing = db_client.table("fraud_cases").select("*").eq("invoice_id", invoice_id).limit(1).execute()
    if existing.data:
        updated = (
            db_client.table("fraud_cases")
            .update(payload)
            .eq("id", existing.data[0]["id"])
            .execute()
        )
        case_record = updated.data[0]
        _best_effort_audit_log(db_client, "fraud_case_updated", case_record)
        _best_effort_sync_case_to_vector_db(case_record)
        return case_record

    inserted = db_client.table("fraud_cases").insert(payload).execute()
    case_record = inserted.data[0]
    _best_effort_audit_log(db_client, "fraud_case_created", case_record)
    _best_effort_sync_case_to_vector_db(case_record)
    return case_record


async def generate_sar_draft(case_id: str, db_client) -> str:
    """
    Generate a SAR draft for the case. If Anthropic is not configured, returns a templated fallback.
    """
    settings = get_settings()

    case_res = db_client.table("fraud_cases").select("*").eq("id", case_id).limit(1).execute()
    if not case_res.data:
        return "SAR draft not generated: case not found."

    case = case_res.data[0]
    invoice_id = case.get("invoice_id")
    decision = case.get("decision") or FraudDecision.REVIEW
    severity = case.get("severity") or _severity_from_score(float(case.get("fraud_score", 0.0) or 0.0))
    patterns = case.get("fraud_patterns") or []
    narrative = case.get("alert_narrative") or ""

    fallback = (
        f"Suspicious Activity Report (draft)\n"
        f"- Case: {case.get('case_number')}\n"
        f"- Invoice: {invoice_id}\n"
        f"- Decision: {decision}\n"
        f"- Severity: {severity}\n"
        f"- Detected patterns: {patterns}\n\n"
        f"Summary narrative:\n{narrative}\n\n"
        f"Recommended action: {_recommended_action(decision)}\n"
    )

    if not settings.anthropic_api_key:
        return fallback

    # Local import to avoid hard dependency at import-time
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model=settings.claude_model,
        anthropic_api_key=settings.anthropic_api_key,
        max_tokens=900,
    )

    prompt = (
        "You are an expert compliance officer writing a SAR draft for supply chain finance fraud detection.\n"
        "Use the case evidence and narrative below. Provide a structured SAR draft with:\n"
        "1) Parties involved\n"
        "2) Summary of suspicious activity\n"
        "3) Fraud patterns\n"
        "4) Regulatory references (if provided)\n"
        "5) Actions taken / recommended next steps\n\n"
        f"Case evidence:\n"
        f"- Case Number: {case.get('case_number')}\n"
        f"- Invoice ID: {invoice_id}\n"
        f"- Fraud Score: {case.get('fraud_score')}\n"
        f"- Decision: {decision}\n"
        f"- Severity: {severity}\n"
        f"- Fraud Patterns: {patterns}\n"
        f"- Alert Narrative: {narrative}\n"
        f"- Regulation Citations: {case.get('regulation_citations')}\n\n"
        "Write the SAR draft now."
    )

    try:
        resp = await llm.ainvoke(prompt)
        sar = getattr(resp, "content", "") or ""
        if not sar.strip():
            sar = fallback
    except Exception:
        sar = fallback

    # Persist SAR draft
    try:
        db_client.table("fraud_cases").update({"sar_draft": sar}).eq("id", case_id).execute()
    except Exception:
        pass

    return sar


async def update_analyst_decision(
    case_id: str,
    decision: str,
    notes: str | None,
    db_client,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"analyst_decision": decision}
    if notes is not None:
        payload["analyst_notes"] = notes

    updated = db_client.table("fraud_cases").update(payload).eq("id", case_id).execute()
    if not updated.data:
        return {}
    _best_effort_sync_case_to_vector_db(updated.data[0])

    # Best-effort audit log
    try:
        db_client.table("audit_log").insert(
            {
                "event_type": "analyst_decision_updated",
                "case_id": case_id,
                "actor": "analyst",
                "payload": {"decision": decision},
            }
        ).execute()
    except Exception:
        pass

    return updated.data[0]

