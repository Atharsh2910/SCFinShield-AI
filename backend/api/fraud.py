from __future__ import annotations

import time
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from supabase import Client

from backend.db.supabase import get_db
from backend.schemas.fraud import AnalystDecisionUpdate, FraudAnalysisResponse, FraudCaseResponse
from backend.services.langgraph.workflow import run_fraud_investigation
from backend.services.reporting.case_manager import create_fraud_case, generate_sar_draft, update_analyst_decision
from backend.services.rag.knowledge_base import upsert_fraud_cases_to_pinecone

router = APIRouter()

def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


async def _fetch_enriched_invoice(invoice_id: str, db: Client) -> dict[str, Any]:
    invoice_result = db.table("invoices").select("*").eq("id", invoice_id).limit(1).execute()
    if not invoice_result.data:
        raise HTTPException(status_code=404, detail="Invoice not found")

    invoice_row = invoice_result.data[0]
    supplier_id = invoice_row.get("supplier_id")
    buyer_id = invoice_row.get("buyer_id")
    lender_id = invoice_row.get("lender_id")

    supplier = db.table("entities").select("name").eq("id", supplier_id).limit(1).execute() if supplier_id else None
    buyer = db.table("entities").select("name").eq("id", buyer_id).limit(1).execute() if buyer_id else None
    lender = db.table("entities").select("name").eq("id", lender_id).limit(1).execute() if lender_id else None

    supplier_name = supplier.data[0]["name"] if supplier and getattr(supplier, "data", None) else ""
    buyer_name = buyer.data[0]["name"] if buyer and getattr(buyer, "data", None) else ""
    lender_name = lender.data[0]["name"] if lender and getattr(lender, "data", None) else None

    return {
        "invoice_number": invoice_row.get("invoice_number"),
        "supplier_name": supplier_name,
        "buyer_name": buyer_name,
        "lender_name": lender_name,
        "po_number": invoice_row.get("po_number"),
        "grn_number": invoice_row.get("grn_number"),
        "invoice_date": str(invoice_row.get("invoice_date") or ""),
        "due_date": str(invoice_row.get("due_date") or "") if invoice_row.get("due_date") else None,
        "amount": _to_float(invoice_row.get("amount")),
        "currency": invoice_row.get("currency") or "INR",
        "line_items": invoice_row.get("line_items") or [],
        "source_format": "stored",
        "raw": invoice_row.get("metadata") or {},
        "_supplier_id": str(supplier_id or ""),
    }


@router.post("/fraud/analyze/{invoice_id}")
async def analyze_fraud(invoice_id: str, db: Client = Depends(get_db)) -> FraudAnalysisResponse:
    """
    Run LangGraph investigation and create/update fraud_case record.
    """
    start = time.time()
    enriched_invoice = await _fetch_enriched_invoice(invoice_id, db)
    fraud_state = await run_fraud_investigation(enriched_invoice, invoice_id)
    case = await create_fraud_case(invoice_id, fraud_state, db)

    cascade_depth = len((fraud_state.get("graph_findings") or {}).get("downstream_invoices", []) or [])
    cascade_exposure = _to_float((fraud_state.get("graph_findings") or {}).get("total_cascade_exposure", 0.0))

    ensemble_score = _to_float(fraud_state.get("ensemble_score"))
    individual_scores = (fraud_state.get("ml_result") or {}).get("individual_scores", {}) or {}

    processing_time_ms = int((time.time() - start) * 1000)

    return FraudAnalysisResponse(
        invoice_id=invoice_id,
        case_id=case.get("id") if case else None,
        ensemble_score=ensemble_score,
        fraud_decision=str(fraud_state.get("fraud_decision")),
        fraud_patterns=fraud_state.get("fraud_patterns", []) or [],
        individual_scores=individual_scores,
        top_shap_features=(fraud_state.get("ml_result") or {}).get("top_shap_features", []) or [],
        cascade_depth=cascade_depth,
        cascade_exposure=cascade_exposure,
        alert_narrative=fraud_state.get("alert_narrative", "") or "",
        regulation_citations=fraud_state.get("regulation_citations", []) or [],
        processing_time_ms=processing_time_ms,
        processing_errors=fraud_state.get("processing_errors", []) or [],
    )


@router.get("/fraud/cases")
async def list_cases(
    db: Client = Depends(get_db),
    decision: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    lender_id: str | None = Query(default=None),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
) -> list[FraudCaseResponse]:
    query = db.table("fraud_cases").select("*")
    if decision:
        query = query.eq("decision", decision)
    if severity:
        query = query.eq("severity", severity)
    if date_from:
        query = query.gte("created_at", date_from.isoformat())
    if date_to:
        query = query.lte("created_at", date_to.isoformat())

    result = query.limit(200).execute()
    rows = result.data or []

    # Optional lender filter via invoice join done in application layer.
    if lender_id:
        filtered_rows: list[dict[str, Any]] = []
        for row in rows:
            invoice_id = row.get("invoice_id")
            if not invoice_id:
                continue
            inv = db.table("invoices").select("lender_id").eq("id", invoice_id).limit(1).execute()
            inv_lender_id = inv.data[0].get("lender_id") if inv.data else None
            if inv_lender_id == lender_id:
                filtered_rows.append(row)
        rows = filtered_rows
    return [FraudCaseResponse(**row) for row in rows]


@router.get("/fraud/cases/{case_id}")
async def get_case(case_id: str, db: Client = Depends(get_db)) -> FraudCaseResponse:
    result = db.table("fraud_cases").select("*").eq("id", case_id).limit(1).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Fraud case not found")
    return FraudCaseResponse(**result.data[0])


@router.patch("/fraud/cases/{case_id}")
async def update_case(
    case_id: str,
    body: AnalystDecisionUpdate,
    db: Client = Depends(get_db),
) -> dict[str, Any]:
    updated = await update_analyst_decision(
        case_id=case_id,
        decision=body.analyst_decision,
        notes=body.analyst_notes,
        db_client=db,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Fraud case not found")
    return updated


@router.post("/fraud/cases/{case_id}/sar")
async def sar_draft(case_id: str, db: Client = Depends(get_db)) -> dict[str, Any]:
    sar = await generate_sar_draft(case_id=case_id, db_client=db)
    return {"case_id": case_id, "sar_draft": sar}


@router.post("/fraud/cases/{case_id}/reindex")
async def reindex_case(case_id: str, db: Client = Depends(get_db)) -> dict[str, Any]:
    """
    Reindex a fraud case into the Pinecone fraud_cases namespace.
    Useful after updating analyst notes or historical backfills.
    """
    result = db.table("fraud_cases").select("*").eq("id", case_id).limit(1).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Fraud case not found")

    case = result.data[0]
    document = {
        "id": str(case.get("id") or case_id),
        "title": f"Fraud Case {case.get('case_number', case_id)}",
        "source": f"fraud_case:{case_id}",
        "category": "fraud_case",
        "content": (
            f"Case Number: {case.get('case_number', '')}\n"
            f"Invoice ID: {case.get('invoice_id', '')}\n"
            f"Decision: {case.get('decision', '')}\n"
            f"Severity: {case.get('severity', '')}\n"
            f"Fraud Score: {case.get('fraud_score', 0)}\n"
            f"Fraud Patterns: {case.get('fraud_patterns', [])}\n"
            f"Alert Narrative: {case.get('alert_narrative', '')}\n"
            f"Analyst Notes: {case.get('analyst_notes', '')}\n"
            f"Regulation Citations: {case.get('regulation_citations', [])}\n"
        ),
    }
    vector_count = upsert_fraud_cases_to_pinecone([document])
    return {"case_id": case_id, "vector_count": vector_count}


@router.get("/fraud/dashboard/summary")
async def fraud_dashboard_summary(db: Client = Depends(get_db)) -> dict[str, Any]:
    """
    Backward-compatible dashboard summary under /fraud namespace.
    """
    invoices = db.table("invoices").select("id,amount,fraud_decision").limit(500).execute()
    rows = invoices.data or []
    total = len(rows)
    flagged = [r for r in rows if r.get("fraud_decision") in ("REVIEW", "HOLD")]
    flagged_count = len(flagged)
    return {
        "total_invoices": total,
        "flagged_count": flagged_count,
        "fraud_rate": (flagged_count / total) if total else 0.0,
        "total_exposure": sum(float(r.get("amount") or 0) for r in flagged),
    }

