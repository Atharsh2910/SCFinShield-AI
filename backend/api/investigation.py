from __future__ import annotations

import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from backend.db.supabase import get_db
from backend.schemas.investigation import (
    InvestigationAnswer,
    InvestigationQuestion,
    InvestigationStartResponse,
)
from backend.services.rag.analyst_qa import ask_analyst_question, delete_session, get_or_create_session, get_session_messages

router = APIRouter()

_session_to_case_id: dict[str, str] = {}


async def _fetch_case_and_invoice_state(case_id: str, db: Client) -> Dict[str, Any]:
    case_res = db.table("fraud_cases").select("*").eq("id", case_id).limit(1).execute()
    if not case_res.data:
        raise HTTPException(status_code=404, detail="Fraud case not found")

    case = case_res.data[0]
    invoice_id = case.get("invoice_id")

    inv_res = (
        db.table("invoices").select("*").eq("id", invoice_id).limit(1).execute() if invoice_id else None
    )
    invoice = inv_res.data[0] if inv_res is not None and inv_res.data else {}

    # Enrich invoice with names from entities for better prompt context.
    supplier_name = ""
    buyer_name = ""
    lender_name = None

    supplier_id = invoice.get("supplier_id")
    buyer_id = invoice.get("buyer_id")
    lender_id = invoice.get("lender_id")

    if supplier_id:
        s_res = db.table("entities").select("name").eq("id", supplier_id).limit(1).execute()
        supplier_name = s_res.data[0]["name"] if s_res.data else ""
    if buyer_id:
        b_res = db.table("entities").select("name").eq("id", buyer_id).limit(1).execute()
        buyer_name = b_res.data[0]["name"] if b_res.data else ""
    if lender_id:
        l_res = db.table("entities").select("name").eq("id", lender_id).limit(1).execute()
        lender_name = l_res.data[0]["name"] if l_res.data else None

    return {
        "invoice": {
            "invoice_number": invoice.get("invoice_number") or "",
            "supplier_name": supplier_name,
            "buyer_name": buyer_name,
            "lender_name": lender_name,
            "amount": float(invoice.get("amount") or 0),
            "currency": invoice.get("currency") or "INR",
            "invoice_date": str(invoice.get("invoice_date") or ""),
        },
        "ensemble_score": float(case.get("fraud_score") or 0),
        "fraud_decision": case.get("decision") or "REVIEW",
        "cascade_depth": len(case.get("cascade_path") or []),
        "cascade_exposure": 0.0,
        "narrative": case.get("alert_narrative") or "",
        "case": case,
    }


@router.post("/investigation/start/{case_id}", response_model=InvestigationStartResponse)
async def start_investigation(case_id: str, db: Client = Depends(get_db)) -> InvestigationStartResponse:
    # Validate case existence and create a new session id.
    _ = await _fetch_case_and_invoice_state(case_id, db)
    session_id = str(uuid.uuid4())
    _session_to_case_id[session_id] = case_id

    # Create session memory in rag module.
    get_or_create_session(session_id)

    return InvestigationStartResponse(session_id=session_id, case_id=case_id, messages=[])


@router.post("/investigation/{session_id}/ask", response_model=InvestigationAnswer)
async def ask(
    session_id: str,
    body: InvestigationQuestion,
    db: Client = Depends(get_db),
) -> InvestigationAnswer:
    if session_id not in _session_to_case_id:
        raise HTTPException(status_code=404, detail="Investigation session not found")

    case_id = _session_to_case_id[session_id]
    fraud_state = await _fetch_case_and_invoice_state(case_id, db)

    result = await ask_analyst_question(
        investigation_id=session_id,
        question=body.question,
        fraud_state=fraud_state,
    )

    return InvestigationAnswer(
        session_id=session_id,
        answer=result.get("answer") or "",
        citations=[],
    )


@router.get("/investigation/{session_id}/history")
async def history(session_id: str) -> dict[str, Any]:
    messages = get_session_messages(session_id)
    return {"session_id": session_id, "messages": messages}


@router.delete("/investigation/{session_id}")
async def end_session(session_id: str) -> dict[str, Any]:
    ok = delete_session(session_id)
    _session_to_case_id.pop(session_id, None)
    if not ok:
        raise HTTPException(status_code=404, detail="Investigation session not found")
    return {"deleted": True, "session_id": session_id}

