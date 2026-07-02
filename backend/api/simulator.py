from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from supabase import Client

from backend.db.pinecone import get_pinecone_index
from backend.db.supabase import get_db
from backend.schemas.invoice import InvoiceUploadResponse
from backend.services.entities import get_or_create_entity
from backend.services.fingerprinting.embedding import upsert_invoice_embedding
from backend.services.fingerprinting.sha_fingerprint import generate_sha256_fingerprint
from backend.services.graph.builder import upsert_invoice_graph
from backend.services.simulator.synthetic_generator import SCENARIO_TEMPLATES, generate_synthetic_invoices

router = APIRouter()

ScenarioName = Literal["phantom_invoice", "duplicate_financing", "carousel_trade", "cascade_amplification"]


class GenerateRequest(BaseModel):
    n: int = Field(default=10, ge=1, le=500)
    scenario: ScenarioName = "phantom_invoice"
    persist: bool = False


@router.get("/simulator/scenarios")
async def list_scenarios() -> Dict[str, Any]:
    return {
        "scenarios": [
            {"name": name, "description": spec.get("description", "")}
            for name, spec in SCENARIO_TEMPLATES.items()
        ]
    }


async def _persist_invoice_event(db: Client, pinecone_index, invoice_event: Dict[str, Any]) -> str:
    supplier_id = await get_or_create_entity(db, entity_type="supplier", name=invoice_event["supplier_name"])
    buyer_id = await get_or_create_entity(db, entity_type="buyer", name=invoice_event["buyer_name"])

    lender_id: str | None = None
    if invoice_event.get("lender_name"):
        lender_id = await get_or_create_entity(db, entity_type="lender", name=invoice_event["lender_name"])

    sha256 = generate_sha256_fingerprint(invoice_event)

    invoice_payload = {
        "invoice_number": invoice_event["invoice_number"],
        "supplier_id": supplier_id,
        "buyer_id": buyer_id,
        "lender_id": lender_id,
        "po_number": invoice_event.get("po_number"),
        "grn_number": invoice_event.get("grn_number"),
        "invoice_date": invoice_event["invoice_date"],
        "due_date": invoice_event.get("due_date"),
        "amount": float(invoice_event.get("amount", 0) or 0),
        "currency": invoice_event.get("currency") or "INR",
        "line_items": invoice_event.get("line_items", []),
        "status": "pending",
        "sha256_fingerprint": sha256,
        "fraud_score": 0.0,
        "fraud_decision": "PASS",
        "fraud_patterns": [],
        "metadata": invoice_event.get("raw") or {},
        "file_type": invoice_event.get("source_format") or "simulator",
    }

    inserted = db.table("invoices").insert(invoice_payload).execute()
    if not inserted.data:
        raise RuntimeError("Failed to insert synthetic invoice")
    invoice_id = str(inserted.data[0]["id"])

    await upsert_invoice_graph(invoice_event, invoice_id, supplier_id, buyer_id, lender_id)

    db.table("fingerprint_registry").upsert(
        {
            "sha256_hash": sha256,
            "invoice_number": invoice_event.get("invoice_number"),
            "amount": float(invoice_event.get("amount", 0) or 0),
            "supplier_id": supplier_id,
            "buyer_id": buyer_id,
            "lender_id": lender_id,
            "invoice_date": invoice_event.get("invoice_date"),
            "status": "active",
        }
    ).execute()

    upsert_invoice_embedding(invoice_id, invoice_event, pinecone_index)
    return invoice_id


@router.post("/simulator/generate")
async def generate(
    body: GenerateRequest,
    db: Client = Depends(get_db),
) -> Dict[str, Any]:
    invoices = generate_synthetic_invoices(n=body.n, scenario=body.scenario)

    if not body.persist:
        return {"invoices": invoices, "scenario": body.scenario, "count": len(invoices)}

    pinecone_index = get_pinecone_index()
    invoice_ids: List[str] = []
    for ev in invoices:
        invoice_ids.append(await _persist_invoice_event(db, pinecone_index, ev))

    return InvoiceUploadResponse(
        invoice_ids=invoice_ids,
        accepted_count=len(invoice_ids),
        rejected_count=0,
        errors=[],
    )


@router.post("/simulator/scenario")
async def scenario(
    body: GenerateRequest,
    db: Client = Depends(get_db),
) -> Dict[str, Any]:
    # Alias of /generate but kept as a separate route for PRD compliance.
    return await generate(body=body, db=db)

