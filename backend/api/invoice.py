from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from supabase import Client

from backend.core.exceptions import FileParsingError
from backend.db.supabase import get_db
from backend.db.pinecone import get_pinecone_index
from backend.schemas.invoice import InvoiceResponse, InvoiceUploadResponse
from backend.services.entities import get_or_create_entity
from backend.services.fingerprinting.sha_fingerprint import generate_sha256_fingerprint
from backend.services.ingestion.csv_parser import parse_csv
from backend.services.ingestion.json_parser import parse_json
from backend.services.ingestion.normaliser import normalise_invoice
from backend.services.ingestion.pdf_parser import parse_pdf
from backend.services.graph.builder import upsert_invoice_graph
from backend.services.langgraph.workflow import run_fraud_investigation
from backend.services.fingerprinting.embedding import upsert_invoice_embedding

router = APIRouter()


async def _detect_and_parse(file: UploadFile) -> list[dict[str, Any]]:
    content = await file.read()
    content_type = (file.content_type or "").lower()
    filename = file.filename or ""

    try:
        if content_type.endswith("csv") or filename.lower().endswith(".csv"):
            return await parse_csv(content)
        if content_type.endswith("json") or filename.lower().endswith(".json"):
            return await parse_json(content)
        if "pdf" in content_type or filename.lower().endswith(".pdf"):
            return [await parse_pdf(content)]
    except FileParsingError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise FileParsingError(f"Failed to parse uploaded file: {exc}") from exc

    raise FileParsingError("Unsupported file type. Expected CSV, JSON, or PDF.")


@router.post("/invoices/upload", response_model=InvoiceUploadResponse)
async def upload_invoice(
    file: UploadFile = File(...),
    lender_name: str = Form(default=""),
    db: Client = Depends(get_db),
) -> InvoiceUploadResponse:
    """
    Upload invoices as CSV/PDF/JSON, normalise, persist, and index them.
    """
    try:
        raw_records = await _detect_and_parse(file)
    except FileParsingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pinecone_index = get_pinecone_index()
    created_ids: list[str] = []

    for raw in raw_records:
        # Ensure lender name is present (optional in many upload payloads)
        raw["lender_name"] = raw.get("lender_name") or lender_name
        invoice_event = normalise_invoice(raw, source_format="upload")

        supplier_id = await get_or_create_entity(
            db,
            entity_type="supplier",
            name=invoice_event["supplier_name"],
        )
        buyer_id = await get_or_create_entity(
            db,
            entity_type="buyer",
            name=invoice_event["buyer_name"],
        )

        lender_id: str | None = None
        if invoice_event.get("lender_name"):
            lender_id = await get_or_create_entity(
                db,
                entity_type="lender",
                name=invoice_event["lender_name"],
            )

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
            "file_type": invoice_event.get("source_format") or "upload",
        }

        insert_result = db.table("invoices").insert(invoice_payload).execute()
        if not insert_result.data:
            continue

        invoice_row = insert_result.data[0]
        invoice_id = str(invoice_row["id"])
        created_ids.append(invoice_id)

        # Upsert graph
        await upsert_invoice_graph(invoice_event, invoice_id, supplier_id, buyer_id, lender_id)

        # Register fingerprint in consortium table
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

        # Upsert embedding in Pinecone
        upsert_invoice_embedding(invoice_id, invoice_event, pinecone_index)

    if not created_ids:
        raise HTTPException(status_code=500, detail="No invoices were created from the upload")

    return InvoiceUploadResponse(
        invoice_ids=created_ids,
        accepted_count=len(created_ids),
        rejected_count=0,
        errors=[],
    )


@router.post("/invoices/analyze/{invoice_id}")
async def analyze_invoice(invoice_id: str, db: Client = Depends(get_db)) -> dict[str, Any]:
    """
    Run full LangGraph fraud investigation on an existing invoice.
    """
    result = db.table("invoices").select("*").eq("id", invoice_id).limit(1).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Invoice not found")

    invoice_row = result.data[0]

    # Enrich invoice payload for downstream services (they expect names, not UUIDs)
    supplier_id = invoice_row.get("supplier_id")
    buyer_id = invoice_row.get("buyer_id")
    lender_id = invoice_row.get("lender_id")

    supplier = (
        db.table("entities").select("name").eq("id", supplier_id).limit(1).execute()
    )
    buyer = db.table("entities").select("name").eq("id", buyer_id).limit(1).execute()
    lender = (
        db.table("entities").select("name").eq("id", lender_id).limit(1).execute() if lender_id else None
    )

    supplier_name = supplier.data[0]["name"] if getattr(supplier, "data", None) else ""
    buyer_name = buyer.data[0]["name"] if getattr(buyer, "data", None) else ""
    lender_name = lender.data[0]["name"] if lender is not None and getattr(lender, "data", None) else None

    enriched_invoice = {
        "invoice_number": invoice_row.get("invoice_number"),
        "supplier_name": supplier_name,
        "buyer_name": buyer_name,
        "lender_name": lender_name,
        "po_number": invoice_row.get("po_number"),
        "grn_number": invoice_row.get("grn_number"),
        "invoice_date": str(invoice_row.get("invoice_date") or ""),
        "due_date": str(invoice_row.get("due_date") or "") if invoice_row.get("due_date") else None,
        "amount": float(invoice_row.get("amount") or 0),
        "currency": invoice_row.get("currency") or "INR",
        "line_items": invoice_row.get("line_items") or [],
        "source_format": "stored",
        "raw": invoice_row.get("metadata") or {},
        "_supplier_id": str(supplier_id or ""),
    }

    fraud_state = await run_fraud_investigation(enriched_invoice, invoice_id)

    # Persist summary fields back to invoice
    db.table("invoices").update(
        {
            "fraud_score": fraud_state.get("ensemble_score", 0.0),
            "fraud_decision": fraud_state.get("fraud_decision"),
            "fraud_patterns": fraud_state.get("fraud_patterns", []),
        }
    ).eq("id", invoice_id).execute()

    return fraud_state


def _invoice_row_to_response(row: dict[str, Any], supplier_name: str, buyer_name: str) -> InvoiceResponse:
    return InvoiceResponse(
        id=str(row.get("id")),
        invoice_number=row.get("invoice_number") or "",
        supplier_name=supplier_name,
        buyer_name=buyer_name,
        amount=float(row.get("amount") or 0),
        invoice_date=str(row.get("invoice_date") or ""),
        status=row.get("status") or "pending",
        fraud_score=float(row.get("fraud_score") or 0),
        fraud_decision=row.get("fraud_decision"),
        fraud_patterns=row.get("fraud_patterns") or [],
        created_at=str(row.get("created_at") or ""),
    )


@router.get("/invoices/", response_model=list[InvoiceResponse])
async def list_invoices(
    db: Client = Depends(get_db),
) -> list[InvoiceResponse]:
    result = db.table("invoices").select("*").limit(100).execute()
    rows = result.data or []
    responses: list[InvoiceResponse] = []
    for row in rows:
        supplier_id = row.get("supplier_id")
        buyer_id = row.get("buyer_id")
        s_res = db.table("entities").select("name").eq("id", supplier_id).limit(1).execute() if supplier_id else None
        b_res = db.table("entities").select("name").eq("id", buyer_id).limit(1).execute() if buyer_id else None
        supplier_name = s_res.data[0]["name"] if s_res and s_res.data else ""
        buyer_name = b_res.data[0]["name"] if b_res and b_res.data else ""
        responses.append(_invoice_row_to_response(row, supplier_name, buyer_name))
    return responses


@router.get("/invoices/{invoice_id}", response_model=InvoiceResponse)
async def get_invoice(invoice_id: str, db: Client = Depends(get_db)) -> InvoiceResponse:
    result = db.table("invoices").select("*").eq("id", invoice_id).limit(1).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Invoice not found")
    row = result.data[0]
    supplier_id = row.get("supplier_id")
    buyer_id = row.get("buyer_id")
    s_res = db.table("entities").select("name").eq("id", supplier_id).limit(1).execute() if supplier_id else None
    b_res = db.table("entities").select("name").eq("id", buyer_id).limit(1).execute() if buyer_id else None
    supplier_name = s_res.data[0]["name"] if s_res and s_res.data else ""
    buyer_name = b_res.data[0]["name"] if b_res and b_res.data else ""
    return _invoice_row_to_response(row, supplier_name, buyer_name)


@router.delete("/invoices/{invoice_id}")
async def delete_invoice(invoice_id: str, db: Client = Depends(get_db)) -> dict[str, Any]:
    """
    Soft-delete invoice by marking status as 'rejected'.
    """
    result = db.table("invoices").update({"status": "rejected"}).eq("id", invoice_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return {"deleted": True, "id": invoice_id}

