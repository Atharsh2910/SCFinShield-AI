from __future__ import annotations

import asyncio
from random import uniform

from backend.db.pinecone import get_pinecone_index
from backend.db.supabase import get_supabase_client
from backend.services.entities import get_or_create_entity
from backend.services.fingerprinting.embedding import upsert_invoice_embedding
from backend.services.fingerprinting.sha_fingerprint import generate_sha256_fingerprint
from backend.services.graph.builder import upsert_invoice_graph


async def seed_demo_data() -> None:
    db = get_supabase_client()
    try:
        pinecone_index = get_pinecone_index()
    except Exception:
        pinecone_index = None

    suppliers = [
        await get_or_create_entity(db, entity_type="supplier", name=f"Supplier Tier {tier}", tier=tier, sector="Manufacturing")
        for tier in [1, 1, 2, 2, 3]
    ]
    buyers = [
        await get_or_create_entity(db, entity_type="buyer", name=f"Buyer {idx}", tier=1, sector="Retail")
        for idx in range(1, 4)
    ]
    lenders = [
        await get_or_create_entity(db, entity_type="lender", name=f"Lender {idx}")
        for idx in range(1, 3)
    ]

    invoice_ids: list[str] = []

    # 20 clean invoices
    for idx in range(20):
        supplier_id = suppliers[idx % len(suppliers)]
        buyer_id = buyers[idx % len(buyers)]
        lender_id = lenders[idx % len(lenders)]
        invoice_event = {
            "invoice_number": f"INV-CLEAN-{idx+1:04d}",
            "supplier_name": f"Supplier Tier {(idx % 3) + 1}",
            "buyer_name": f"Buyer {(idx % 3) + 1}",
            "lender_name": f"Lender {(idx % 2) + 1}",
            "po_number": f"PO-{idx+1:05d}",
            "grn_number": f"GRN-{idx+1:05d}",
            "invoice_date": "2026-07-01",
            "due_date": "2026-08-01",
            "amount": round(uniform(120000, 680000), 2),
            "currency": "INR",
            "line_items": [{"description": "Goods shipment"}],
            "source_format": "seed",
            "raw": {"is_fraud": False},
        }
        sha = generate_sha256_fingerprint(invoice_event)
        inserted = db.table("invoices").insert(
            {
                "invoice_number": invoice_event["invoice_number"],
                "supplier_id": supplier_id,
                "buyer_id": buyer_id,
                "lender_id": lender_id,
                "po_number": invoice_event["po_number"],
                "grn_number": invoice_event["grn_number"],
                "invoice_date": invoice_event["invoice_date"],
                "due_date": invoice_event["due_date"],
                "amount": invoice_event["amount"],
                "currency": "INR",
                "line_items": invoice_event["line_items"],
                "status": "pending",
                "sha256_fingerprint": sha,
                "fraud_score": 0.05,
                "fraud_decision": "PASS",
                "fraud_patterns": [],
                "metadata": invoice_event["raw"],
                "file_type": "seed",
            }
        ).execute()
        if not inserted.data:
            continue
        invoice_id = str(inserted.data[0]["id"])
        invoice_ids.append(invoice_id)
        await upsert_invoice_graph(invoice_event, invoice_id, supplier_id, buyer_id, lender_id)
        if pinecone_index is not None:
            try:
                upsert_invoice_embedding(invoice_id, invoice_event, pinecone_index)
            except Exception:
                pass

    # 5 fraud examples
    fraud_patterns = [
        "duplicate_financing",
        "phantom_invoice",
        "carousel_trade",
        "cascade_amplification",
        "velocity_anomaly",
    ]
    for idx, pattern in enumerate(fraud_patterns):
        supplier_id = suppliers[idx % len(suppliers)]
        buyer_id = buyers[idx % len(buyers)]
        lender_id = lenders[idx % len(lenders)]
        invoice_event = {
            "invoice_number": f"INV-FRAUD-{idx+1:04d}",
            "supplier_name": f"Supplier Tier {(idx % 3) + 1}",
            "buyer_name": f"Buyer {(idx % 3) + 1}",
            "lender_name": f"Lender {(idx % 2) + 1}",
            "po_number": f"PO-FRAUD-{idx+1:05d}",
            "grn_number": None if pattern == "phantom_invoice" else f"GRN-FRAUD-{idx+1:05d}",
            "invoice_date": "2026-07-01",
            "due_date": "2026-08-01",
            "amount": round(uniform(700000, 1500000), 2),
            "currency": "INR",
            "line_items": [{"description": f"Fraud scenario {pattern}"}],
            "source_format": "seed",
            "raw": {"is_fraud": True, "fraud_pattern": pattern},
        }
        sha = generate_sha256_fingerprint(invoice_event)
        inserted = db.table("invoices").insert(
            {
                "invoice_number": invoice_event["invoice_number"],
                "supplier_id": supplier_id,
                "buyer_id": buyer_id,
                "lender_id": lender_id,
                "po_number": invoice_event["po_number"],
                "grn_number": invoice_event["grn_number"],
                "invoice_date": invoice_event["invoice_date"],
                "due_date": invoice_event["due_date"],
                "amount": invoice_event["amount"],
                "currency": "INR",
                "line_items": invoice_event["line_items"],
                "status": "flagged",
                "sha256_fingerprint": sha,
                "fraud_score": 0.9,
                "fraud_decision": "HOLD",
                "fraud_patterns": [pattern],
                "metadata": invoice_event["raw"],
                "file_type": "seed",
            }
        ).execute()
        if not inserted.data:
            continue
        invoice_id = str(inserted.data[0]["id"])
        invoice_ids.append(invoice_id)
        await upsert_invoice_graph(invoice_event, invoice_id, supplier_id, buyer_id, lender_id)
        if pinecone_index is not None:
            try:
                upsert_invoice_embedding(invoice_id, invoice_event, pinecone_index)
            except Exception:
                pass

    print(
        f"Seed complete. Suppliers={len(suppliers)}, Buyers={len(buyers)}, "
        f"Lenders={len(lenders)}, Invoices={len(invoice_ids)}"
    )


if __name__ == "__main__":
    asyncio.run(seed_demo_data())
