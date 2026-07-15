from __future__ import annotations

from typing import Any

from backend.db.graph import upsert_edge, upsert_node


async def upsert_invoice_graph(
    invoice: dict[str, Any],
    invoice_id: str,
    supplier_id: str,
    buyer_id: str,
    lender_id: str | None = None,
) -> None:
    """
    Persist invoice graph relationships to the in-memory NetworkX graph
    and Supabase graph_nodes / graph_edges tables.
    """
    invoice_date = str(invoice.get("invoice_date") or "2024-01-01")
    amount = float(invoice.get("amount", 0) or 0)

    # Upsert entity nodes
    await upsert_node(
        supplier_id,
        node_type="supplier",
        name=invoice.get("supplier_name", ""),
    )
    await upsert_node(
        buyer_id,
        node_type="buyer",
        name=invoice.get("buyer_name", ""),
    )

    # Upsert invoice node
    await upsert_node(
        invoice_id,
        node_type="invoice",
        name=invoice.get("invoice_number", ""),
        props={
            "amount": amount,
            "invoice_date": invoice_date,
            "fraud_score": 0.0,
            "status": invoice.get("status", "pending"),
        },
    )

    # Supplier → Buyer supply relationship
    await upsert_edge(
        supplier_id,
        buyer_id,
        rel_type="SUPPLIES_TO",
        props={"since": invoice_date, "invoice_count": 1},
    )

    # Supplier → Invoice issued relationship
    await upsert_edge(
        supplier_id,
        invoice_id,
        rel_type="ISSUED",
        props={"date": invoice_date, "amount": amount},
    )

    # Invoice → Lender financing relationship
    if lender_id:
        await upsert_node(
            lender_id,
            node_type="lender",
            name=invoice.get("lender_name", ""),
        )
        await upsert_edge(
            invoice_id,
            lender_id,
            rel_type="FINANCED_BY",
            props={"amount": amount},
        )
