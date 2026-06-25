from typing import Any

from backend.db.neo4j import run_query


async def upsert_invoice_graph(
    invoice: dict[str, Any],
    invoice_id: str,
    supplier_id: str,
    buyer_id: str,
    lender_id: str | None = None,
) -> None:
    invoice_date = str(invoice.get("invoice_date") or "2024-01-01")
    amount = float(invoice.get("amount", 0) or 0)

    queries = [
        (
            """
            MERGE (s:Entity {id: $supplier_id})
            ON CREATE SET s.name = $supplier_name, s.type = 'supplier',
                          s.risk_score = 0.0, s.is_flagged = false, s.created_at = datetime()
            ON MATCH SET s.name = $supplier_name, s.updated_at = datetime()
            """,
            {"supplier_id": supplier_id, "supplier_name": invoice.get("supplier_name", "")},
        ),
        (
            """
            MERGE (b:Entity {id: $buyer_id})
            ON CREATE SET b.name = $buyer_name, b.type = 'buyer',
                          b.risk_score = 0.0, b.is_flagged = false, b.created_at = datetime()
            ON MATCH SET b.name = $buyer_name, b.updated_at = datetime()
            """,
            {"buyer_id": buyer_id, "buyer_name": invoice.get("buyer_name", "")},
        ),
        (
            """
            MATCH (s:Entity {id: $supplier_id})
            MATCH (b:Entity {id: $buyer_id})
            MERGE (s)-[r:SUPPLIES_TO]->(b)
            ON CREATE SET r.since = date($invoice_date), r.invoice_count = 1
            ON MATCH SET r.invoice_count = coalesce(r.invoice_count, 0) + 1,
                         r.last_invoice = date($invoice_date)
            """,
            {"supplier_id": supplier_id, "buyer_id": buyer_id, "invoice_date": invoice_date},
        ),
        (
            """
            MERGE (i:Invoice {id: $invoice_id})
            ON CREATE SET i.invoice_number = $invoice_number,
                          i.amount = $amount, i.date = date($invoice_date),
                          i.fraud_score = 0.0, i.status = 'pending',
                          i.created_at = datetime()
            ON MATCH SET i.amount = $amount, i.date = date($invoice_date),
                         i.updated_at = datetime()
            """,
            {
                "invoice_id": invoice_id,
                "invoice_number": invoice.get("invoice_number", ""),
                "amount": amount,
                "invoice_date": invoice_date,
            },
        ),
        (
            """
            MATCH (s:Entity {id: $supplier_id})
            MATCH (i:Invoice {id: $invoice_id})
            MERGE (s)-[r:ISSUED]->(i)
            ON CREATE SET r.date = date($invoice_date), r.amount = $amount
            """,
            {"supplier_id": supplier_id, "invoice_id": invoice_id, "invoice_date": invoice_date, "amount": amount},
        ),
    ]

    if lender_id:
        queries.extend(
            [
                (
                    """
                    MERGE (l:Entity {id: $lender_id})
                    ON CREATE SET l.name = $lender_name, l.type = 'lender',
                                  l.risk_score = 0.0, l.is_flagged = false, l.created_at = datetime()
                    ON MATCH SET l.name = $lender_name, l.updated_at = datetime()
                    """,
                    {"lender_id": lender_id, "lender_name": invoice.get("lender_name", "")},
                ),
                (
                    """
                    MATCH (i:Invoice {id: $invoice_id})
                    MATCH (l:Entity {id: $lender_id})
                    MERGE (i)-[r:FINANCED_BY]->(l)
                    ON CREATE SET r.amount = $amount, r.created_at = datetime()
                    """,
                    {"invoice_id": invoice_id, "lender_id": lender_id, "amount": amount},
                ),
            ]
        )

    for query, params in queries:
        await run_query(query, params)
