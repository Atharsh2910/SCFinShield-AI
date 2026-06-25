from typing import Any

from backend.db.neo4j import run_query


async def detect_carousel_trades(supplier_id: str) -> dict[str, Any]:
    query = """
    MATCH (start:Entity {id: $supplier_id})
    MATCH path = (start)-[:SUPPLIES_TO*2..6]->(start)
    RETURN [node IN nodes(path) | {id: node.id, name: node.name}] as ring_members,
           length(path) as ring_length
    LIMIT 10
    """
    results = await run_query(query, {"supplier_id": supplier_id})
    return {
        "has_carousel": len(results) > 0,
        "carousel_rings": results,
        "ring_count": len(results),
    }


async def trace_cascade(invoice_id: str, hours: int = 72) -> dict[str, Any]:
    invoice_result = await run_query(
        "MATCH (i:Invoice {id: $id}) RETURN i.date as date",
        {"id": invoice_id},
    )
    invoice_date = str(invoice_result[0]["date"]) if invoice_result else "2024-01-01"

    query = """
    MATCH (i:Invoice {id: $invoice_id})<-[:ISSUED]-(s:Entity)-[:SUPPLIES_TO]->(b:Entity)
    MATCH (b)<-[:SUPPLIES_TO]-(s2:Entity)-[:ISSUED]->(i2:Invoice)
    WHERE i2.id <> $invoice_id
      AND i2.date >= date($invoice_date) - duration('P1D')
      AND i2.date <= date($invoice_date) + duration({hours: $hours})
    RETURN i2.id as downstream_invoice_id,
           i2.invoice_number as invoice_number,
           i2.amount as amount,
           i2.date as date,
           s2.name as downstream_supplier
    LIMIT 20
    """
    downstream = await run_query(query, {"invoice_id": invoice_id, "invoice_date": invoice_date, "hours": hours})
    total_exposure = sum(float(row.get("amount", 0) or 0) for row in downstream)
    return {
        "cascade_depth": len(downstream),
        "downstream_invoices": downstream,
        "total_cascade_exposure": total_exposure,
        "has_cascade": len(downstream) > 0,
    }


async def get_entity_network(entity_id: str, depth: int = 2) -> dict[str, Any]:
    bounded_depth = max(1, min(depth, 3))
    query = f"""
    MATCH path = (center:Entity {{id: $entity_id}})-[:SUPPLIES_TO*0..{bounded_depth}]-(connected:Entity)
    WITH collect(path) as paths
    UNWIND paths as path
    UNWIND nodes(path) as n
    WITH paths, collect(DISTINCT n) as nodes
    UNWIND paths as path
    UNWIND relationships(path) as r
    RETURN nodes,
           collect(DISTINCT {{
             source: startNode(r).id,
             target: endNode(r).id,
             weight: coalesce(r.invoice_count, 1)
           }}) as edges
    """
    results = await run_query(query, {"entity_id": entity_id})
    if not results:
        return {"nodes": [], "edges": [], "center_entity_id": entity_id, "fraud_clusters": []}

    row = results[0]
    nodes = [
        {
            "id": node.get("id"),
            "label": node.get("name", ""),
            "type": node.get("type", "entity"),
            "risk_score": float(node.get("risk_score", 0.0) or 0.0),
            "is_flagged": bool(node.get("is_flagged", False)),
        }
        for node in row.get("nodes", [])
        if node.get("id")
    ]
    return {
        "nodes": nodes,
        "edges": row.get("edges", []),
        "center_entity_id": entity_id,
        "fraud_clusters": [],
    }


async def get_concentration_risk(lender_id: str) -> list[dict[str, Any]]:
    query = """
    MATCH (i:Invoice)-[:FINANCED_BY]->(l:Entity {id: $lender_id})
    MATCH (s:Entity)-[:ISSUED]->(i)
    WITH s.id as supplier_id, s.name as supplier_name,
         sum(i.amount) as total_financed, count(i) as invoice_count
    ORDER BY total_financed DESC
    RETURN supplier_id, supplier_name, total_financed, invoice_count
    LIMIT 20
    """
    return await run_query(query, {"lender_id": lender_id})
