"""
backend/services/graph/analytics.py
-------------------------------------
Fraud-graph analytics powered by NetworkX (replaces Neo4j Cypher queries).

Algorithms
----------
- detect_carousel_trades  : simple_cycles() on supplier→buyer subgraph
- trace_cascade           : descendants() from invoice node, filtered by date
- get_entity_network      : ego_graph() for neighbourhood traversal
- get_concentration_risk  : FINANCED_BY edge aggregation by lender
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import networkx as nx

from backend.db.graph import get_graph, graph_stats


# ---------------------------------------------------------------------------
# Carousel / circular trade detection
# ---------------------------------------------------------------------------

async def detect_carousel_trades(supplier_id: str) -> dict[str, Any]:
    """
    Find circular supply chains involving the given supplier using
    simple cycle detection on the SUPPLIES_TO subgraph.
    """
    G = await get_graph()

    # Build supplier-only subgraph (SUPPLIES_TO edges between entity nodes)
    supply_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("rel_type") == "SUPPLIES_TO"
    ]
    supply_graph = nx.DiGraph(supply_edges)

    if supplier_id not in supply_graph:
        return {"has_carousel": False, "carousel_rings": [], "ring_count": 0}

    # Find all simple cycles (up to length 6) that include this supplier
    rings = []
    try:
        for cycle in nx.simple_cycles(supply_graph):
            if supplier_id in cycle and 2 <= len(cycle) <= 6:
                ring_members = [
                    {
                        "id": nid,
                        "name": G.nodes[nid].get("name", nid) if nid in G else nid,
                    }
                    for nid in cycle
                ]
                rings.append({"ring_members": ring_members, "ring_length": len(cycle)})
            if len(rings) >= 10:
                break
    except Exception:
        pass

    return {
        "has_carousel": len(rings) > 0,
        "carousel_rings": rings,
        "ring_count": len(rings),
    }


# ---------------------------------------------------------------------------
# Cascade / downstream exposure tracing
# ---------------------------------------------------------------------------

async def trace_cascade(invoice_id: str, hours: int = 72) -> dict[str, Any]:
    """
    Trace downstream invoices reachable from this invoice's supplier
    within the given time window.
    """
    G = await get_graph()

    if invoice_id not in G:
        return {
            "cascade_depth": 0,
            "downstream_invoices": [],
            "total_cascade_exposure": 0.0,
            "has_cascade": False,
        }

    # Determine the source invoice's date
    invoice_attrs = G.nodes[invoice_id]
    try:
        inv_date = date.fromisoformat(str(invoice_attrs.get("invoice_date", "2024-01-01")))
    except ValueError:
        inv_date = date.today()

    cutoff = inv_date + timedelta(hours=hours)

    # Find the supplier(s) who issued this invoice (ISSUED edges pointing to invoice)
    issuing_suppliers = [
        u for u, v, d in G.edges(data=True)
        if v == invoice_id and d.get("rel_type") == "ISSUED"
    ]

    downstream: list[dict[str, Any]] = []
    seen: set[str] = {invoice_id}

    for supplier_id in issuing_suppliers:
        # All invoices issued by buyers that this supplier supplies to
        for buyer_id in G.successors(supplier_id):
            edge_data = G.edges[supplier_id, buyer_id]
            if edge_data.get("rel_type") != "SUPPLIES_TO":
                continue
            # Find invoices connected to this buyer or its downstream
            for node in nx.descendants(G, buyer_id):
                node_data = G.nodes.get(node, {})
                if node_data.get("node_type") != "invoice" or node in seen:
                    continue
                try:
                    node_date = date.fromisoformat(
                        str(node_data.get("invoice_date", "2024-01-01"))
                    )
                except ValueError:
                    continue
                if inv_date - timedelta(days=1) <= node_date <= cutoff:
                    seen.add(node)
                    downstream.append({
                        "downstream_invoice_id": node,
                        "invoice_number": node_data.get("name", ""),
                        "amount": float(node_data.get("amount", 0) or 0),
                        "date": str(node_date),
                        "downstream_supplier": supplier_id,
                    })
                if len(downstream) >= 20:
                    break

    total_exposure = sum(row["amount"] for row in downstream)
    return {
        "cascade_depth": len(downstream),
        "downstream_invoices": downstream,
        "total_cascade_exposure": total_exposure,
        "has_cascade": len(downstream) > 0,
    }


# ---------------------------------------------------------------------------
# Entity neighbourhood network
# ---------------------------------------------------------------------------

async def get_entity_network(entity_id: str, depth: int = 2) -> dict[str, Any]:
    """
    Return the ego-graph (neighbours within `depth` hops) around an entity.
    """
    G = await get_graph()
    bounded_depth = max(1, min(depth, 3))

    if entity_id not in G:
        return {"nodes": [], "edges": [], "center_entity_id": entity_id, "fraud_clusters": []}

    ego = nx.ego_graph(G.to_undirected(), entity_id, radius=bounded_depth)

    nodes = [
        {
            "id": n,
            "label": G.nodes[n].get("name", n) if n in G else n,
            "type": G.nodes[n].get("node_type", "entity") if n in G else "entity",
            "risk_score": float(G.nodes[n].get("risk_score", 0.0) if n in G else 0.0),
            "is_flagged": bool(G.nodes[n].get("is_flagged", False) if n in G else False),
        }
        for n in ego.nodes()
    ]

    edges = [
        {
            "source": u,
            "target": v,
            "weight": int(G.edges[u, v].get("invoice_count", 1)) if G.has_edge(u, v) else 1,
        }
        for u, v in ego.edges()
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "center_entity_id": entity_id,
        "fraud_clusters": [],
    }


# ---------------------------------------------------------------------------
# Concentration / lender risk
# ---------------------------------------------------------------------------

async def get_concentration_risk(lender_id: str) -> list[dict[str, Any]]:
    """
    Aggregate financing by supplier for a given lender.
    Returns list ordered by total financed amount descending.
    """
    G = await get_graph()

    if lender_id not in G:
        return []

    # Find all invoices that point to this lender (FINANCED_BY edges)
    invoice_ids = [
        u for u, v, d in G.edges(data=True)
        if v == lender_id and d.get("rel_type") == "FINANCED_BY"
    ]

    # For each invoice, find the issuing supplier (ISSUED edges)
    supplier_totals: dict[str, dict[str, Any]] = {}
    for inv_id in invoice_ids:
        inv_data = G.nodes.get(inv_id, {})
        amount = float(inv_data.get("amount", 0) or 0)

        for u, v, d in G.in_edges(inv_id, data=True):
            if d.get("rel_type") == "ISSUED":
                s_data = G.nodes.get(u, {})
                if u not in supplier_totals:
                    supplier_totals[u] = {
                        "supplier_id": u,
                        "supplier_name": s_data.get("name", u),
                        "total_financed": 0.0,
                        "invoice_count": 0,
                    }
                supplier_totals[u]["total_financed"] += amount
                supplier_totals[u]["invoice_count"] += 1

    result = sorted(supplier_totals.values(), key=lambda x: x["total_financed"], reverse=True)
    return result[:20]


# ---------------------------------------------------------------------------
# Graph-level stats (used by the /graph/stats endpoint)
# ---------------------------------------------------------------------------

async def get_graph_stats() -> dict[str, Any]:
    await get_graph()  # ensure loaded
    stats = graph_stats()
    return {
        "node_count": stats["node_count"],
        "edge_count": stats["edge_count"],
        "fraud_clusters": [],
    }
