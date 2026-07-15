"""
backend/db/graph.py
-------------------
In-memory NetworkX DiGraph rebuilt from existing Supabase tables
(invoices + entities) on first use. No dedicated graph_nodes/graph_edges
tables required — the invoice/entity data IS the graph.

Public API
----------
  get_graph()                          -> nx.DiGraph   (load-on-first-use)
  upsert_node(id, type, name, props)   -> update in-memory node
  upsert_edge(src, dst, rel, props)    -> update in-memory edge
  close_graph()                        -> clear in-memory graph (shutdown)
  graph_stats()                        -> dict with node/edge counts
"""
from __future__ import annotations

import asyncio
from typing import Any

import networkx as nx

from backend.db.supabase import get_supabase_client

_graph: nx.DiGraph | None = None
_load_lock = asyncio.Lock()


async def get_graph() -> nx.DiGraph:
    """Return the in-memory DiGraph, loading from Supabase on first call."""
    global _graph
    if _graph is not None:
        return _graph
    async with _load_lock:
        if _graph is not None:
            return _graph
        _graph = await _load_graph_from_supabase()
    return _graph


async def _load_graph_from_supabase() -> nx.DiGraph:
    """
    Build the NetworkX graph from the existing `entities` and `invoices`
    tables — no dedicated graph tables needed.
    """
    G: nx.DiGraph = nx.DiGraph()
    client = get_supabase_client()

    # --- Load entity nodes ---
    try:
        entity_rows = (
            client.table("entities")
            .select("id, name, entity_type, risk_score, is_flagged, tier, sector")
            .execute()
            .data or []
        )
        for row in entity_rows:
            G.add_node(
                str(row["id"]),
                node_type=row.get("entity_type", "entity"),
                name=row.get("name", ""),
                risk_score=float(row.get("risk_score") or 0.0),
                is_flagged=bool(row.get("is_flagged", False)),
                tier=row.get("tier"),
                sector=row.get("sector", ""),
            )
    except Exception:
        pass

    # --- Load invoice nodes + edges ---
    try:
        invoice_rows = (
            client.table("invoices")
            .select(
                "id, invoice_number, supplier_id, buyer_id, lender_id, "
                "invoice_date, amount, fraud_score, status"
            )
            .execute()
            .data or []
        )
        for row in invoice_rows:
            inv_id = str(row["id"])
            supplier_id = str(row["supplier_id"]) if row.get("supplier_id") else None
            buyer_id = str(row["buyer_id"]) if row.get("buyer_id") else None
            lender_id = str(row["lender_id"]) if row.get("lender_id") else None
            amount = float(row.get("amount") or 0)
            inv_date = str(row.get("invoice_date") or "2024-01-01")

            # Invoice node
            G.add_node(
                inv_id,
                node_type="invoice",
                name=row.get("invoice_number", ""),
                amount=amount,
                invoice_date=inv_date,
                fraud_score=float(row.get("fraud_score") or 0.0),
                status=row.get("status", "pending"),
            )

            if supplier_id and buyer_id:
                # Supplier → Buyer supply edge
                G.add_edge(
                    supplier_id,
                    buyer_id,
                    rel_type="SUPPLIES_TO",
                    invoice_count=1,
                    since=inv_date,
                )
                # Supplier → Invoice issued edge
                G.add_edge(
                    supplier_id,
                    inv_id,
                    rel_type="ISSUED",
                    date=inv_date,
                    amount=amount,
                )

            if lender_id and inv_id:
                # Invoice → Lender financed edge
                G.add_edge(
                    inv_id,
                    lender_id,
                    rel_type="FINANCED_BY",
                    amount=amount,
                )
    except Exception:
        pass

    return G


# ---------------------------------------------------------------------------
# Write helpers — in-memory only (source of truth is invoices/entities)
# ---------------------------------------------------------------------------

async def upsert_node(
    node_id: str,
    node_type: str = "entity",
    name: str = "",
    risk_score: float = 0.0,
    is_flagged: bool = False,
    props: dict[str, Any] | None = None,
) -> None:
    """Add/update a node in the in-memory graph."""
    G = await get_graph()
    G.add_node(
        node_id,
        node_type=node_type,
        name=name,
        risk_score=risk_score,
        is_flagged=is_flagged,
        **(props or {}),
    )


async def upsert_edge(
    source_id: str,
    target_id: str,
    rel_type: str,
    props: dict[str, Any] | None = None,
) -> None:
    """Add/update a directed edge in the in-memory graph."""
    G = await get_graph()
    G.add_edge(source_id, target_id, rel_type=rel_type, **(props or {}))


async def close_graph() -> None:
    """Clear the in-memory graph (called on app shutdown)."""
    global _graph
    _graph = None


def graph_stats() -> dict[str, int]:
    """Return node/edge counts without triggering a load."""
    if _graph is None:
        return {"node_count": 0, "edge_count": 0}
    return {
        "node_count": _graph.number_of_nodes(),
        "edge_count": _graph.number_of_edges(),
    }
