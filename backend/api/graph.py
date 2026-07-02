from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.db.neo4j import run_query
from backend.schemas.graph import NetworkGraphResponse
from backend.services.graph.analytics import (
    detect_carousel_trades,
    get_concentration_risk,
    get_entity_network,
    trace_cascade,
)

router = APIRouter()


@router.get("/graph/network/{entity_id}", response_model=NetworkGraphResponse)
async def network(entity_id: str, depth: int = 2) -> dict[str, Any]:
    return await get_entity_network(entity_id=entity_id, depth=depth)


@router.get("/graph/carousel/{entity_id}")
async def carousel(entity_id: str) -> dict[str, Any]:
    return await detect_carousel_trades(supplier_id=entity_id)


@router.get("/graph/cascade/{invoice_id}")
async def cascade(invoice_id: str, hours: int = 72) -> dict[str, Any]:
    return await trace_cascade(invoice_id=invoice_id, hours=hours)


@router.get("/graph/concentration/{lender_id}")
async def concentration(lender_id: str) -> list[dict[str, Any]]:
    return await get_concentration_risk(lender_id=lender_id)


@router.get("/graph/stats")
async def graph_stats() -> dict[str, Any]:
    nodes = await run_query("MATCH (n) RETURN count(n) AS node_count")
    edges = await run_query("MATCH ()-[r]->() RETURN count(r) AS edge_count")
    return {
        "node_count": int(nodes[0].get("node_count", 0)) if nodes else 0,
        "edge_count": int(edges[0].get("edge_count", 0)) if edges else 0,
        "fraud_clusters": [],
    }

