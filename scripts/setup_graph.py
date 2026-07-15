"""
scripts/setup_graph.py
-----------------------
Verifies the NetworkX graph loads correctly from existing Supabase data.
No DDL is needed — the graph is built from the entities + invoices tables.

Run after setup_supabase.py and seed_demo_data.py.
"""
from __future__ import annotations

import asyncio

from backend.db.graph import get_graph, graph_stats


async def setup_graph() -> None:
    print("Loading NetworkX graph from Supabase (entities + invoices)...")
    G = await get_graph()
    stats = graph_stats()
    print(
        f"Graph loaded successfully:\n"
        f"  Nodes : {stats['node_count']}\n"
        f"  Edges : {stats['edge_count']}\n"
    )

    # Report node type breakdown
    type_counts: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        t = data.get("node_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, count in sorted(type_counts.items()):
        print(f"  {t:12s}: {count} node(s)")

    # Report edge type breakdown
    edge_counts: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        r = data.get("rel_type", "unknown")
        edge_counts[r] = edge_counts.get(r, 0) + 1
    for r, count in sorted(edge_counts.items()):
        print(f"  {r:20s}: {count} edge(s)")


if __name__ == "__main__":
    asyncio.run(setup_graph())
