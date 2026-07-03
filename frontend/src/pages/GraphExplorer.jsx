import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";

import GraphCanvas from "../components/GraphCanvas";
import { detectCarousel, getEntityNetwork } from "../services/graphService";

export default function GraphExplorer() {
  const { entityId: routeEntityId } = useParams();
  const [entityId, setEntityId] = useState(routeEntityId || "");
  const [depth, setDepth] = useState(2);
  const [tierFilter, setTierFilter] = useState("all");
  const [selectedNode, setSelectedNode] = useState(null);

  const { data, refetch, isFetching, error } = useQuery({
    queryKey: ["graph", entityId, depth],
    queryFn: () => getEntityNetwork(entityId, depth),
    enabled: false,
  });

  const carouselQuery = useQuery({
    queryKey: ["carousel", entityId],
    queryFn: () => detectCarousel(entityId),
    enabled: false,
  });

  const filteredNodes = (data?.nodes || []).filter((node) => {
    if (tierFilter === "all") return true;
    const numeric = Number(tierFilter);
    return Number(node.tier || 0) === numeric;
  });

  const visibleIds = new Set(filteredNodes.map((n) => n.id));
  const filteredEdges = (data?.edges || []).filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
  );

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Graph Explorer</h2>
          <p className="muted">Inspect buyer-supplier network structure and connected entities.</p>
        </div>
      </div>

      <div className="card">
        <div className="grid grid-3">
          <input
            value={entityId}
            onChange={(e) => setEntityId(e.target.value)}
            placeholder="Enter entity UUID"
          />
          <select value={depth} onChange={(e) => setDepth(Number(e.target.value))}>
            <option value={1}>1 hop</option>
            <option value={2}>2 hops</option>
            <option value={3}>3 hops</option>
          </select>
          <select value={tierFilter} onChange={(e) => setTierFilter(e.target.value)}>
            <option value="all">All tiers</option>
            <option value="1">Tier 1</option>
            <option value="2">Tier 2</option>
            <option value="3">Tier 3</option>
          </select>
          <button className="button" onClick={() => refetch()} disabled={!entityId || isFetching}>
            {isFetching ? "Loading..." : "Load Network"}
          </button>
          <button
            className="button secondary"
            onClick={() => carouselQuery.refetch()}
            disabled={!entityId || carouselQuery.isFetching}
          >
            {carouselQuery.isFetching ? "Checking..." : "Highlight Carousel"}
          </button>
        </div>
      </div>

      {error ? <div className="card" style={{ marginTop: 16 }}>Failed to load graph: {error.message}</div> : null}

      <div className="card" style={{ marginTop: 16 }}>
        <GraphCanvas nodes={filteredNodes} edges={filteredEdges} onNodeSelect={setSelectedNode} />
      </div>

      <div className="grid grid-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h3>Entity Insights</h3>
          <p className="muted">Nodes: {filteredNodes.length} · Edges: {filteredEdges.length}</p>
          {carouselQuery.data?.has_carousel ? (
            <p>
              Carousel rings detected: <strong>{carouselQuery.data.ring_count}</strong>
            </p>
          ) : (
            <p className="muted">No carousel ring detected yet (run highlight check).</p>
          )}
        </div>
        <div className="card">
          <h3>Selected Node</h3>
          {selectedNode ? (
            <>
              <p>
                <strong>{selectedNode.label}</strong>
              </p>
              <p className="muted">ID: {selectedNode.id}</p>
              <p className="muted">Type: {selectedNode.type || "entity"}</p>
              <p className="muted">Tier: {selectedNode.tier || "N/A"}</p>
              <p className="muted">Risk Score: {(Number(selectedNode.risk || 0) * 100).toFixed(1)}%</p>
            </>
          ) : (
            <p className="muted">Click a node in the graph to inspect details.</p>
          )}
        </div>
      </div>
    </div>
  );
}
