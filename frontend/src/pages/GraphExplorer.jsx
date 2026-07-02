import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import CytoscapeComponent from "react-cytoscapejs";

import { getEntityNetwork } from "../services/graphService";

export default function GraphExplorer() {
  const [entityId, setEntityId] = useState("");
  const [depth, setDepth] = useState(2);

  const { data, refetch, isFetching, error } = useQuery({
    queryKey: ["graph", entityId, depth],
    queryFn: () => getEntityNetwork(entityId, depth),
    enabled: false,
  });

  const elements = [
    ...((data?.nodes || []).map((node) => ({
      data: { id: node.id, label: node.label, risk: node.risk_score },
    })) || []),
    ...((data?.edges || []).map((edge, index) => ({
      data: { id: `${edge.source}-${edge.target}-${index}`, source: edge.source, target: edge.target },
    })) || []),
  ];

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
          <button className="button" onClick={() => refetch()} disabled={!entityId || isFetching}>
            {isFetching ? "Loading..." : "Load Network"}
          </button>
        </div>
      </div>

      {error ? <div className="card" style={{ marginTop: 16 }}>Failed to load graph: {error.message}</div> : null}

      <div className="card" style={{ marginTop: 16 }}>
        <div style={{ height: 480 }}>
          <CytoscapeComponent
            elements={elements}
            style={{ width: "100%", height: "100%" }}
            layout={{ name: "cose" }}
            stylesheet={[
              {
                selector: "node",
                style: {
                  label: "data(label)",
                  "background-color": "#2563eb",
                  color: "#fff",
                  "text-valign": "center",
                  "text-halign": "center",
                  width: 46,
                  height: 46,
                  "font-size": 10,
                },
              },
              {
                selector: "edge",
                style: {
                  width: 2,
                  "line-color": "#64748b",
                  "target-arrow-color": "#64748b",
                  "target-arrow-shape": "triangle",
                  "curve-style": "bezier",
                },
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
