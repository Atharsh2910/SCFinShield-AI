import CytoscapeComponent from "react-cytoscapejs";

export default function GraphCanvas({ nodes = [], edges = [], onNodeSelect }) {
  const elements = [
    ...nodes.map((node) => ({
      data: {
        id: node.id,
        label: node.label,
        risk: Number(node.risk_score || 0),
        type: node.type || "entity",
        tier: node.tier ?? "",
      },
    })),
    ...edges.map((edge, index) => ({
      data: {
        id: `${edge.source}-${edge.target}-${index}`,
        source: edge.source,
        target: edge.target,
        weight: edge.weight || 1,
        label: String(edge.weight || 1),
      },
    })),
  ];

  return (
    <div style={{ height: 480 }}>
      <CytoscapeComponent
        elements={elements}
        style={{ width: "100%", height: "100%" }}
        layout={{ name: "cose" }}
        cy={(cy) => {
          if (!onNodeSelect) return;
          cy.on("tap", "node", (evt) => {
            const data = evt.target.data();
            onNodeSelect(data);
          });
        }}
        stylesheet={[
          {
            selector: "node",
            style: {
              label: "data(label)",
              "background-color": "#22c55e",
              color: "#fff",
              "text-valign": "center",
              "text-halign": "center",
              width: 48,
              height: 48,
              "font-size": 10,
            },
          },
          {
            selector: "edge",
            style: {
              width: "mapData(weight, 1, 20, 1, 8)",
              label: "data(label)",
              "font-size": 9,
              color: "#cbd5e1",
              "text-rotation": "autorotate",
              "text-margin-y": -8,
              "line-color": "#64748b",
              "target-arrow-color": "#64748b",
              "target-arrow-shape": "triangle",
              "curve-style": "bezier",
            },
          },
          {
            selector: "node[risk >= 0.85]",
            style: {
              "background-color": "#ef4444",
            },
          },
          {
            selector: "node[risk >= 0.7][risk < 0.85]",
            style: {
              "background-color": "#f59e0b",
            },
          },
        ]}
      />
    </div>
  );
}
