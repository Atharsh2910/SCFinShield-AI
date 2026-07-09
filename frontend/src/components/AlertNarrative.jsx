export default function AlertNarrative({ narrative = "", citations = [] }) {
  return (
    <div>
      <p style={{ whiteSpace: "pre-wrap", lineHeight: 1.7, color: "#cbd5e1" }}>
        {narrative || (
          <span className="muted">No alert narrative generated yet.</span>
        )}
      </p>
      {citations.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <p
            style={{
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: "#64748b",
              fontWeight: 600,
              marginBottom: 10,
            }}
          >
            📚 Regulation Citations
          </p>
          <div className="grid grid-2">
            {citations.map((citation, index) => (
              <div
                key={`${citation.title || citation.source}-${index}`}
                className="mini-card"
                style={{ padding: "12px 14px" }}
              >
                <div style={{ fontSize: 18, marginBottom: 6 }}>📋</div>
                <strong
                  style={{
                    fontSize: 13,
                    color: "#e2e8f0",
                    display: "block",
                    marginBottom: 4,
                  }}
                >
                  {citation.title || "Untitled citation"}
                </strong>
                <p className="muted" style={{ marginBottom: 2 }}>
                  {citation.category || "regulation"}
                </p>
                <p className="muted">{citation.source || "source unavailable"}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
