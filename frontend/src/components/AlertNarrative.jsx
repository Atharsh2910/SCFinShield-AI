export default function AlertNarrative({ narrative = "", citations = [] }) {
  return (
    <div>
      <p style={{ whiteSpace: "pre-wrap" }}>{narrative || "Narrative not available."}</p>
      {citations.length ? (
        <div style={{ marginTop: 12 }}>
          <h4>Regulation Citations</h4>
          <div className="grid grid-2">
            {citations.map((citation, index) => (
              <div key={`${citation.title || citation.source}-${index}`} className="card mini-card">
                <strong>{citation.title || "Untitled citation"}</strong>
                <p className="muted">{citation.category || "regulation"}</p>
                <p className="muted">{citation.source || "source unavailable"}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
