export default function CascadeTimeline({ cascadePath = [] }) {
  if (!cascadePath.length) {
    return (
      <div className="state-card">
        <div style={{ fontSize: 32 }}>✅</div>
        <p className="muted">No cascade chain detected for this case.</p>
      </div>
    );
  }

  return (
    <div className="timeline">
      {cascadePath.map((item, index) => (
        <div
          key={`${item.downstream_invoice_id || index}`}
          className="timeline-item"
          style={{ animationDelay: `${index * 60}ms` }}
        >
          <div className="timeline-dot" />
          <div className="timeline-content">
            <strong
              style={{ display: "block", color: "#e2e8f0", marginBottom: 4 }}
            >
              {item.invoice_number || item.downstream_invoice_id || "Invoice"}
            </strong>
            <div className="muted">
              {item.downstream_supplier || "Unknown supplier"}
            </div>
            <div style={{ display: "flex", gap: 16, marginTop: 6 }}>
              <span
                style={{ color: "#f59e0b", fontWeight: 600, fontSize: 13 }}
              >
                ₹{Number(item.amount || 0).toLocaleString("en-IN")}
              </span>
              <span className="muted">{item.date || ""}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
