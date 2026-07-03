export default function CascadeTimeline({ cascadePath = [] }) {
  if (!cascadePath.length) {
    return <p className="muted">No cascade chain detected for this case.</p>;
  }

  return (
    <div className="timeline">
      {cascadePath.map((item, index) => (
        <div key={`${item.downstream_invoice_id || index}`} className="timeline-item">
          <div className="timeline-dot" />
          <div className="timeline-content">
            <strong>{item.invoice_number || item.downstream_invoice_id || "Invoice"}</strong>
            <div className="muted">
              {item.downstream_supplier || "Unknown supplier"} · INR {Number(item.amount || 0).toLocaleString()}
            </div>
            <div className="muted">{item.date || "Date unavailable"}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
