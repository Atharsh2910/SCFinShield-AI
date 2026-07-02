import { useState } from "react";

import FileUploadZone from "../components/FileUploadZone";
import FraudScoreBadge from "../components/FraudScoreBadge";
import { analyzeInvoice, uploadInvoice } from "../services/invoiceService";

export default function InvoiceUpload() {
  const [file, setFile] = useState(null);
  const [lenderName, setLenderName] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function handleSubmit() {
    if (!file) return;

    setIsSubmitting(true);
    setError("");
    setResult(null);

    try {
      const upload = await uploadInvoice(file, lenderName);
      const invoiceIds = upload.invoice_ids || [];
      const analyses = [];

      for (const invoiceId of invoiceIds) {
        const analysis = await analyzeInvoice(invoiceId);
        analyses.push({ invoiceId, analysis });
      }

      setResult({ upload, analyses });
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Invoice Upload</h2>
          <p className="muted">Upload CSV, JSON, or PDF invoices and run analysis.</p>
        </div>
      </div>

      <div className="card">
        <div className="form-row">
          <label htmlFor="lenderName">Lender Name</label>
          <input
            id="lenderName"
            value={lenderName}
            onChange={(e) => setLenderName(e.target.value)}
            placeholder="Optional lender name"
          />
        </div>

        <FileUploadZone onFileSelected={setFile} disabled={isSubmitting} />

        <div style={{ marginTop: 16 }}>
          <button className="button" onClick={handleSubmit} disabled={!file || isSubmitting}>
            {isSubmitting ? "Uploading and analyzing..." : "Upload and Analyze"}
          </button>
        </div>

        {error ? <p style={{ color: "#fca5a5" }}>{error}</p> : null}
      </div>

      {result ? (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Results</h3>
          <p className="muted">
            Accepted {result.upload.accepted_count} invoice(s), rejected {result.upload.rejected_count}.
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Invoice ID</th>
                  <th>Decision</th>
                  <th>Patterns</th>
                </tr>
              </thead>
              <tbody>
                {result.analyses.map((item) => (
                  <tr key={item.invoiceId}>
                    <td>{item.invoiceId}</td>
                    <td>
                      <FraudScoreBadge
                        decision={item.analysis.fraud_decision}
                        score={item.analysis.ensemble_score || 0}
                      />
                    </td>
                    <td>{(item.analysis.fraud_patterns || []).join(", ") || "None"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </div>
  );
}
