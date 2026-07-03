import { useState } from "react";

import AlertNarrative from "../components/AlertNarrative";
import FileUploadZone from "../components/FileUploadZone";
import FraudScoreBadge from "../components/FraudScoreBadge";
import SignalBreakdown from "../components/SignalBreakdown";
import { analyzeInvoice, uploadInvoice } from "../services/invoiceService";

export default function InvoiceUpload() {
  const [file, setFile] = useState(null);
  const [lenderName, setLenderName] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [previewRows, setPreviewRows] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function handleFileSelected(selectedFile) {
    setFile(selectedFile);
    setPreviewRows([]);

    if (!selectedFile) return;
    const name = selectedFile.name.toLowerCase();
    try {
      if (name.endsWith(".csv")) {
        const text = await selectedFile.text();
        const lines = text.split(/\r?\n/).filter(Boolean);
        const headers = (lines[0] || "").split(",").map((h) => h.trim());
        const rows = lines.slice(1, 6).map((line) => {
          const cells = line.split(",");
          return Object.fromEntries(headers.map((h, idx) => [h, cells[idx] ?? ""]));
        });
        setPreviewRows(rows);
      } else if (name.endsWith(".json")) {
        const text = await selectedFile.text();
        const payload = JSON.parse(text);
        const rows = Array.isArray(payload) ? payload.slice(0, 5) : [payload];
        setPreviewRows(rows);
      } else {
        setPreviewRows([{ notice: "Preview not available for PDF. File will be parsed server-side." }]);
      }
    } catch {
      setPreviewRows([{ notice: "Unable to preview file, but upload is still available." }]);
    }
  }

  async function handleSubmit() {
    if (!file) return;

    setIsSubmitting(true);
    setProgress(5);
    setError("");
    setResult(null);

    try {
      const upload = await uploadInvoice(file, lenderName);
      setProgress(45);
      const invoiceIds = upload.invoice_ids || [];
      const analyses = [];

      for (const invoiceId of invoiceIds) {
        const analysis = await analyzeInvoice(invoiceId);
        analyses.push({ invoiceId, analysis });
        setProgress(Math.min(45 + Math.round((analyses.length / Math.max(invoiceIds.length, 1)) * 55), 100));
      }

      setResult({ upload, analyses });
      setProgress(100);
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

        <FileUploadZone onFileSelected={handleFileSelected} disabled={isSubmitting} />

        {previewRows.length ? (
          <div style={{ marginTop: 16 }}>
            <h4>File Preview</h4>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {Object.keys(previewRows[0]).map((key) => (
                      <th key={key}>{key}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewRows.map((row, idx) => (
                    <tr key={idx}>
                      {Object.keys(previewRows[0]).map((key) => (
                        <td key={`${idx}-${key}`}>{String(row[key] ?? "")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}

        <div style={{ marginTop: 16 }}>
          <button className="button" onClick={handleSubmit} disabled={!file || isSubmitting}>
            {isSubmitting ? "Uploading and analyzing..." : "Upload and Analyze"}
          </button>
        </div>
        {isSubmitting ? (
          <div style={{ marginTop: 12 }}>
            <p className="muted">Progress: {progress}%</p>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>
        ) : null}

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
          {result.analyses.map((item) => (
            <div key={`${item.invoiceId}-details`} className="card mini-card" style={{ marginTop: 12 }}>
              <h4>Invoice {item.invoiceId}</h4>
              <div className="grid grid-2">
                <div>
                  <h5>Signal Breakdown</h5>
                  <SignalBreakdown scores={item.analysis.ml_result?.individual_scores || {}} />
                </div>
                <div>
                  <h5>Alert Narrative</h5>
                  <AlertNarrative
                    narrative={item.analysis.alert_narrative || ""}
                    citations={item.analysis.regulation_citations || []}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
