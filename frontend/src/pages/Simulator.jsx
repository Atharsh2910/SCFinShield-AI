import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import { analyzeFraud } from "../services/fraudService";
import { generateDataset, listScenarioTemplates, runScenario } from "../services/simulatorService";
import FraudScoreBadge from "../components/FraudScoreBadge";

export default function Simulator() {
  const [scenario, setScenario] = useState("phantom_invoice");
  const [n, setN] = useState(10);
  const [amountMin, setAmountMin] = useState(100000);
  const [amountMax, setAmountMax] = useState(1000000);
  const [lenderCount, setLenderCount] = useState(2);
  const [fraudRate, setFraudRate] = useState(0.15);
  const [persist, setPersist] = useState(false);
  const [result, setResult] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const scenariosQuery = useQuery({
    queryKey: ["simulator-scenarios"],
    queryFn: listScenarioTemplates,
  });

  async function handleGenerate() {
    setLoading(true);
    setError("");
    try {
      const data = await generateDataset({
        scenario,
        n: Number(n),
        persist,
        fraud_rate: Number(fraudRate),
        amount_min: Number(amountMin),
        amount_max: Number(amountMax),
        lender_count: Number(lenderCount),
      });
      setResult(data);
      setAnalysisResult(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleRunAnalysis() {
    if (!result?.invoice_ids?.length) {
      setError("Persist synthetic invoices first to run analysis.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const analyses = [];
      for (const invoiceId of result.invoice_ids) {
        const response = await analyzeFraud(invoiceId);
        analyses.push(response);
      }
      setAnalysisResult(analyses);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // Synthetic invoices from scenario output
  const syntheticInvoices = result?.invoices || [];

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Simulator</h2>
          <p className="muted">Generate demo datasets for fraud scenarios and optional persistence.</p>
        </div>
      </div>

      <div className="card">
        <div className="grid grid-3">
          <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
            {(scenariosQuery.data?.scenarios || []).map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}
              </option>
            ))}
          </select>
          <input type="number" min="1" max="100" value={n} onChange={(e) => setN(e.target.value)} />
          <select value={persist ? "yes" : "no"} onChange={(e) => setPersist(e.target.value === "yes")}>
            <option value="no">Preview only</option>
            <option value="yes">Persist to backend</option>
          </select>
        </div>
        <div className="grid grid-3" style={{ marginTop: 12 }}>
          <input
            type="number"
            min="10000"
            step="10000"
            value={amountMin}
            onChange={(e) => setAmountMin(e.target.value)}
            placeholder="Amount min"
          />
          <input
            type="number"
            min="10000"
            step="10000"
            value={amountMax}
            onChange={(e) => setAmountMax(e.target.value)}
            placeholder="Amount max"
          />
          <input
            type="number"
            min="1"
            max="10"
            value={lenderCount}
            onChange={(e) => setLenderCount(e.target.value)}
            placeholder="Lender count"
          />
        </div>
        <div className="form-row" style={{ marginTop: 12 }}>
          <label htmlFor="fraudRate">Fraud Rate</label>
          <input
            id="fraudRate"
            type="number"
            min="0"
            max="1"
            step="0.01"
            value={fraudRate}
            onChange={(e) => setFraudRate(e.target.value)}
          />
        </div>
        <div style={{ marginTop: 16 }}>
          <button className="button" onClick={handleGenerate} disabled={loading}>
            {loading ? "Generating..." : "Generate Scenario"}
          </button>
          <button
            className="button secondary"
            style={{ marginLeft: 8 }}
            onClick={() => runScenario({ scenario, n: Number(n), persist })}
            disabled={loading}
          >
            Run Scenario Alias
          </button>
          <button
            className="button"
            style={{ marginLeft: 8 }}
            onClick={handleRunAnalysis}
            disabled={loading || !result?.invoice_ids?.length}
          >
            Run Analysis
          </button>
        </div>
        {error ? <p style={{ color: "#fca5a5" }}>{error}</p> : null}
      </div>

      {result ? (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Scenario Output</h3>
          <p className="muted" style={{ marginBottom: 12 }}>
            Scenario: <strong>{result.scenario || scenario}</strong> &nbsp;·&nbsp;
            Generated: <strong>{result.count ?? syntheticInvoices.length ?? 0}</strong> invoices
            {result.persisted ? " · Persisted to backend" : " · Preview only"}
          </p>

          {syntheticInvoices.length > 0 ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Invoice #</th>
                    <th>Supplier</th>
                    <th>Buyer</th>
                    <th>Amount (INR)</th>
                    <th>Scenario</th>
                  </tr>
                </thead>
                <tbody>
                  {syntheticInvoices.map((inv, idx) => (
                    <tr key={inv.invoice_number || idx}>
                      <td>{inv.invoice_number || `—`}</td>
                      <td>{inv.supplier_name || inv.supplier_id || "—"}</td>
                      <td>{inv.buyer_name || inv.buyer_id || "—"}</td>
                      <td>{Number(inv.amount || 0).toLocaleString()}</td>
                      <td>{inv.scenario || scenario}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">
              {result.invoice_ids?.length
                ? `${result.invoice_ids.length} invoice IDs persisted. Click "Run Analysis" to score them.`
                : "No invoice details returned. Enable persist and re-generate to see scored invoices."}
            </p>
          )}
        </div>
      ) : null}

      {analysisResult ? (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Analysis Results</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Invoice ID</th>
                  <th>Decision</th>
                  <th>Ensemble Score</th>
                  <th>Fraud Patterns</th>
                </tr>
              </thead>
              <tbody>
                {analysisResult.map((res, idx) => {
                  const patterns = Array.isArray(res.fraud_patterns)
                    ? res.fraud_patterns.join(", ")
                    : res.fraud_patterns || "—";
                  const score = res.ensemble_score ?? res.fraud_score ?? 0;
                  return (
                    <tr key={res.invoice_id || idx}>
                      <td>{res.invoice_id || "—"}</td>
                      <td>
                        <FraudScoreBadge
                          decision={res.fraud_decision || res.decision || "PASS"}
                          score={score}
                        />
                      </td>
                      <td>{(Number(score) * 100).toFixed(1)}%</td>
                      <td>{patterns}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </div>
  );
}
