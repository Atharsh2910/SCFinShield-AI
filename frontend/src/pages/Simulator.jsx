import { useState } from "react";

import { generateDataset } from "../services/simulatorService";

export default function Simulator() {
  const [scenario, setScenario] = useState("phantom_invoice");
  const [n, setN] = useState(10);
  const [persist, setPersist] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleGenerate() {
    setLoading(true);
    setError("");
    try {
      const data = await generateDataset({ scenario, n: Number(n), persist });
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

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
            <option value="phantom_invoice">Phantom Invoice</option>
            <option value="duplicate_financing">Duplicate Financing</option>
            <option value="carousel_trade">Carousel Trade</option>
            <option value="cascade_amplification">Cascade Amplification</option>
          </select>
          <input type="number" min="1" max="100" value={n} onChange={(e) => setN(e.target.value)} />
          <select value={persist ? "yes" : "no"} onChange={(e) => setPersist(e.target.value === "yes")}>
            <option value="no">Preview only</option>
            <option value="yes">Persist to backend</option>
          </select>
        </div>
        <div style={{ marginTop: 16 }}>
          <button className="button" onClick={handleGenerate} disabled={loading}>
            {loading ? "Generating..." : "Generate Scenario"}
          </button>
        </div>
        {error ? <p style={{ color: "#fca5a5" }}>{error}</p> : null}
      </div>

      {result ? (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Scenario Output</h3>
          <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(result, null, 2)}</pre>
        </div>
      ) : null}
    </div>
  );
}
