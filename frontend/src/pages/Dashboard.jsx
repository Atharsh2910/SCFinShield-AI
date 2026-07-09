import { useQuery } from "@tanstack/react-query";
import { BarChart3 } from "lucide-react";
import { Link } from "react-router-dom";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import { useFraudStore } from "../context/FraudContext";
import { api } from "../services/api";

async function getDashboardData() {
  const [summary, timeline, alerts, heatmap, topRisks] = await Promise.all([
    api.get("/dashboard/summary"),
    api.get("/dashboard/timeline"),
    api.get("/dashboard/recent-alerts"),
    api.get("/dashboard/risk-heatmap"),
    api.get("/dashboard/top-risks"),
  ]);
  return {
    summary: summary.data,
    timeline: timeline.data.timeline || [],
    alerts: alerts.data.alerts || [],
    heatmap: heatmap.data || {},
    topRisks: topRisks.data.top_risks || [],
  };
}

export default function Dashboard() {
  const { dashboardFilters, setDashboardFilters } = useFraudStore();
  const { data, isLoading, error } = useQuery({
    queryKey: ["dashboard", dashboardFilters],
    queryFn: getDashboardData,
  });

  if (isLoading) {
    return <LoadingState message="Loading dashboard..." />;
  }

  if (error) {
    return <ErrorState message={`Failed to load dashboard: ${error.message}`} />;
  }

  const summary = data?.summary || {};

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Dashboard</h2>
          <p className="muted">SCF fraud monitoring overview and recent alerts.</p>
        </div>
        <BarChart3 />
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="grid grid-3">
          <select
            value={dashboardFilters.decision || ""}
            onChange={(e) => setDashboardFilters({ ...dashboardFilters, decision: e.target.value })}
          >
            <option value="">Default summary</option>
            <option value="HOLD">Focus HOLD</option>
            <option value="REVIEW">Focus REVIEW</option>
            <option value="PASS">Focus PASS</option>
          </select>
          <button className="button secondary" onClick={() => setDashboardFilters({})}>
            Reset Dashboard Filters
          </button>
        </div>
      </div>

      <div className="grid grid-4">
        <div className="card">
          <div className="muted">📄 Total Invoices</div>
          <div className="kpi-value">{summary.total_invoices ?? 0}</div>
        </div>
        <div className="card">
          <div className="muted">🚨 Flagged Invoices</div>
          <div className="kpi-value">{summary.flagged_count ?? 0}</div>
        </div>
        <div className="card">
          <div className="muted">💰 Total Exposure</div>
          <div className="kpi-value">INR {(summary.total_exposure ?? 0).toLocaleString()}</div>
        </div>
        <div className="card">
          <div className="muted">📊 Fraud Rate</div>
          <div className="kpi-value">{((summary.fraud_rate ?? 0) * 100).toFixed(1)}%</div>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h3>Fraud Timeline</h3>
          <div className="chart-box">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data?.timeline || []} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="fraudGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,137,185,0.1)" />
                <XAxis dataKey="month" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{
                    background: "#0a1628",
                    border: "1px solid rgba(6,182,212,0.2)",
                    borderRadius: 8,
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="flagged"
                  stroke="#06b6d4"
                  strokeWidth={2}
                  fill="url(#fraudGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <h3>Recent Alerts</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Decision</th>
                  <th>Severity</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {(data?.alerts || []).map((alert) => (
                  <tr key={alert.id}>
                    <td>{alert.case_number}</td>
                    <td>{alert.decision}</td>
                    <td>{alert.severity}</td>
                    <td>
                      <Link className="button secondary" to={`/cases/${alert.id}`}>
                        Review
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h3>Risk Heatmap (Tier x Sector)</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Tier</th>
                  {(data?.heatmap?.sectors || []).map((sector) => (
                    <th key={sector}>{sector}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(data?.heatmap?.tiers || []).map((tier, idx) => (
                  <tr key={tier}>
                    <td>Tier {tier}</td>
                    {(data?.heatmap?.matrix?.[idx] || []).map((value, valIdx) => (
                      <td key={`${tier}-${valIdx}`}>INR {Number(value || 0).toLocaleString()}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <h3>Top Risks</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Invoice</th>
                  <th>Decision</th>
                  <th>Score</th>
                  <th>Amount</th>
                </tr>
              </thead>
              <tbody>
                {(data?.topRisks || []).map((risk) => (
                  <tr key={risk.id}>
                    <td>{risk.invoice_number}</td>
                    <td>{risk.fraud_decision}</td>
                    <td>{Number(risk.fraud_score || 0).toFixed(2)}</td>
                    <td>INR {Number(risk.amount || 0).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
