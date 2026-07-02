import { useQuery } from "@tanstack/react-query";
import { BarChart3 } from "lucide-react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";

import { api } from "../services/api";

async function getDashboardData() {
  const [summary, timeline, alerts] = await Promise.all([
    api.get("/dashboard/summary"),
    api.get("/dashboard/timeline"),
    api.get("/dashboard/recent-alerts"),
  ]);
  return {
    summary: summary.data,
    timeline: timeline.data.timeline || [],
    alerts: alerts.data.alerts || [],
  };
}

export default function Dashboard() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["dashboard"],
    queryFn: getDashboardData,
  });

  if (isLoading) {
    return <div className="card">Loading dashboard...</div>;
  }

  if (error) {
    return <div className="card">Failed to load dashboard: {error.message}</div>;
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

      <div className="grid grid-4">
        <div className="card">
          <div className="muted">Total Invoices</div>
          <div className="kpi-value">{summary.total_invoices ?? 0}</div>
        </div>
        <div className="card">
          <div className="muted">Flagged Invoices</div>
          <div className="kpi-value">{summary.flagged_count ?? 0}</div>
        </div>
        <div className="card">
          <div className="muted">Total Exposure</div>
          <div className="kpi-value">INR {(summary.total_exposure ?? 0).toLocaleString()}</div>
        </div>
        <div className="card">
          <div className="muted">Fraud Rate</div>
          <div className="kpi-value">{((summary.fraud_rate ?? 0) * 100).toFixed(1)}%</div>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h3>Fraud Timeline</h3>
          <div className="chart-box">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data?.timeline || []}>
                <XAxis dataKey="month" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip />
                <Bar dataKey="flagged" fill="#f59e0b" />
              </BarChart>
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
                </tr>
              </thead>
              <tbody>
                {(data?.alerts || []).map((alert) => (
                  <tr key={alert.id}>
                    <td>{alert.case_number}</td>
                    <td>{alert.decision}</td>
                    <td>{alert.severity}</td>
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
