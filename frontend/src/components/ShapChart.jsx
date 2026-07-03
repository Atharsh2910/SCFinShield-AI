import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from "recharts";

export default function ShapChart({ features = [] }) {
  const data = (features || []).map((f) => ({
    feature: f.feature,
    shap: Number(f.shap_value || 0),
  }));

  if (!data.length) {
    return <p className="muted">No SHAP explainability data available.</p>;
  }

  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
          <XAxis type="number" stroke="#94a3b8" />
          <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={160} />
          <Tooltip />
          <Bar dataKey="shap">
            {data.map((entry, index) => (
              <Cell key={`${entry.feature}-${index}`} fill={entry.shap >= 0 ? "#ef4444" : "#22c55e"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
