import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function ShapChart({ features = [] }) {
  if (!features.length) {
    return (
      <p className="muted">No SHAP feature importance data available.</p>
    );
  }

  const data = [...features]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, 8)
    .map((f) => ({
      name: f.feature,
      value: Number(f.shap_value || 0),
    }));

  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 30 }}>
          <XAxis
            type="number"
            stroke="#64748b"
            tickFormatter={(v) => v.toFixed(2)}
            domain={["auto", "auto"]}
          />
          <YAxis
            dataKey="name"
            type="category"
            stroke="#64748b"
            width={150}
            tick={{ fontSize: 12, fill: "#94a3b8" }}
          />
          <Tooltip
            contentStyle={{
              background: "#0a1628",
              border: "1px solid rgba(6,182,212,0.2)",
              borderRadius: 8,
              color: "#e2e8f0",
            }}
            formatter={(value) => [value.toFixed(4), "SHAP"]}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, idx) => (
              <Cell
                key={`shap-${idx}`}
                fill={entry.value >= 0 ? "#ef4444" : "#10b981"}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
