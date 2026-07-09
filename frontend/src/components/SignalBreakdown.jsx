import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const SIGNAL_COLORS = {
  dnn: "#8b5cf6",
  isolation_forest: "#06b6d4",
  siamese: "#3b82f6",
  graph_carousel: "#ef4444",
  graph_cascade: "#f59e0b",
};

export default function SignalBreakdown({ scores = {} }) {
  const data = Object.entries(scores).map(([name, value]) => ({
    name: name.replace(/_/g, " "),
    score: Number(value || 0),
    key: name,
  }));

  if (!data.length) {
    return <p className="muted">No model signal data available.</p>;
  }

  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 30 }}>
          <XAxis
            type="number"
            domain={[0, 1]}
            stroke="#64748b"
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <YAxis
            dataKey="name"
            type="category"
            stroke="#64748b"
            width={130}
            tick={{ fontSize: 12, fill: "#94a3b8" }}
          />
          <Tooltip
            contentStyle={{
              background: "#0a1628",
              border: "1px solid rgba(6,182,212,0.2)",
              borderRadius: 8,
              color: "#e2e8f0",
            }}
            formatter={(value) => [
              `${(value * 100).toFixed(1)}%`,
              "Risk Score",
            ]}
          />
          <Bar dataKey="score" radius={[0, 4, 4, 0]}>
            {data.map((entry, idx) => (
              <Cell
                key={`signal-${idx}`}
                fill={SIGNAL_COLORS[entry.key] || "#8b5cf6"}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
