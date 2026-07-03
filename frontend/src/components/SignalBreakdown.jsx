import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";

export default function SignalBreakdown({ scores = {} }) {
  const data = Object.entries(scores).map(([name, value]) => ({
    name,
    score: Number(value || 0),
  }));

  if (!data.length) {
    return <p className="muted">No model signal data available.</p>;
  }

  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
          <XAxis type="number" domain={[0, 1]} stroke="#94a3b8" />
          <YAxis dataKey="name" type="category" stroke="#94a3b8" width={140} />
          <Tooltip />
          <Bar dataKey="score" fill="#38bdf8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
