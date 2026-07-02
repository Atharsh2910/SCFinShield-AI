export default function FraudScoreBadge({ decision = "PASS", score = 0 }) {
  const normalized = String(decision || "PASS").toLowerCase();
  const badgeClass =
    normalized === "hold" ? "badge hold" : normalized === "review" ? "badge review" : "badge pass";

  return <span className={badgeClass}>{decision} · {(Number(score) * 100).toFixed(1)}%</span>;
}
