export default function FraudScoreBadge({ decision = "PASS", score = 0 }) {
  const normalized = String(decision || "PASS").toUpperCase();
  const badgeClass =
    normalized === "HOLD"
      ? "badge hold"
      : normalized === "REVIEW"
      ? "badge review"
      : "badge pass";
  const emoji =
    normalized === "HOLD" ? "🔴" : normalized === "REVIEW" ? "🟡" : "🟢";

  return (
    <span className={badgeClass}>
      {emoji} {normalized} &nbsp;·&nbsp; {(Number(score) * 100).toFixed(1)}%
    </span>
  );
}
