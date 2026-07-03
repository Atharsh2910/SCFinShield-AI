export default function ErrorState({ message = "Something went wrong.", onRetry }) {
  return (
    <div className="card state-card">
      <p style={{ color: "#fca5a5", marginBottom: 10 }}>{message}</p>
      {onRetry ? (
        <button className="button secondary" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}
