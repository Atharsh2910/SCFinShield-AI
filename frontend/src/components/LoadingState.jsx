export default function LoadingState({ message = "Loading..." }) {
  return (
    <div className="card state-card">
      <p className="muted">{message}</p>
    </div>
  );
}
