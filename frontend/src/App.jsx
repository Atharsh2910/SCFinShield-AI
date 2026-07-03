import { NavLink } from "react-router-dom";

import { useAuth } from "./context/AuthContext";
import AppRoutes from "./routes/AppRoutes";

function AppLayout() {
  const { user, isAuthenticated, loginDemo, logout } = useAuth();

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>SCFinShield-AI</h1>
        <div className="card mini-card" style={{ marginBottom: 12 }}>
          {isAuthenticated ? (
            <>
              <p className="muted" style={{ marginBottom: 8 }}>
                Signed in as <strong>{user?.name}</strong>
              </p>
              <button className="button secondary" onClick={logout}>
                Logout
              </button>
            </>
          ) : (
            <button className="button" onClick={loginDemo}>
              Login Demo User
            </button>
          )}
        </div>
        <nav>
          <NavLink to="/">Dashboard</NavLink>
          <NavLink to="/upload">Invoice Upload</NavLink>
          <NavLink to="/cases">Fraud Cases</NavLink>
          <NavLink to="/graph">Graph Explorer</NavLink>
          <NavLink to="/simulator">Simulator</NavLink>
        </nav>
      </aside>

      <main className="content">
        <AppRoutes />
      </main>
    </div>
  );
}

export default AppLayout;
