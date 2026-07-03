import { createContext, useContext, useMemo, useState } from "react";

const AuthContext = createContext(null);

const DEFAULT_USER = {
  id: "demo-analyst",
  name: "Demo Analyst",
  role: "fraud_analyst",
};

export function AuthProvider({ children }) {
  const [user, setUser] = useState(DEFAULT_USER);

  const value = useMemo(
    () => ({
      user,
      isAuthenticated: Boolean(user),
      loginDemo: () => setUser(DEFAULT_USER),
      logout: () => setUser(null),
    }),
    [user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
