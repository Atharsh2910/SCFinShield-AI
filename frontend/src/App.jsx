import { NavLink, Route, Routes } from "react-router-dom";

import Dashboard from "./pages/Dashboard";
import FraudCases from "./pages/FraudCases";
import GraphExplorer from "./pages/GraphExplorer";
import InvestigationChat from "./pages/InvestigationChat";
import InvoiceUpload from "./pages/InvoiceUpload";
import Simulator from "./pages/Simulator";

function AppLayout() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>SCFinShield-AI</h1>
        <nav>
          <NavLink to="/">Dashboard</NavLink>
          <NavLink to="/upload">Invoice Upload</NavLink>
          <NavLink to="/cases">Fraud Cases</NavLink>
          <NavLink to="/graph">Graph Explorer</NavLink>
          <NavLink to="/simulator">Simulator</NavLink>
        </nav>
      </aside>

      <main className="content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<InvoiceUpload />} />
          <Route path="/cases" element={<FraudCases />} />
          <Route path="/graph" element={<GraphExplorer />} />
          <Route path="/investigate/:caseId" element={<InvestigationChat />} />
          <Route path="/simulator" element={<Simulator />} />
        </Routes>
      </main>
    </div>
  );
}

export default AppLayout;
