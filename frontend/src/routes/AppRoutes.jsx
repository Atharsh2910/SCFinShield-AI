import { Link, Route, Routes } from "react-router-dom";

import CaseDetail from "../pages/CaseDetail";
import Dashboard from "../pages/Dashboard";
import FraudCases from "../pages/FraudCases";
import GraphExplorer from "../pages/GraphExplorer";
import InvestigationChat from "../pages/InvestigationChat";
import InvoiceUpload from "../pages/InvoiceUpload";
import Simulator from "../pages/Simulator";
import GuardedRoute from "./GuardedRoute";

function NotFound() {
  return (
    <div className="card">
      <h2>Page not found</h2>
      <p className="muted">The route you requested does not exist.</p>
      <Link className="button secondary" to="/">
        Back to Dashboard
      </Link>
    </div>
  );
}

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/upload" element={<InvoiceUpload />} />
      <Route path="/cases" element={<FraudCases />} />
      <Route path="/cases/:caseId" element={<CaseDetail />} />
      <Route path="/graph" element={<GraphExplorer />} />
      <Route path="/graph/:entityId" element={<GraphExplorer />} />
      <Route
        path="/investigate/:caseId"
        element={
          <GuardedRoute>
            <InvestigationChat />
          </GuardedRoute>
        }
      />
      <Route path="/simulator" element={<Simulator />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}
