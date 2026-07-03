import { Fragment } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import CascadeTimeline from "../components/CascadeTimeline";
import ErrorState from "../components/ErrorState";
import FraudScoreBadge from "../components/FraudScoreBadge";
import LoadingState from "../components/LoadingState";
import SignalBreakdown from "../components/SignalBreakdown";
import ShapChart from "../components/ShapChart";
import { useFraudStore } from "../context/FraudContext";
import { generateSAR, listCases, updateDecision } from "../services/fraudService";
import { useState } from "react";

export default function FraudCases() {
  const queryClient = useQueryClient();
  const [expandedCaseId, setExpandedCaseId] = useState(null);
  const { dashboardFilters, setDashboardFilters } = useFraudStore();
  const [filters, setFilters] = useState(
    dashboardFilters.cases || {
      decision: "",
      severity: "",
      lender_id: "",
      date_from: "",
      date_to: "",
    },
  );
  const setAndPersistFilters = (updater) => {
    setFilters((prev) => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      setDashboardFilters({ ...dashboardFilters, cases: next });
      return next;
    });
  };

  const initialFilters = {
    decision: "",
    severity: "",
    lender_id: "",
    date_from: "",
    date_to: "",
  };

  const { data: cases = [], isLoading, error } = useQuery({
    queryKey: ["fraud-cases", filters],
    queryFn: () =>
      listCases(
        Object.fromEntries(Object.entries(filters).filter(([, value]) => String(value).trim().length > 0)),
      ),
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, analyst_decision }) => updateDecision(id, { analyst_decision }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["fraud-cases"] }),
  });

  const sarMutation = useMutation({
    mutationFn: generateSAR,
  });

  if (isLoading) return <LoadingState message="Loading fraud cases..." />;
  if (error) return <ErrorState message={`Failed to load fraud cases: ${error.message}`} />;

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Fraud Cases</h2>
          <p className="muted">Review decisions, evidence, and SAR drafts.</p>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="grid grid-3">
          <select
            value={filters.decision}
            onChange={(e) => setAndPersistFilters((prev) => ({ ...prev, decision: e.target.value }))}
          >
            <option value="">All Decisions</option>
            <option value="HOLD">HOLD</option>
            <option value="REVIEW">REVIEW</option>
            <option value="PASS">PASS</option>
          </select>
          <select
            value={filters.severity}
            onChange={(e) => setAndPersistFilters((prev) => ({ ...prev, severity: e.target.value }))}
          >
            <option value="">All Severities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
          <input
            value={filters.lender_id}
            onChange={(e) => setAndPersistFilters((prev) => ({ ...prev, lender_id: e.target.value }))}
            placeholder="Filter by lender UUID"
          />
        </div>
        <div className="grid grid-2" style={{ marginTop: 12 }}>
          <input
            type="date"
            value={filters.date_from}
            onChange={(e) => setAndPersistFilters((prev) => ({ ...prev, date_from: e.target.value }))}
          />
          <input
            type="date"
            value={filters.date_to}
            onChange={(e) => setAndPersistFilters((prev) => ({ ...prev, date_to: e.target.value }))}
          />
        </div>
        <div style={{ marginTop: 10 }}>
          <button className="button secondary" onClick={() => setAndPersistFilters(initialFilters)}>
            Reset Case Filters
          </button>
        </div>
      </div>

      <div className="card table-wrap">
        <table>
          <thead>
            <tr>
              <th>Case</th>
              <th>Score</th>
              <th>Decision</th>
              <th>Severity</th>
              <th>Patterns</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {cases.map((item) => (
              <Fragment key={item.id}>
                <tr>
                  <td>{item.case_number}</td>
                  <td>
                    <FraudScoreBadge decision={item.decision} score={item.fraud_score} />
                  </td>
                  <td>{item.decision}</td>
                  <td>{item.severity}</td>
                  <td>{(item.fraud_patterns || []).join(", ")}</td>
                  <td>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      <button
                        className="button secondary"
                        onClick={() =>
                          updateMutation.mutate({ id: item.id, analyst_decision: "confirmed_fraud" })
                        }
                      >
                        Confirm Fraud
                      </button>
                      <button
                        className="button secondary"
                        onClick={() =>
                          updateMutation.mutate({ id: item.id, analyst_decision: "false_positive" })
                        }
                      >
                        False Positive
                      </button>
                      <button className="button" onClick={() => sarMutation.mutate(item.id)}>
                        Draft SAR
                      </button>
                      <Link className="button secondary" to={`/cases/${item.id}`}>
                        View Details
                      </Link>
                      <button
                        className="button secondary"
                        onClick={() =>
                          setExpandedCaseId((prev) => (prev === item.id ? null : item.id))
                        }
                      >
                        {expandedCaseId === item.id ? "Collapse" : "Expand"}
                      </button>
                    </div>
                    {sarMutation.data?.case_id === item.id ? (
                      <p className="muted" style={{ marginTop: 8 }}>
                        SAR drafted.
                      </p>
                    ) : null}
                  </td>
                </tr>
                {expandedCaseId === item.id ? (
                  <tr>
                    <td colSpan={6}>
                      <div className="grid grid-2">
                        <div>
                          <strong>Model Signals</strong>
                          <SignalBreakdown scores={item.ensemble_scores || {}} />
                        </div>
                        <div>
                          <strong>Cascade Path</strong>
                          <CascadeTimeline cascadePath={item.cascade_path || []} />
                        </div>
                      </div>
                      <div style={{ marginTop: 12 }}>
                        <strong>SHAP Features</strong>
                        <ShapChart
                          features={Object.entries(item.shap_values || {}).map(([feature, shapValue]) => ({
                            feature,
                            shap_value: shapValue,
                          }))}
                        />
                      </div>
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
