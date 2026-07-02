import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import FraudScoreBadge from "../components/FraudScoreBadge";
import { generateSAR, listCases, updateDecision } from "../services/fraudService";

export default function FraudCases() {
  const queryClient = useQueryClient();
  const { data: cases = [], isLoading, error } = useQuery({
    queryKey: ["fraud-cases"],
    queryFn: listCases,
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, analyst_decision }) => updateDecision(id, { analyst_decision }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["fraud-cases"] }),
  });

  const sarMutation = useMutation({
    mutationFn: generateSAR,
  });

  if (isLoading) return <div className="card">Loading fraud cases...</div>;
  if (error) return <div className="card">Failed to load fraud cases: {error.message}</div>;

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Fraud Cases</h2>
          <p className="muted">Review decisions, evidence, and SAR drafts.</p>
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
              <tr key={item.id}>
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
                  </div>
                  {sarMutation.data?.case_id === item.id ? (
                    <p className="muted" style={{ marginTop: 8 }}>
                      SAR drafted.
                    </p>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
