import { useMutation, useQuery } from "@tanstack/react-query";
import { useParams, Link } from "react-router-dom";

import AlertNarrative from "../components/AlertNarrative";
import CascadeTimeline from "../components/CascadeTimeline";
import FraudScoreBadge from "../components/FraudScoreBadge";
import ShapChart from "../components/ShapChart";
import SignalBreakdown from "../components/SignalBreakdown";
import { generateSAR, getCase } from "../services/fraudService";

export default function CaseDetail() {
  const { caseId } = useParams();
  const { data: fraudCase, isLoading, error } = useQuery({
    queryKey: ["fraud-case", caseId],
    queryFn: () => getCase(caseId),
    enabled: Boolean(caseId),
  });

  const sarMutation = useMutation({
    mutationFn: generateSAR,
  });

  if (isLoading) return <div className="card">Loading case details...</div>;
  if (error) return <div className="card">Failed to load case: {error.message}</div>;
  if (!fraudCase) return <div className="card">Case not found.</div>;

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Case Detail</h2>
          <p className="muted">
            {fraudCase.case_number} · <FraudScoreBadge decision={fraudCase.decision} score={fraudCase.fraud_score} />
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <Link className="button secondary" to={`/investigate/${fraudCase.id}`}>
            Investigation Chat
          </Link>
          <button className="button" onClick={() => sarMutation.mutate(fraudCase.id)}>
            Generate SAR
          </button>
        </div>
      </div>

      <div className="grid grid-2">
        <div className="card">
          <h3>Signal Breakdown</h3>
          <SignalBreakdown scores={fraudCase.ensemble_scores || {}} />
        </div>
        <div className="card">
          <h3>SHAP Explainability</h3>
          <ShapChart features={Object.entries(fraudCase.shap_values || {}).map(([feature, shapValue]) => ({ feature, shap_value: shapValue }))} />
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h3>Cascade Timeline</h3>
          <CascadeTimeline cascadePath={fraudCase.cascade_path || []} />
        </div>
        <div className="card">
          <h3>Alert Narrative</h3>
          <AlertNarrative narrative={fraudCase.alert_narrative || ""} citations={fraudCase.regulation_citations || []} />
        </div>
      </div>
    </div>
  );
}
