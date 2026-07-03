import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";

import { getCase } from "../services/fraudService";
import { askQuestion, endInvestigation, getHistory, startInvestigation } from "../services/investigationService";
import { useFraudStore } from "../context/FraudContext";

export default function InvestigationChat() {
  const { caseId } = useParams();
  const { investigationSessionId, setInvestigationSessionId } = useFraudStore();
  const [question, setQuestion] = useState("");
  const quickQuestions = [
    "Why was this flagged?",
    "What is the maximum cascade exposure?",
    "Draft SAR summary for this case.",
  ];

  const { data: fraudCase } = useQuery({
    queryKey: ["case", caseId],
    queryFn: () => getCase(caseId),
    enabled: Boolean(caseId),
  });

  const startMutation = useMutation({
    mutationFn: () => startInvestigation(caseId),
    onSuccess: (data) => setInvestigationSessionId(data.session_id),
  });

  const historyQuery = useQuery({
    queryKey: ["investigation-history", investigationSessionId],
    queryFn: () => getHistory(investigationSessionId),
    enabled: Boolean(investigationSessionId),
  });

  const askMutation = useMutation({
    mutationFn: () => askQuestion(investigationSessionId, question),
    onSuccess: async () => {
      setQuestion("");
      await historyQuery.refetch();
    },
  });

  const endMutation = useMutation({
    mutationFn: () => endInvestigation(investigationSessionId),
    onSuccess: () => setInvestigationSessionId(null),
  });

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Investigation Chat</h2>
          <p className="muted">Ask follow-up questions about a flagged fraud case.</p>
        </div>
      </div>

      <div className="split">
        <div className="card">
          <h3>Case Summary</h3>
          {fraudCase ? (
            <>
              <p><strong>{fraudCase.case_number}</strong></p>
              <p className="muted">Decision: {fraudCase.decision}</p>
              <p className="muted">Severity: {fraudCase.severity}</p>
              <p>{fraudCase.alert_narrative}</p>
            </>
          ) : (
            <p className="muted">Loading case...</p>
          )}
          <button className="button" onClick={() => startMutation.mutate()} disabled={!caseId}>
            {investigationSessionId ? "Session Ready" : "Start Investigation"}
          </button>
          <button
            className="button secondary"
            style={{ marginLeft: 8 }}
            onClick={() => endMutation.mutate()}
            disabled={!investigationSessionId}
          >
            End Session
          </button>
        </div>

        <div className="card">
          <h3>Analyst Q&A</h3>
          <div className="chat-box">
            {(historyQuery.data?.messages || []).map((message, index) => (
              <div key={index} className="chat-message">
                <strong>{message.role}</strong>
                <p>{message.content}</p>
                {(message.citations || []).length ? (
                  <p className="muted">
                    Citations: {(message.citations || []).map((c) => c.title || c.source).join(", ")}
                  </p>
                ) : null}
              </div>
            ))}
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 }}>
            {quickQuestions.map((item) => (
              <button
                key={item}
                className="button secondary"
                onClick={() => setQuestion(item)}
                disabled={!investigationSessionId}
              >
                {item}
              </button>
            ))}
          </div>

          <div className="form-row" style={{ marginTop: 16 }}>
            <textarea
              rows={4}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask: Why was this flagged? What is the max exposure? Draft SAR?"
            />
            <button
              className="button"
              onClick={() => askMutation.mutate()}
              disabled={!investigationSessionId || !question.trim()}
            >
              Ask Question
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
