from __future__ import annotations

from langgraph.graph import StateGraph, END

from backend.services.langgraph.agents import (
    dedup_agent,
    finalize_agent,
    graph_agent,
    match_agent,
    ml_agent,
    rag_agent,
)
from backend.services.langgraph.state import FraudState


def build_fraud_investigation_graph():
    """
    Build the LangGraph StateGraph for fraud investigation.

    Flow: dedup → match → graph → ml → rag → END
    Run sequentially to keep resource usage low on free tiers.
    """
    graph = StateGraph(FraudState)

    graph.add_node("dedup", dedup_agent)
    graph.add_node("match", match_agent)
    graph.add_node("graph_analysis", graph_agent)
    graph.add_node("ml_inference", ml_agent)
    graph.add_node("rag_narrative", rag_agent)
    graph.add_node("finalize", finalize_agent)

    graph.set_entry_point("dedup")
    graph.add_edge("dedup", "match")
    graph.add_edge("match", "graph_analysis")
    graph.add_edge("graph_analysis", "ml_inference")

    def route_after_ml(state: FraudState) -> str:
        return "rag_narrative" if state.get("fraud_decision") != "PASS" else "finalize"

    graph.add_conditional_edges(
        "ml_inference",
        route_after_ml,
        {
            "rag_narrative": "rag_narrative",
            "finalize": "finalize",
        },
    )
    graph.add_edge("rag_narrative", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


fraud_graph = build_fraud_investigation_graph()


async def run_fraud_investigation(invoice: dict, invoice_id: str) -> FraudState:
    """Run the full fraud investigation workflow for a single invoice."""
    initial_state: FraudState = {
        "invoice": invoice,
        "invoice_id": invoice_id,
        "dedup_result": {},
        "match_result": {},
        "graph_findings": {},
        "ml_result": {},
        "rag_result": {},
        "ensemble_score": 0.0,
        "fraud_decision": "PASS",
        "fraud_patterns": [],
        "alert_narrative": "",
        "regulation_citations": [],
        "shap_values": {},
        "cascade_exposure": 0.0,
        "processing_errors": [],
        "audit_trail": [],
        "retrieved_documents": [],
    }
    result = await fraud_graph.ainvoke(initial_state)
    return result

