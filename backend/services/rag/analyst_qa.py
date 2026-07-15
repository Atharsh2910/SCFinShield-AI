from __future__ import annotations

from collections import deque
from typing import Any, Dict

from backend.core.config import get_settings
from backend.core.constants import PineconeNamespace
from backend.services.rag.retrieval_chain import _retrieve_context, build_analyst_rag_chain, get_llm

_sessions: Dict[str, Dict[str, Any]] = {}

_MAX_HISTORY = 10  # Keep last N message pairs in memory


def get_or_create_session(investigation_id: str) -> Dict[str, Any]:
    if investigation_id not in _sessions:
        settings = get_settings()
        _sessions[investigation_id] = {
            "history": deque(maxlen=_MAX_HISTORY * 2),  # alternating user/assistant
            "llm_enabled": bool(settings.anthropic_api_key),
            "fraud_state": {},
            "messages": [],
        }
    return _sessions[investigation_id]


async def ask_analyst_question(
    investigation_id: str,
    question: str,
    fraud_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Answer analyst questions about a fraud case using conversation history.
    Augments the LLM context with the current fraud investigation state.
    """
    session = get_or_create_session(investigation_id)
    session["fraud_state"] = fraud_state

    context_header = (
        "Current fraud investigation state:\n"
        f"- Invoice: {fraud_state.get('invoice', {}).get('invoice_number', 'N/A')}\n"
        f"- Fraud Score: {float(fraud_state.get('ensemble_score', 0) or 0):.2%}\n"
        f"- Decision: {fraud_state.get('fraud_decision', 'N/A')}\n"
        f"- Cascade Depth: {fraud_state.get('cascade_depth', 0)}\n"
        f"- Cascade Exposure: INR {float(fraud_state.get('cascade_exposure', 0) or 0):,.2f}\n"
        f"- Alert Narrative: {fraud_state.get('narrative', 'Not generated yet')}\n"
    )

    history_lines = list(session["history"])
    history_context = "\n".join(history_lines[-6:]) if history_lines else ""

    augmented_question = (
        f"{history_context}\n\n{context_header}\n\nAnalyst question: {question}"
        if history_context
        else f"{context_header}\n\nAnalyst question: {question}"
    )

    answer = ""
    citations: list[dict[str, Any]] = []

    try:
        rag_context, citations = _retrieve_context(augmented_question, PineconeNamespace.FRAUD_CASES)
        chain = build_analyst_rag_chain()
        answer = await chain.ainvoke({"context": rag_context, "question": augmented_question})
        answer = str(answer).strip()
    except Exception:
        settings = get_settings()
        if settings.anthropic_api_key:
            try:
                response = await get_llm().ainvoke(augmented_question)
                answer = str(response.content).strip()
            except Exception:
                answer = _fallback_answer()
        else:
            answer = _fallback_answer()

    # Update rolling history
    session["history"].append(f"User: {question}")
    session["history"].append(f"Assistant: {answer}")

    session["messages"].append({"role": "user", "content": question, "citations": []})
    session["messages"].append({"role": "assistant", "content": answer, "citations": citations})

    return {
        "answer": answer,
        "investigation_id": investigation_id,
        "message_count": len(session["messages"]),
        "citations": citations,
    }


def _fallback_answer() -> str:
    return (
        "Analyst Q&A is in fallback mode. The current investigation state has been captured, "
        "but retrieval-backed answers will become available after Pinecone and Anthropic credentials are configured."
    )


def get_session_messages(investigation_id: str) -> list[dict[str, Any]]:
    session = _sessions.get(investigation_id)
    if not session:
        return []
    return list(session.get("messages", []))


def delete_session(investigation_id: str) -> bool:
    return _sessions.pop(investigation_id, None) is not None
