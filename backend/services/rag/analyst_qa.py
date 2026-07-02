from __future__ import annotations

from typing import Any, Dict

from langchain.memory import ConversationBufferWindowMemory

from backend.core.config import get_settings
from backend.services.rag.retrieval_chain import build_analyst_rag_chain, get_llm

_sessions: Dict[str, Dict[str, Any]] = {}


def get_or_create_session(investigation_id: str) -> Dict[str, Any]:
    if investigation_id not in _sessions:
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10,
        )
        settings = get_settings()
        _sessions[investigation_id] = {
            "memory": memory,
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
    Answer analyst questions about a fraud case using conversation memory.
    Augments the LLM context with the current fraud investigation state.
    """
    session = get_or_create_session(investigation_id)
    session["fraud_state"] = fraud_state

    context = (
        "Current fraud investigation state:\n"
        f"- Invoice: {fraud_state.get('invoice', {}).get('invoice_number', 'N/A')}\n"
        f"- Fraud Score: {float(fraud_state.get('ensemble_score', 0) or 0):.2%}\n"
        f"- Decision: {fraud_state.get('fraud_decision', 'N/A')}\n"
        f"- Cascade Depth: {fraud_state.get('cascade_depth', 0)}\n"
        f"- Cascade Exposure: INR {float(fraud_state.get('cascade_exposure', 0) or 0):,.2f}\n"
        f"- Alert Narrative: {fraud_state.get('narrative', 'Not generated yet')}\n"
    )

    augmented_question = f"{context}\n\nAnalyst question: {question}"

    memory: ConversationBufferWindowMemory = session["memory"]
    history = memory.load_memory_variables({}).get("chat_history", [])
    history_context = "\n".join(
        getattr(message, "content", str(message))
        for message in history[-6:]
    )

    prompt = augmented_question if not history_context else f"{history_context}\n\n{augmented_question}"

    answer = ""
    citations: list[dict[str, Any]] = []
    try:
        rag_chain = build_analyst_rag_chain()
        result = rag_chain.invoke({"query": prompt})
        answer = str(result.get("result") or "").strip()
        for doc in result.get("source_documents", []) or []:
            metadata = getattr(doc, "metadata", {}) or {}
            citations.append(
                {
                    "title": metadata.get("title", ""),
                    "source": metadata.get("source", ""),
                    "category": metadata.get("category", ""),
                }
            )
    except Exception:
        settings = get_settings()
        if settings.anthropic_api_key:
            response = await get_llm().ainvoke(prompt)
            answer = str(response.content).strip()
        else:
            answer = (
                "Analyst Q&A is in fallback mode. The current investigation state has been captured, "
                "but retrieval-backed answers will become available after Pinecone and Anthropic credentials are configured."
            )

    memory.save_context(
        {"input": question},
        {"output": answer},
    )

    session["messages"].append({"role": "user", "content": question, "citations": []})
    session["messages"].append({"role": "assistant", "content": answer, "citations": citations})

    return {
        "answer": answer,
        "investigation_id": investigation_id,
        "message_count": len(session["messages"]),
        "citations": citations,
    }


def get_session_messages(investigation_id: str) -> list[dict[str, Any]]:
    session = _sessions.get(investigation_id)
    if not session:
        return []
    return list(session.get("messages", []))


def delete_session(investigation_id: str) -> bool:
    return _sessions.pop(investigation_id, None) is not None

