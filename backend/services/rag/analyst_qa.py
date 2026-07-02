from __future__ import annotations

from typing import Any, Dict

from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferWindowMemory

from backend.core.config import get_settings

_sessions: Dict[str, Dict[str, Any]] = {}


def get_or_create_session(investigation_id: str) -> Dict[str, Any]:
    if investigation_id not in _sessions:
        settings = get_settings()
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10,
        )
        llm = ChatAnthropic(
            model=settings.claude_model,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=800,
        )
        _sessions[investigation_id] = {
            "memory": memory,
            "llm": llm,
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

    llm = session["llm"]
    memory: ConversationBufferWindowMemory = session["memory"]

    _ = memory.load_memory_variables({})

    response = await llm.ainvoke(augmented_question)
    answer = response.content

    memory.save_context(
        {"input": question},
        {"output": answer},
    )

    session["messages"].append({"role": "user", "content": question})
    session["messages"].append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "investigation_id": investigation_id,
        "message_count": len(session["messages"]),
    }


def get_session_messages(investigation_id: str) -> list[dict[str, Any]]:
    session = _sessions.get(investigation_id)
    if not session:
        return []
    return list(session.get("messages", []))


def delete_session(investigation_id: str) -> bool:
    return _sessions.pop(investigation_id, None) is not None

