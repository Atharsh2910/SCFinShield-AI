from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone

from backend.core.config import get_settings
from backend.core.constants import PineconeNamespace
from backend.core.exceptions import RAGRetrievalError
from backend.db.pinecone import PineconeConfigurationError, get_pinecone_index

ALERT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert SCF fraud analyst. Using the regulatory context below,\n"
        "write a concise fraud alert brief. Be specific, cite regulation names,\n"
        "and state the recommended action.\n\n"
        "Regulatory context:\n"
        "{context}\n\n"
        "Fraud signal summary:\n"
        "{question}\n\n"
        "Write a 3-paragraph alert brief:\n"
        "1. What fraud pattern was detected and why it is suspicious\n"
        "2. Which regulation or typology this violates (cite the source)\n"
        "3. Recommended action (HOLD/REVIEW/PASS) with rationale\n\n"
        "Alert brief:"
    ),
)


ANALYST_CONTEXT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a senior supply chain finance fraud analyst.\n"
        "Use the retrieved context to answer the analyst's question accurately.\n"
        "If the answer is uncertain, say so clearly.\n\n"
        "Retrieved context:\n"
        "{context}\n\n"
        "Analyst question:\n"
        "{question}\n\n"
        "Answer in concise professional language and cite the most relevant sources."
    ),
)


@lru_cache()
def get_langchain_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


@lru_cache()
def get_regulations_vectorstore() -> LangChainPinecone:
    index = get_pinecone_index()
    return LangChainPinecone(
        index=index,
        embedding=get_langchain_embeddings(),
        text_key="text",
        namespace=PineconeNamespace.REGULATIONS,
    )


@lru_cache()
def get_fraud_cases_vectorstore() -> LangChainPinecone:
    index = get_pinecone_index()
    return LangChainPinecone(
        index=index,
        embedding=get_langchain_embeddings(),
        text_key="text",
        namespace=PineconeNamespace.FRAUD_CASES,
    )


@lru_cache()
def get_llm() -> ChatAnthropic:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RAGRetrievalError("Anthropic credentials are missing. Set ANTHROPIC_API_KEY.")
    return ChatAnthropic(
        model=settings.claude_model,
        anthropic_api_key=settings.anthropic_api_key,
        max_tokens=1000,
    )


def build_rag_chain() -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=get_regulations_vectorstore().as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": ALERT_PROMPT},
        return_source_documents=True,
    )


def build_analyst_rag_chain() -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=get_fraud_cases_vectorstore().as_retriever(search_kwargs={"k": 4}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": ANALYST_CONTEXT_PROMPT},
        return_source_documents=True,
    )


async def generate_fraud_narrative(
    invoice: Dict[str, Any],
    fraud_result: Dict[str, Any],
    graph_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a grounded fraud alert narrative using RAG."""
    top_signals = ", ".join(
        f.get("feature", "") for f in (fraud_result.get("top_shap_features", []) or [])[:3]
    )
    siamese_score = float(fraud_result.get("individual_scores", {}).get("siamese", 0) or 0)
    cascade_exposure = float(graph_context.get("total_cascade_exposure", 0) or 0)

    signal_summary = (
        f"Invoice: {invoice.get('invoice_number')} | "
        f"Amount: {invoice.get('currency', 'INR')} {float(invoice.get('amount', 0) or 0):,.2f}\n"
        f"Supplier: {invoice.get('supplier_name')} -> Buyer: {invoice.get('buyer_name')}\n"
        f"Fraud Score: {float(fraud_result.get('ensemble_score', 0) or 0):.2%} | "
        f"Decision: {fraud_result.get('fraud_decision')}\n"
        f"Top Signals: {top_signals}\n"
        f"Duplicate Risk: {siamese_score:.2%}\n"
        f"Graph: Carousel={'YES' if graph_context.get('has_carousel') else 'NO'} | "
        f"Cascade Depth={graph_context.get('cascade_depth', 0)}\n"
        f"Cascade Exposure: INR {cascade_exposure:,.2f}\n"
        f"Match Score: {float(fraud_result.get('match_score', 0) or 0):.2%}"
    )

    try:
        chain = build_rag_chain()
        result = chain.invoke({"query": signal_summary})
        narrative = result.get("result", "")
        citations = _extract_citations(result.get("source_documents", []) or [])
        if not narrative.strip():
            raise RAGRetrievalError("No narrative returned from retrieval chain")
        return {
            "narrative": narrative,
            "regulation_citations": citations,
        }
    except (RAGRetrievalError, PineconeConfigurationError, Exception):
        # Graceful fallback until env keys/indexes are configured.
        return {
            "narrative": _build_fallback_narrative(invoice, fraud_result, graph_context, top_signals),
            "regulation_citations": [],
        }


def _extract_citations(source_documents: list[Any]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for doc in source_documents:
        metadata = getattr(doc, "metadata", {}) or {}
        citations.append(
            {
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
                "category": metadata.get("category", ""),
            }
        )
    return citations


def _build_fallback_narrative(
    invoice: Dict[str, Any],
    fraud_result: Dict[str, Any],
    graph_context: Dict[str, Any],
    top_signals: str,
) -> str:
    decision = str(fraud_result.get("fraud_decision") or "REVIEW")
    return (
        f"Invoice `{invoice.get('invoice_number')}` for {invoice.get('currency', 'INR')} "
        f"{float(invoice.get('amount', 0) or 0):,.2f} was flagged for {decision}. "
        f"The strongest signals were: {top_signals or 'duplicate and graph anomalies'}.\n\n"
        f"Duplicate similarity reached {float(fraud_result.get('individual_scores', {}).get('siamese', 0) or 0):.2%}, "
        f"and the graph analysis found carousel={graph_context.get('has_carousel', False)} "
        f"with cascade depth={graph_context.get('cascade_depth', 0)}.\n\n"
        "RAG-backed regulatory citations are not available yet because the vector store and/or LLM credentials "
        "have not been configured. Once Pinecone and Anthropic credentials are added, this narrative will include "
        "grounded citations from indexed regulations."
    )

