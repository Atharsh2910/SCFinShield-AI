from __future__ import annotations

from typing import Any, Dict

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone

from backend.core.config import get_settings
from backend.core.constants import PineconeNamespace
from backend.db.pinecone import get_pinecone_index

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


def build_rag_chain() -> RetrievalQA:
    settings = get_settings()
    index = get_pinecone_index()

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    vectorstore = LangChainPinecone(
        index=index,
        embedding=embeddings,
        text_key="text",
        namespace=PineconeNamespace.REGULATIONS,
    )

    llm = ChatAnthropic(
        model=settings.claude_model,
        anthropic_api_key=settings.anthropic_api_key,
        max_tokens=1000,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": ALERT_PROMPT},
        return_source_documents=True,
    )


async def generate_fraud_narrative(
    invoice: Dict[str, Any],
    fraud_result: Dict[str, Any],
    graph_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a grounded fraud alert narrative using RAG."""
    chain = build_rag_chain()

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

    result = chain.invoke({"query": signal_summary})
    narrative = result.get("result", "")

    citations: list[dict[str, Any]] = []
    for doc in result.get("source_documents", []) or []:
        metadata = getattr(doc, "metadata", {}) or {}
        citations.append(
            {
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
                "category": metadata.get("category", ""),
            }
        )

    return {
        "narrative": narrative,
        "regulation_citations": citations,
    }

