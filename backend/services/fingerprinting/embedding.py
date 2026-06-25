from functools import lru_cache
from typing import Any

from backend.core.config import get_settings
from backend.core.exceptions import RAGRetrievalError


@lru_cache()
def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RAGRetrievalError(
            "Invoice embeddings require sentence-transformers. Install project requirements first."
        ) from exc

    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def invoice_to_text(invoice: dict[str, Any]) -> str:
    parts = [
        f"Invoice {invoice.get('invoice_number', '')}",
        f"From {invoice.get('supplier_name', '')}",
        f"To {invoice.get('buyer_name', '')}",
        f"Amount {invoice.get('amount', 0)} {invoice.get('currency', 'INR')}",
        f"Date {invoice.get('invoice_date', '')}",
        f"PO {invoice.get('po_number', '')}",
        f"GRN {invoice.get('grn_number', '')}",
    ]
    for item in invoice.get("line_items", []) or []:
        if isinstance(item, dict):
            parts.append(str(item.get("description", "")))
        else:
            parts.append(str(item))
    return " ".join(part for part in parts if part)


def generate_embedding(invoice: dict[str, Any]) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode(invoice_to_text(invoice), normalize_embeddings=True)
    return embedding.tolist()


def upsert_invoice_embedding(
    invoice_id: str,
    invoice: dict[str, Any],
    pinecone_index,
    namespace: str = "invoices",
) -> None:
    metadata = {
        "invoice_number": str(invoice.get("invoice_number", "")),
        "supplier_name": str(invoice.get("supplier_name", "")),
        "buyer_name": str(invoice.get("buyer_name", "")),
        "amount": float(invoice.get("amount", 0) or 0),
        "invoice_date": str(invoice.get("invoice_date", "")),
        "lender_name": str(invoice.get("lender_name", "")),
    }
    pinecone_index.upsert(
        vectors=[{"id": invoice_id, "values": generate_embedding(invoice), "metadata": metadata}],
        namespace=namespace,
    )


def find_similar_invoices(
    invoice: dict[str, Any],
    pinecone_index,
    top_k: int = 10,
    namespace: str = "invoices",
) -> list[dict[str, Any]]:
    results = pinecone_index.query(
        vector=generate_embedding(invoice),
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    return [
        {
            "invoice_id": match.id,
            "similarity_score": float(match.score),
            "metadata": match.metadata,
        }
        for match in getattr(results, "matches", [])
        if float(match.score) > 0.7
    ]
