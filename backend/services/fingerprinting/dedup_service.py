from typing import Any

from backend.db.pinecone import get_pinecone_index
from backend.services.fingerprinting.embedding import find_similar_invoices, upsert_invoice_embedding
from backend.services.fingerprinting.minhash_lsh import find_lsh_candidates, index_invoice
from backend.services.fingerprinting.sha_fingerprint import check_exact_duplicate, generate_sha256_fingerprint


async def run_dedup_pipeline(
    invoice: dict[str, Any],
    invoice_id: str,
    db_client=None,
    pinecone_index=None,
) -> dict[str, Any]:
    sha256 = generate_sha256_fingerprint(invoice)
    is_exact_duplicate = check_exact_duplicate(sha256, db_client) if db_client is not None else False

    lsh_candidates = find_lsh_candidates(invoice, invoice_id)
    similar_invoices = []
    embedding_error = None

    try:
        index = pinecone_index or get_pinecone_index()
        similar_invoices = find_similar_invoices(invoice, index)
        upsert_invoice_embedding(invoice_id, invoice, index)
    except Exception as exc:
        embedding_error = str(exc)

    index_invoice(invoice_id, invoice)

    duplicate_risk_score = 0.0
    if is_exact_duplicate:
        duplicate_risk_score = 1.0
        method = "sha256"
    elif similar_invoices:
        duplicate_risk_score = max(item["similarity_score"] for item in similar_invoices)
        method = "semantic"
    elif lsh_candidates:
        duplicate_risk_score = max(score for _, score in lsh_candidates)
        method = "lsh"
    else:
        method = "none"

    return {
        "sha256_fingerprint": sha256,
        "is_exact_duplicate": is_exact_duplicate,
        "lsh_candidates": [candidate_id for candidate_id, _ in lsh_candidates],
        "similar_invoices": similar_invoices[:5],
        "duplicate_risk_score": duplicate_risk_score,
        "duplicate_details": {
            "method": method,
            "top_match": similar_invoices[0] if similar_invoices else None,
            "embedding_error": embedding_error,
        },
    }
