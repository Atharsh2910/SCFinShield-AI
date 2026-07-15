from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]

from backend.core.constants import PineconeNamespace
from backend.db.pinecone import get_pinecone_index
from backend.services.fingerprinting.embedding import get_embedding_model

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
UPSERT_BATCH_SIZE = 100


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def load_regulations_to_pinecone(documents: list[dict[str, Any]]) -> int:
    """
    Chunk and embed regulation documents into the Pinecone regulations namespace.
    Returns the number of vectors written.
    """
    return _upsert_chunked_documents(
        documents=documents,
        namespace=PineconeNamespace.REGULATIONS,
        default_category="regulation",
    )


def upsert_fraud_cases_to_pinecone(case_summaries: list[dict[str, Any]]) -> int:
    """
    Chunk and embed fraud case summaries into the Pinecone fraud_cases namespace.
    Each case summary may contain alert narrative, patterns, and analyst notes.
    """
    return _upsert_chunked_documents(
        documents=case_summaries,
        namespace=PineconeNamespace.FRAUD_CASES,
        default_category="fraud_case",
    )


def load_documents_from_directory(directory: str | Path, category: str = "regulation") -> list[dict[str, Any]]:
    """
    Build document dicts from a directory of text/markdown files.
    This is safe to call before API keys exist; it only reads local files.
    """
    base = Path(directory)
    if not base.exists():
        return []

    documents: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        documents.append(
            {
                "id": path.stem,
                "title": path.stem.replace("_", " ").replace("-", " ").title(),
                "content": content,
                "source": str(path),
                "category": category,
            }
        )
    return documents


def search_namespace(
    query_text: str,
    namespace: str,
    top_k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Query a Pinecone namespace directly and return normalized matches.
    """
    index = get_pinecone_index()
    model = get_embedding_model()
    vector = model.encode(query_text, normalize_embeddings=True).tolist()
    result = index.query(
        vector=vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=metadata_filter,
    )
    return [
        {
            "id": match.id,
            "score": float(match.score),
            "metadata": match.metadata,
        }
        for match in result.matches
    ]


def _upsert_chunked_documents(
    documents: list[dict[str, Any]],
    namespace: str,
    default_category: str,
) -> int:
    if not documents:
        return 0

    splitter = get_text_splitter()
    model = get_embedding_model()
    index = get_pinecone_index()

    total_written = 0
    vectors: list[dict[str, Any]] = []

    for doc in documents:
        content = str(doc.get("content") or "").strip()
        if not content:
            continue

        chunks = splitter.split_text(content)
        doc_id = str(doc.get("id") or uuid.uuid4())
        title = str(doc.get("title") or doc_id)
        source = str(doc.get("source") or "")
        category = str(doc.get("category") or default_category)

        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk, normalize_embeddings=True).tolist()
            vectors.append(
                {
                    "id": f"{doc_id}_{i}",
                    "values": embedding,
                    "metadata": {
                        "doc_id": doc_id,
                        "title": title,
                        "source": source,
                        "category": category,
                        "chunk_index": i,
                        "text": chunk[:500],
                    },
                }
            )
            if len(vectors) >= UPSERT_BATCH_SIZE:
                index.upsert(vectors=vectors, namespace=namespace)
                total_written += len(vectors)
                vectors = []

    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)
        total_written += len(vectors)

    return total_written

