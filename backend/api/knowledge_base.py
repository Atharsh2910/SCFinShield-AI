from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from backend.core.constants import PineconeNamespace
from backend.services.rag.knowledge_base import (
    load_documents_from_directory,
    load_regulations_to_pinecone,
    search_namespace,
)

router = APIRouter()


@router.post("/knowledge-base/load")
async def load_knowledge_base(
    directory: str = Query(default="docs"),
    category: str = Query(default="regulation"),
) -> dict[str, Any]:
    """
    Load local markdown/text files into the regulations namespace.
    Safe to call after Pinecone credentials are configured.
    """
    base = Path(directory)
    if not base.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    documents = load_documents_from_directory(base, category=category)
    if not documents:
        return {
            "directory": str(base),
            "category": category,
            "document_count": 0,
            "vector_count": 0,
        }

    vector_count = load_regulations_to_pinecone(documents)
    return {
        "directory": str(base),
        "category": category,
        "document_count": len(documents),
        "vector_count": vector_count,
    }


@router.get("/knowledge-base/search")
async def search_knowledge_base(
    query: str,
    namespace: str = Query(default=PineconeNamespace.REGULATIONS),
    top_k: int = Query(default=5, ge=1, le=20),
) -> dict[str, Any]:
    """
    Search indexed knowledge for quick validation/debugging.
    """
    matches = search_namespace(query_text=query, namespace=namespace, top_k=top_k)
    return {
        "query": query,
        "namespace": namespace,
        "matches": matches,
    }

