from __future__ import annotations

from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.db.pinecone import get_pinecone_index
from backend.services.fingerprinting.embedding import get_embedding_model
from backend.core.constants import PineconeNamespace

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def load_regulations_to_pinecone(documents: list[dict[str, Any]]) -> None:
    """
    Chunk and embed regulation documents into Pinecone regulations namespace.

    Each doc:
      {
        "title": str,
        "content": str,
        "source": str,
        "category": str,
        "id": Optional[str]
      }
    """
    if not documents:
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    model = get_embedding_model()
    index = get_pinecone_index()

    vectors: list[dict[str, Any]] = []

    for doc in documents:
        content = str(doc.get("content") or "").strip()
        if not content:
            continue

        chunks = splitter.split_text(content)
        base_id = str(doc.get("id") or doc.get("title") or "").replace(" ", "_") or "reg"

        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk, normalize_embeddings=True).tolist()
            vectors.append(
                {
                    "id": f"{base_id}_{i}",
                    "values": embedding,
                    "metadata": {
                        "title": str(doc.get("title") or ""),
                        "source": str(doc.get("source") or ""),
                        "category": str(doc.get("category") or ""),
                        "chunk_index": i,
                        "text": chunk[:500],
                    },
                }
            )

            if len(vectors) >= 100:
                index.upsert(vectors=vectors, namespace=PineconeNamespace.REGULATIONS)
                vectors = []

    if vectors:
        index.upsert(vectors=vectors, namespace=PineconeNamespace.REGULATIONS)

