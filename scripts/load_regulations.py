from __future__ import annotations

from pathlib import Path

from backend.services.rag.knowledge_base import load_documents_from_directory, load_regulations_to_pinecone


DEFAULT_REG_DIR = Path("datasets/regulations")


def load_regulations(directory: Path = DEFAULT_REG_DIR) -> None:
    documents = load_documents_from_directory(directory, category="regulation")
    if not documents:
        print(f"No documents found under {directory}.")
        return

    vector_count = load_regulations_to_pinecone(documents)
    print(
        f"Loaded {len(documents)} regulation document(s) from {directory} and "
        f"upserted {vector_count} vector chunk(s) to Pinecone."
    )


if __name__ == "__main__":
    load_regulations()
