from __future__ import annotations

from backend.services.rag.knowledge_base import load_documents_from_directory, load_regulations_to_pinecone


def bootstrap_knowledge_base(directory: str = "docs") -> None:
    documents = load_documents_from_directory(directory, category="regulation")
    if not documents:
        print(f"No markdown/text documents found in '{directory}'.")
        return

    vector_count = load_regulations_to_pinecone(documents)
    print(
        f"Loaded {len(documents)} document(s) from '{directory}' "
        f"and wrote {vector_count} vector chunk(s) to Pinecone."
    )


if __name__ == "__main__":
    bootstrap_knowledge_base()
