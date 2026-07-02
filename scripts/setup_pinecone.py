from pinecone import ServerlessSpec

from backend.core.config import get_settings
from backend.core.constants import PineconeNamespace
from backend.db.pinecone import get_pinecone_client
from backend.services.rag.knowledge_base import load_documents_from_directory, load_regulations_to_pinecone


def setup_pinecone() -> None:
    settings = get_settings()
    client = get_pinecone_client()
    existing_indexes = {index["name"] for index in client.list_indexes()}

    if settings.pinecone_index_name not in existing_indexes:
        client.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.pinecone_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
        )

    # Optional bootstrap: if a local docs directory exists, seed regulations namespace.
    docs = load_documents_from_directory("docs", category="regulation")
    if docs:
        count = load_regulations_to_pinecone(docs)
        print(
            f"Seeded {count} regulation vectors into namespace '{PineconeNamespace.REGULATIONS}' "
            f"from {len(docs)} local docs."
        )
    else:
        print("No local docs found to seed into Pinecone.")


if __name__ == "__main__":
    setup_pinecone()
