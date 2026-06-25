from pinecone import ServerlessSpec

from backend.core.config import get_settings
from backend.db.pinecone import get_pinecone_client


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


if __name__ == "__main__":
    setup_pinecone()
