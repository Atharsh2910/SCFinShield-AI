from functools import lru_cache

from pinecone import Pinecone

from backend.core.config import get_settings


class PineconeConfigurationError(RuntimeError):
    pass


@lru_cache()
def get_pinecone_client() -> Pinecone:
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise PineconeConfigurationError("Pinecone credentials are missing. Set PINECONE_API_KEY.")
    return Pinecone(api_key=settings.pinecone_api_key)


def get_pinecone_index():
    settings = get_settings()
    return get_pinecone_client().Index(settings.pinecone_index_name)
