from functools import lru_cache
from typing import Any

from backend.core.config import get_settings


class PineconeConfigurationError(RuntimeError):
    pass


@lru_cache()
def get_pinecone_client() -> Any:
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise PineconeConfigurationError(
            "Pinecone client is not installed. Install project requirements first."
        ) from exc

    settings = get_settings()
    if not settings.pinecone_api_key:
        raise PineconeConfigurationError("Pinecone credentials are missing. Set PINECONE_API_KEY.")
    return Pinecone(api_key=settings.pinecone_api_key)


def get_pinecone_index():
    settings = get_settings()
    return get_pinecone_client().Index(settings.pinecone_index_name)
