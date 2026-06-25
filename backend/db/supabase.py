from functools import lru_cache

from supabase import Client, create_client

from backend.core.config import get_settings


class SupabaseConfigurationError(RuntimeError):
    pass


@lru_cache()
def get_supabase_client() -> Client:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_key:
        raise SupabaseConfigurationError(
            "Supabase credentials are missing. Set SUPABASE_URL and SUPABASE_SERVICE_KEY."
        )
    return create_client(settings.supabase_url, settings.supabase_service_key)


async def get_db() -> Client:
    return get_supabase_client()
