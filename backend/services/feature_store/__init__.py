"""
Feature store service.
Feast was removed from the simplified stack.
All feature data is served directly from Supabase PostgreSQL.
"""

from backend.services.feature_store.feature_service import get_entity_history

__all__ = ["get_entity_history"]
