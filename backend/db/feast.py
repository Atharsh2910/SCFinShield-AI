from __future__ import annotations

# Feast feature store was removed from the simplified SCFinShield-AI stack.
# All entity history / feature data is served directly from Supabase PostgreSQL.
# This stub exists to prevent ImportError from any legacy references.


def get_feature_store() -> None:
    raise NotImplementedError(
        "Feast was removed. Use Supabase PostgreSQL directly for feature data."
    )
