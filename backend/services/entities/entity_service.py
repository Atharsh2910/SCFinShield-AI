from __future__ import annotations

from typing import Any

from supabase import Client


def _normalize_name(name: str | None) -> str:
    return (name or "").strip()


async def get_or_create_entity(
    db: Client,
    *,
    entity_type: str,
    name: str,
    gst_number: str | None = None,
    pan_number: str | None = None,
    bank_account: str | None = None,
    tier: int | None = None,
    sector: str | None = None,
    country: str = "India",
    state: str | None = None,
) -> str:
    """
    Upsert-like helper for Supabase `entities` table.

    Uses (entity_type, name) and optionally GST/PAN to find an existing entity.
    If none exists, inserts a new one and returns the UUID `id`.
    """
    clean_name = _normalize_name(name)
    if not clean_name:
        raise ValueError("Entity name is required")

    query = db.table("entities").select("id").eq("entity_type", entity_type).eq("name", clean_name)

    if gst_number:
        query = query.eq("gst_number", gst_number)
    if pan_number:
        query = query.eq("pan_number", pan_number)

    existing = query.limit(1).execute()
    if existing.data:
        return str(existing.data[0]["id"])

    payload: dict[str, Any] = {
        "entity_type": entity_type,
        "name": clean_name,
        "gst_number": gst_number,
        "pan_number": pan_number,
        "bank_account": bank_account,
        "tier": tier,
        "sector": sector,
        "country": country,
        "state": state,
    }

    inserted = db.table("entities").insert(payload).execute()
    if not inserted.data:
        # Fallback: re-query once
        existing = (
            db.table("entities")
            .select("id")
            .eq("entity_type", entity_type)
            .eq("name", clean_name)
            .limit(1)
            .execute()
        )
        if existing.data:
            return str(existing.data[0]["id"])
        raise RuntimeError("Failed to create entity in Supabase")

    return str(inserted.data[0]["id"])

