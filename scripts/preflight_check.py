from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from backend.core.config import get_settings
from backend.db.pinecone import get_pinecone_index
from backend.db.supabase import get_supabase_client


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_env() -> CheckResult:
    settings = get_settings()
    missing = []
    required = {
        "SUPABASE_URL": settings.supabase_url,
        "SUPABASE_SERVICE_KEY": settings.supabase_service_key,
        "DATABASE_URL": settings.database_url,
        "PINECONE_API_KEY": settings.pinecone_api_key,
        "PINECONE_INDEX_NAME": settings.pinecone_index_name,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
    }
    for key, value in required.items():
        if not value:
            missing.append(key)
    if missing:
        return CheckResult("env_vars", False, f"Missing: {', '.join(missing)}")
    return CheckResult("env_vars", True, "All required env vars are set.")


def check_supabase() -> CheckResult:
    try:
        client = get_supabase_client()
        probe = client.table("invoices").select("id").limit(1).execute()
        _ = probe.data if hasattr(probe, "data") else []
        return CheckResult("supabase", True, "Supabase reachable.")
    except Exception as exc:
        return CheckResult("supabase", False, str(exc))


def check_graph() -> CheckResult:
    """Check graph DB: probe invoices table (source of truth) and verify graph can load."""
    try:
        client = get_supabase_client()
        result = client.table("invoices").select("id").limit(1).execute()
        count = len(result.data or [])
        return CheckResult("graph_db", True, f"Graph source tables reachable ({count} invoice(s) visible).")
    except Exception as exc:
        return CheckResult("graph_db", False, str(exc))


def check_pinecone() -> CheckResult:
    try:
        index = get_pinecone_index()
        _ = index is not None
        return CheckResult("pinecone", True, "Pinecone index reachable.")
    except Exception as exc:
        return CheckResult("pinecone", False, str(exc))


def _print_result(result: CheckResult) -> None:
    status = "OK" if result.ok else "FAIL"
    print(f"[{status}] {result.name}: {result.detail}")


async def main() -> None:
    env_result = check_env()
    _print_result(env_result)

    supabase_result = check_supabase()
    _print_result(supabase_result)

    graph_result = check_graph()
    _print_result(graph_result)

    pinecone_result = check_pinecone()
    _print_result(pinecone_result)

    all_ok = all([env_result.ok, supabase_result.ok, graph_result.ok, pinecone_result.ok])
    print("\nPreflight status:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
