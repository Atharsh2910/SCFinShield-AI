from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from backend.core.config import get_settings
from backend.db.neo4j import get_neo4j_driver
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
        "NEO4J_URI": settings.neo4j_uri,
        "NEO4J_PASSWORD": settings.neo4j_password,
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


async def check_neo4j() -> CheckResult:
    try:
        driver = await get_neo4j_driver()
        async with driver.session() as session:
            await session.run("RETURN 1 as ok")
        return CheckResult("neo4j", True, "Neo4j reachable.")
    except Exception as exc:
        return CheckResult("neo4j", False, str(exc))


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

    neo4j_result = await check_neo4j()
    _print_result(neo4j_result)

    pinecone_result = check_pinecone()
    _print_result(pinecone_result)

    all_ok = all([env_result.ok, supabase_result.ok, neo4j_result.ok, pinecone_result.ok])
    print("\nPreflight status:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
