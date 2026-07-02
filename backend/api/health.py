from __future__ import annotations

from fastapi import APIRouter

from backend.core.config import get_settings
from backend.db.neo4j import get_neo4j_driver
from backend.db.pinecone import get_pinecone_index
from backend.services.ml.model_loader import ModelRegistry

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """
    Basic health check with app and environment info.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.app_env,
        "api_version": settings.api_v1_str,
    }


@router.get("/health/models")
async def health_models() -> dict:
    """
    Return status of loaded ML artifacts.
    """
    registry = ModelRegistry.get_instance()
    loaded = {name: registry.get(name) is not None for name in registry.models.keys()}  # type: ignore[attr-defined]
    return {"models_loaded": loaded}


@router.get("/health/dependencies")
async def health_dependencies() -> dict:
    """
    Lightweight check for external dependencies (Neo4j, Pinecone).
    """
    neo4j_ok = False
    pinecone_ok = False

    try:
        driver = await get_neo4j_driver()
        neo4j_ok = driver is not None
    except Exception:
        neo4j_ok = False

    try:
        index = get_pinecone_index()
        pinecone_ok = index is not None
    except Exception:
        pinecone_ok = False

    return {
        "neo4j": neo4j_ok,
        "pinecone": pinecone_ok,
    }

