from __future__ import annotations

from fastapi import APIRouter

from backend.core.config import get_settings
from backend.core.constants import PineconeNamespace
from backend.db.supabase import get_supabase_client as _get_supabase
from backend.db.pinecone import get_pinecone_index
from backend.db.supabase import get_supabase_client
from backend.services.ml.model_loader import ModelRegistry
from backend.services.rag.knowledge_base import load_documents_from_directory

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """
    Health check with app info, model status, and dependency readiness.
    """
    settings = get_settings()
    model_registry = ModelRegistry.get_instance()
    models_loaded = {
        name: model_registry.get(name) is not None for name in model_registry.models.keys()  # type: ignore[attr-defined]
    }

    db_connected = False
    try:
        supabase = get_supabase_client()
        probe = supabase.table("invoices").select("id").limit(1).execute()
        _ = probe.data if hasattr(probe, "data") else []
        db_connected = True
    except Exception:
        db_connected = False

    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.app_env,
        "api_version": settings.api_v1_str,
        "models_loaded": models_loaded,
        "db_connected": db_connected,
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
    Lightweight check for external dependencies (graph DB via Supabase, Pinecone).
    """
    graph_ok = False
    pinecone_ok = False

    try:
        client = _get_supabase()
        result = client.table("invoices").select("id").limit(1).execute()
        graph_ok = result is not None
    except Exception:
        graph_ok = False

    try:
        index = get_pinecone_index()
        pinecone_ok = index is not None
    except Exception:
        pinecone_ok = False

    return {
        "graph_db": graph_ok,
        "pinecone": pinecone_ok,
    }


@router.get("/health/knowledge-base")
async def health_knowledge_base() -> dict:
    """
    Return lightweight readiness information for the local and vectorized KB.
    """
    docs = load_documents_from_directory("docs")
    pinecone_ok = False
    try:
        pinecone_ok = get_pinecone_index() is not None
    except Exception:
        pinecone_ok = False

    return {
        "local_documents_found": len(docs),
        "default_namespace": PineconeNamespace.REGULATIONS,
        "pinecone_ready": pinecone_ok,
    }

