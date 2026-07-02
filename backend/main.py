from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import fraud, health, invoice, graph, investigation, dashboard, simulator
from backend.core.config import get_settings
from backend.core.logger import setup_logger
from backend.db.neo4j import close_neo4j_driver
from backend.services.ml.model_loader import ModelRegistry

logger = setup_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting SCFinShield-AI backend...")
    ModelRegistry.get_instance()
    logger.info("ML models loaded")
    yield
    # Shutdown
    await close_neo4j_driver()
    logger.info("Shutting down SCFinShield-AI backend")


app = FastAPI(
    title="SCFinShield-AI",
    description="Supply Chain Finance Fraud Detection System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, prefix=settings.api_v1_str, tags=["health"])
app.include_router(invoice.router, prefix=settings.api_v1_str, tags=["invoices"])
app.include_router(fraud.router, prefix=settings.api_v1_str, tags=["fraud"])
app.include_router(graph.router, prefix=settings.api_v1_str, tags=["graph"])
app.include_router(investigation.router, prefix=settings.api_v1_str, tags=["investigation"])
app.include_router(dashboard.router, prefix=settings.api_v1_str, tags=["dashboard"])
app.include_router(simulator.router, prefix=settings.api_v1_str, tags=["simulator"])


@app.get("/")
async def root() -> dict[str, str]:
    return {"app": settings.app_name, "status": "running"}

