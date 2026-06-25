from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.health import router as health_router
from backend.core.config import get_settings
from backend.core.logger import setup_logger

settings = get_settings()
logger = setup_logger()

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=settings.api_v1_str)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting {}", settings.app_name)


@app.get("/")
async def root():
    return {"app": settings.app_name, "status": "running"}
