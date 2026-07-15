from functools import lru_cache
from typing import List

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "SCFinShield-AI"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = Field(default="dev-secret-change-me", validation_alias="SECRET_KEY")
    api_v1_str: str = "/api/v1"

    # Supabase
    supabase_url: str = Field(default="", validation_alias="SUPABASE_URL")
    supabase_anon_key: str = Field(
        default="",
        validation_alias=AliasChoices("SUPABASE_ANON_KEY", "SUPABASE_KEY"),
    )
    supabase_service_key: str = Field(
        default="",
        validation_alias=AliasChoices("SUPABASE_SERVICE_KEY", "SUPABASE_KEY"),
    )
    database_url: str = Field(default="", validation_alias="DATABASE_URL")


    # Pinecone
    pinecone_api_key: str = Field(default="", validation_alias="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-east-1", validation_alias="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="scfinshield", validation_alias="PINECONE_INDEX_NAME")
    pinecone_dimension: int = Field(default=384, validation_alias="PINECONE_DIMENSION")

    # Anthropic
    anthropic_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "GROQ_API_KEY"),
    )
    claude_model: str = Field(default="claude-sonnet-4-6", validation_alias="CLAUDE_MODEL")

    # ML
    model_registry_path: str = Field(
        default="./backend/services/ml/model_registry",
        validation_alias="MODEL_REGISTRY_PATH",
    )
    embedding_model: str = "all-MiniLM-L6-v2"

    # Thresholds
    fraud_threshold_pass: float = 0.3
    fraud_threshold_review: float = 0.7
    fraud_threshold_hold: float = 0.85
    lsh_threshold: float = 0.7
    duplicate_similarity_threshold: float = 0.85

    # CORS — stored as plain string, split at runtime to avoid JSON-decode issues
    allowed_origins: str = Field(
        default="http://localhost:5173",
        validation_alias="ALLOWED_ORIGINS",
    )

    @property
    def allowed_origins_list(self) -> List[str]:
        """Return allowed origins as a list, splitting on commas."""
        raw = self.allowed_origins.strip().strip('"').strip("'")
        return [o.strip() for o in raw.split(",") if o.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
        protected_namespaces=("settings_",),
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
