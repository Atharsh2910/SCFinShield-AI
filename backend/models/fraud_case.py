from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base


class FraudCase(Base):
    __tablename__ = "fraud_cases"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, server_default=func.gen_random_uuid())
    invoice_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("invoices.id"))
    case_number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    fraud_patterns: Mapped[list] = mapped_column(JSONB, default=list)
    fraud_score: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(10), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    primary_signal: Mapped[str | None] = mapped_column(String(100))
    ensemble_scores: Mapped[dict] = mapped_column(JSONB, default=dict)
    shap_values: Mapped[dict] = mapped_column(JSONB, default=dict)
    cascade_path: Mapped[list] = mapped_column(JSONB, default=list)
    rag_context: Mapped[str | None] = mapped_column(Text)
    alert_narrative: Mapped[str | None] = mapped_column(Text)
    regulation_citations: Mapped[list] = mapped_column(JSONB, default=list)
    analyst_decision: Mapped[str | None] = mapped_column(String(20))
    analyst_notes: Mapped[str | None] = mapped_column(Text)
    sar_draft: Mapped[str | None] = mapped_column(Text)
    is_confirmed_fraud: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
