from decimal import Decimal

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, server_default=func.gen_random_uuid())
    invoice_number: Mapped[str] = mapped_column(String(100), nullable=False)
    supplier_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("entities.id"))
    buyer_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("entities.id"))
    lender_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("entities.id"))
    po_number: Mapped[str | None] = mapped_column(String(100))
    grn_number: Mapped[str | None] = mapped_column(String(100))
    invoice_date: Mapped[Date] = mapped_column(Date, nullable=False)
    due_date: Mapped[Date | None] = mapped_column(Date)
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    line_items: Mapped[list] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(String(30), default="pending")
    sha256_fingerprint: Mapped[str | None] = mapped_column(String(64))
    minhash_signature: Mapped[dict | None] = mapped_column(JSONB)
    fraud_score: Mapped[float] = mapped_column(Float, default=0.0)
    fraud_decision: Mapped[str | None] = mapped_column(String(10))
    fraud_patterns: Mapped[list] = mapped_column(JSONB, default=list)
    match_score: Mapped[float | None] = mapped_column(Float)
    cascade_depth: Mapped[int] = mapped_column(Integer, default=0)
    cascade_exposure: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=0)
    raw_file_url: Mapped[str | None] = mapped_column(Text)
    file_type: Mapped[str | None] = mapped_column(String(10))
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
