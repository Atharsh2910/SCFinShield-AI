from sqlalchemy import Boolean, CheckConstraint, Date, DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base


class Entity(Base):
    __tablename__ = "entities"
    __mapper_args__ = {
        "polymorphic_on": "entity_type",
        "polymorphic_identity": "entity",
    }
    __table_args__ = (
        CheckConstraint("entity_type IN ('supplier', 'buyer', 'lender')", name="entity_type_check"),
        CheckConstraint("tier IN (1, 2, 3)", name="entity_tier_check"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, server_default=func.gen_random_uuid())
    entity_type: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    gst_number: Mapped[str | None] = mapped_column(String(15))
    pan_number: Mapped[str | None] = mapped_column(String(10))
    bank_account: Mapped[str | None] = mapped_column(String(20))
    incorporation_date: Mapped[Date | None] = mapped_column(Date)
    tier: Mapped[int | None] = mapped_column(Integer)
    sector: Mapped[str | None] = mapped_column(String(100))
    country: Mapped[str] = mapped_column(String(50), default="India")
    state: Mapped[str | None] = mapped_column(String(50))
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Supplier(Entity):
    __mapper_args__ = {"polymorphic_identity": "supplier"}
