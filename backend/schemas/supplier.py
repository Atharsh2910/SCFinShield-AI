from pydantic import BaseModel, Field


class SupplierCreate(BaseModel):
    name: str
    gst_number: str | None = None
    pan_number: str | None = None
    bank_account: str | None = None
    tier: int | None = Field(default=None, ge=1, le=3)
    sector: str | None = None
    country: str = "India"
    state: str | None = None


class SupplierResponse(SupplierCreate):
    id: str
    entity_type: str = "supplier"
    risk_score: float = 0.0
    is_flagged: bool = False
