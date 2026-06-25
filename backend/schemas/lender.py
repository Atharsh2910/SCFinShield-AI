from pydantic import BaseModel


class LenderCreate(BaseModel):
    name: str
    gst_number: str | None = None
    pan_number: str | None = None
    bank_account: str | None = None
    country: str = "India"
    state: str | None = None


class LenderResponse(LenderCreate):
    id: str
    entity_type: str = "lender"
    risk_score: float = 0.0
    is_flagged: bool = False
