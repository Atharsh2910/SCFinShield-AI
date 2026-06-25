from pydantic import BaseModel


class BuyerCreate(BaseModel):
    name: str
    gst_number: str | None = None
    pan_number: str | None = None
    sector: str | None = None
    country: str = "India"
    state: str | None = None


class BuyerResponse(BuyerCreate):
    id: str
    entity_type: str = "buyer"
    risk_score: float = 0.0
    is_flagged: bool = False
