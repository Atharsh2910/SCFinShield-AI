from typing import Any

from pydantic import BaseModel, Field


class InvoiceCreate(BaseModel):
    invoice_number: str
    supplier_name: str
    buyer_name: str
    amount: float = Field(gt=0)
    invoice_date: str
    po_number: str | None = None
    grn_number: str | None = None
    due_date: str | None = None
    currency: str = "INR"
    lender_name: str | None = None
    line_items: list[dict[str, Any]] = Field(default_factory=list)


class InvoiceResponse(BaseModel):
    id: str
    invoice_number: str
    supplier_name: str
    buyer_name: str
    amount: float
    invoice_date: str
    status: str
    fraud_score: float
    fraud_decision: str | None = None
    fraud_patterns: list[str] = Field(default_factory=list)
    created_at: str


class InvoiceUploadResponse(BaseModel):
    invoice_ids: list[str]
    accepted_count: int
    rejected_count: int
    errors: list[str] = Field(default_factory=list)
