import json
import re
import uuid
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from backend.core.exceptions import FileParsingError


FIELD_ALIASES = {
    "invoice_number": ("invoice_number", "invoice_no", "invoice_id", "invoice", "inv_no", "order_id"),
    "supplier_name": ("supplier_name", "vendor_name", "vendor", "supplier", "seller_name", "seller"),
    "buyer_name": ("buyer_name", "customer_name", "customer", "buyer", "client_name", "client"),
    "amount": ("amount", "invoice_amount", "total_amount", "grand_total", "amount_due", "sales"),
    "invoice_date": ("invoice_date", "date", "order_date", "billing_date", "invoice_dt"),
    "po_number": ("po_number", "po_no", "purchase_order", "purchase_order_number"),
    "grn_number": ("grn_number", "grn_no", "goods_receipt_number"),
    "due_date": ("due_date", "payment_due_date", "maturity_date"),
    "currency": ("currency", "currency_code"),
    "lender_name": ("lender_name", "lender", "financier_name", "bank_name"),
    "line_items": ("line_items", "items"),
    "payment_method": ("payment_method", "payment_type", "payment_mode"),
}

REQUIRED_CANONICAL_FIELDS = ("invoice_number", "supplier_name", "buyer_name", "amount", "invoice_date")


def normalise_invoice(raw: dict[str, Any], source_format: str = "csv") -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise FileParsingError("Invoice record must be an object")

    cleaned = {_normalise_key(key): value for key, value in raw.items()}
    invoice = {
        "id": str(uuid.uuid4()),
        "invoice_number": _to_string(_first_value(cleaned, "invoice_number")),
        "supplier_name": _to_string(_first_value(cleaned, "supplier_name")),
        "buyer_name": _to_string(_first_value(cleaned, "buyer_name")),
        "lender_name": _to_string(_first_value(cleaned, "lender_name")),
        "po_number": _to_optional_string(_first_value(cleaned, "po_number")),
        "grn_number": _to_optional_string(_first_value(cleaned, "grn_number")),
        "invoice_date": _parse_date(_first_value(cleaned, "invoice_date")),
        "due_date": _parse_date(_first_value(cleaned, "due_date")),
        "amount": _parse_amount(_first_value(cleaned, "amount")),
        "currency": (_to_string(_first_value(cleaned, "currency")) or "INR").upper(),
        "line_items": _parse_line_items(_first_value(cleaned, "line_items")),
        "payment_method": _to_optional_string(_first_value(cleaned, "payment_method")),
        "source_format": source_format,
        "raw": raw,
    }

    missing = [
        field
        for field in REQUIRED_CANONICAL_FIELDS
        if invoice[field] in (None, "") or (field == "amount" and invoice[field] <= 0)
    ]
    if missing:
        raise FileParsingError(f"Missing or invalid required invoice fields: {missing}")

    return invoice


def _normalise_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _first_value(raw: dict[str, Any], canonical_field: str) -> Any:
    for alias in FIELD_ALIASES[canonical_field]:
        if alias in raw and raw[alias] not in (None, ""):
            return raw[alias]
    return None


def _to_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_optional_string(value: Any) -> str | None:
    text = _to_string(value)
    return text or None


def _parse_amount(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    cleaned = re.sub(r"[^\d.\-]", "", str(value))
    try:
        return float(Decimal(cleaned).quantize(Decimal("0.01")))
    except (InvalidOperation, ValueError):
        raise FileParsingError(f"Invalid invoice amount: {value}")


def _parse_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = str(value).strip()
    formats = (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d/%m/%y",
        "%m/%d/%y",
        "%d-%b-%Y",
        "%d %b %Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    raise FileParsingError(f"Invalid invoice date: {value}")


def _parse_line_items(value: Any) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [{"description": value}]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    return []
