import re
from io import BytesIO
from typing import Any

from backend.core.exceptions import FileParsingError
from backend.services.ingestion.normaliser import normalise_invoice


INVOICE_PATTERNS = {
    "invoice_number": (
        r"(?:Invoice\s*(?:No\.?|Number|#)|INV)\s*:?\s*([A-Z0-9\-\/]+)",
        r"\b(INV[-\/]?[A-Z0-9\-]+)\b",
    ),
    "po_number": (
        r"(?:P\.?O\.?\s*(?:No\.?|Number|#)|Purchase\s+Order)\s*:?\s*([A-Z0-9\-\/]+)",
    ),
    "grn_number": (r"(?:GRN\s*(?:No\.?|Number|#)?)\s*:?\s*([A-Z0-9\-\/]+)",),
    "amount": (
        r"(?:Grand\s+Total|Total\s+Amount|Amount\s+Due|Invoice\s+Total)\s*:?\s*(?:INR|Rs\.?|USD|\$)?\s*([\d,]+(?:\.\d{1,2})?)",
    ),
    "invoice_date": (
        r"(?:Invoice\s+Date|Date)\s*:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}-\d{2}-\d{2})",
    ),
    "due_date": (
        r"(?:Due\s+Date|Payment\s+Due)\s*:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}-\d{2}-\d{2})",
    ),
    "supplier_name": (
        r"(?:Supplier|Vendor|Seller)\s*:?\s*([^\n\r]+)",
        r"From\s*:?\s*([^\n\r]+)",
    ),
    "buyer_name": (
        r"(?:Buyer|Customer|Client|Bill\s+To)\s*:?\s*([^\n\r]+)",
        r"To\s*:?\s*([^\n\r]+)",
    ),
}


async def parse_pdf(file_bytes: bytes) -> dict[str, Any]:
    text = _extract_text(file_bytes)
    fields = _extract_fields(text)
    return normalise_invoice(fields, source_format="pdf")


def _extract_text(file_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise FileParsingError("PDF parsing requires pdfplumber. Install project requirements first.") from exc

    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as exc:
        raise FileParsingError(f"PDF parsing failed: {exc}") from exc

    if not text.strip():
        raise FileParsingError("No extractable text found in PDF")
    return text


def _extract_fields(text: str) -> dict[str, Any]:
    result = {}
    for field, patterns in INVOICE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result[field] = match.group(1).replace(",", "").strip()
                break
    return result
