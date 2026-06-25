import json
from typing import Any

from pydantic import ValidationError

from backend.core.exceptions import FileParsingError
from backend.schemas.invoice import InvoiceCreate
from backend.services.ingestion.normaliser import normalise_invoice


async def parse_json(file_bytes: bytes) -> list[dict[str, Any]]:
    try:
        payload = json.loads(file_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise FileParsingError("JSON payload must be UTF-8 encoded") from exc
    except json.JSONDecodeError as exc:
        raise FileParsingError(f"Invalid JSON payload: {exc.msg}") from exc

    records = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(record, dict) for record in records):
        raise FileParsingError("JSON payload must be an invoice object or array of invoice objects")

    invoices = []
    errors = []
    for index, record in enumerate(records):
        try:
            normalised = normalise_invoice(record, source_format="json")
            InvoiceCreate(**normalised)
            invoices.append(normalised)
        except (FileParsingError, ValidationError) as exc:
            message = exc.message if isinstance(exc, FileParsingError) else str(exc)
            errors.append(f"item {index}: {message}")

    if errors:
        raise FileParsingError("JSON contains invalid invoice records", {"errors": errors})
    if not invoices:
        raise FileParsingError("JSON payload does not contain invoice records")
    return invoices
