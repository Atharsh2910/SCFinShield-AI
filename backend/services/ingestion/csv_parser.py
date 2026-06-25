import csv
from io import StringIO
from typing import Any

from backend.core.exceptions import FileParsingError
from backend.services.ingestion.normaliser import normalise_invoice


async def parse_csv(file_bytes: bytes) -> list[dict[str, Any]]:
    try:
        text = _decode_csv(file_bytes)
        reader = csv.DictReader(StringIO(text))
        if not reader.fieldnames:
            raise FileParsingError("CSV file is empty or missing a header row")

        invoices = []
        row_errors = []
        for line_number, row in enumerate(reader, start=2):
            if _is_blank_row(row):
                continue
            try:
                invoices.append(normalise_invoice(row, source_format="csv"))
            except FileParsingError as exc:
                row_errors.append(f"line {line_number}: {exc.message}")

        if row_errors:
            raise FileParsingError("CSV contains invalid invoice rows", {"errors": row_errors})
        if not invoices:
            raise FileParsingError("CSV file does not contain invoice rows")
        return invoices
    except FileParsingError:
        raise
    except Exception as exc:
        raise FileParsingError(f"CSV parsing failed: {exc}") from exc


def _decode_csv(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise FileParsingError("CSV file encoding is not supported")


def _is_blank_row(row: dict[str, Any]) -> bool:
    return all(value in (None, "") for value in row.values())
