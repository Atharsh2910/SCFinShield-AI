import hashlib
import json
from typing import Any


FINGERPRINT_FIELDS = (
    "invoice_number",
    "supplier_name",
    "buyer_name",
    "amount",
    "invoice_date",
    "po_number",
)


def generate_sha256_fingerprint(invoice: dict[str, Any]) -> str:
    canonical = {}
    for field in FINGERPRINT_FIELDS:
        value = invoice.get(field, "")
        if field == "amount":
            try:
                canonical[field] = f"{float(value):.2f}"
            except (TypeError, ValueError):
                canonical[field] = "0.00"
        else:
            canonical[field] = str(value or "").lower().strip()

    fingerprint_str = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(fingerprint_str.encode("utf-8")).hexdigest()


def check_exact_duplicate(fingerprint: str, db_client) -> bool:
    result = (
        db_client.table("fingerprint_registry")
        .select("id")
        .eq("sha256_hash", fingerprint)
        .limit(1)
        .execute()
    )
    return bool(getattr(result, "data", None))
