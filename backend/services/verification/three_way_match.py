from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any


AMOUNT_TOLERANCE_PCT = 0.02
DATE_TOLERANCE_DAYS = 3
QUANTITY_TOLERANCE_PCT = 0.05


def run_three_way_match(
    invoice: dict[str, Any],
    po_record: dict[str, Any] | None,
    grn_record: dict[str, Any] | None,
) -> dict[str, Any]:
    result = {
        "po_match": _match_po(invoice, po_record),
        "grn_match": _match_grn(invoice, grn_record),
        "overall_match_score": 0.0,
        "field_results": {},
        "anomalies": [],
        "pass": False,
    }

    scores = []
    if po_record:
        scores.append(result["po_match"]["score"])
    else:
        result["anomalies"].append("NO_PO_FOUND: Invoice references a PO that does not exist")

    if grn_record:
        scores.append(result["grn_match"]["score"])
    else:
        result["anomalies"].append("NO_GRN_FOUND: No goods receipt note found for this invoice")

    result["overall_match_score"] = round(sum(scores) / len(scores), 4) if scores else 0.0
    result["field_results"] = {
        "po": result["po_match"]["fields"],
        "grn": result["grn_match"]["fields"],
    }
    result["anomalies"].extend(result["po_match"]["anomalies"])
    result["anomalies"].extend(result["grn_match"]["anomalies"])
    result["pass"] = result["overall_match_score"] >= 0.8 and not result["anomalies"]
    return result


def _match_po(invoice: dict[str, Any], po: dict[str, Any] | None) -> dict[str, Any]:
    if not po:
        return {"score": 0.0, "fields": {}, "matched": False, "anomalies": []}

    fields = {
        "po_number": _exact_text_field(invoice.get("po_number"), po.get("po_number")),
        "amount": _amount_field(invoice.get("amount"), po.get("amount")),
        "supplier": _name_field(invoice.get("supplier_name"), po.get("supplier_name")),
        "buyer": _name_field(invoice.get("buyer_name"), po.get("buyer_name")),
    }

    if invoice.get("currency") or po.get("currency"):
        fields["currency"] = _exact_text_field(invoice.get("currency", "INR"), po.get("currency", "INR"))

    quantity_result = _line_item_quantity_field(invoice.get("line_items"), po.get("line_items"))
    if quantity_result is not None:
        fields["line_item_quantity"] = quantity_result

    anomalies = _field_anomalies("PO", fields)
    score = _score_fields(fields)
    return {"score": score, "fields": fields, "matched": score >= 0.8, "anomalies": anomalies}


def _match_grn(invoice: dict[str, Any], grn: dict[str, Any] | None) -> dict[str, Any]:
    if not grn:
        return {"score": 0.0, "fields": {}, "matched": False, "anomalies": []}

    fields = {
        "grn_number": _exact_text_field(invoice.get("grn_number"), grn.get("grn_number")),
        "supplier": _name_field(invoice.get("supplier_name"), grn.get("supplier_name")),
        "buyer": _name_field(invoice.get("buyer_name"), grn.get("buyer_name")),
        "delivery_before_invoice": _delivery_date_field(invoice.get("invoice_date"), grn.get("delivery_date")),
    }

    quantity_result = _line_item_quantity_field(invoice.get("line_items"), grn.get("line_items"))
    if quantity_result is not None:
        fields["line_item_quantity"] = quantity_result

    anomalies = _field_anomalies("GRN", fields)
    score = _score_fields(fields)
    return {"score": score, "fields": fields, "matched": score >= 0.8, "anomalies": anomalies}


def _amount_field(invoice_value: Any, reference_value: Any) -> dict[str, Any]:
    invoice_amount = _to_float(invoice_value)
    reference_amount = _to_float(reference_value)
    diff_pct = abs(invoice_amount - reference_amount) / max(abs(reference_amount), 1.0)
    return {
        "invoice": invoice_amount,
        "reference": reference_amount,
        "diff_pct": round(diff_pct, 4),
        "pass": diff_pct <= AMOUNT_TOLERANCE_PCT,
    }


def _name_field(invoice_value: Any, reference_value: Any) -> dict[str, Any]:
    invoice_text = _clean_text(invoice_value)
    reference_text = _clean_text(reference_value)
    ratio = SequenceMatcher(None, invoice_text, reference_text).ratio() if invoice_text and reference_text else 0.0
    return {
        "invoice": invoice_value or "",
        "reference": reference_value or "",
        "similarity": round(ratio, 4),
        "pass": ratio >= 0.8,
    }


def _exact_text_field(invoice_value: Any, reference_value: Any) -> dict[str, Any]:
    invoice_text = _clean_text(invoice_value)
    reference_text = _clean_text(reference_value)
    both_missing = not invoice_text and not reference_text
    return {
        "invoice": invoice_value or "",
        "reference": reference_value or "",
        "pass": both_missing or invoice_text == reference_text,
    }


def _delivery_date_field(invoice_date: Any, delivery_date: Any) -> dict[str, Any]:
    parsed_invoice_date = _parse_date(invoice_date)
    parsed_delivery_date = _parse_date(delivery_date)
    if not parsed_invoice_date or not parsed_delivery_date:
        return {
            "invoice_date": str(invoice_date or ""),
            "delivery_date": str(delivery_date or ""),
            "days_diff": None,
            "pass": False,
        }

    days_diff = (parsed_invoice_date - parsed_delivery_date).days
    return {
        "invoice_date": parsed_invoice_date.isoformat(),
        "delivery_date": parsed_delivery_date.isoformat(),
        "days_diff": days_diff,
        "pass": -DATE_TOLERANCE_DAYS <= days_diff <= 30,
    }


def _line_item_quantity_field(invoice_items: Any, reference_items: Any) -> dict[str, Any] | None:
    invoice_qty = _sum_quantity(invoice_items)
    reference_qty = _sum_quantity(reference_items)
    if invoice_qty is None or reference_qty is None:
        return None
    diff_pct = abs(invoice_qty - reference_qty) / max(abs(reference_qty), 1.0)
    return {
        "invoice_quantity": invoice_qty,
        "reference_quantity": reference_qty,
        "diff_pct": round(diff_pct, 4),
        "pass": diff_pct <= QUANTITY_TOLERANCE_PCT,
    }


def _score_fields(fields: dict[str, dict[str, Any]]) -> float:
    if not fields:
        return 0.0
    return round(sum(1 for field in fields.values() if field["pass"]) / len(fields), 4)


def _field_anomalies(prefix: str, fields: dict[str, dict[str, Any]]) -> list[str]:
    return [f"{prefix}_{field_name.upper()}_MISMATCH" for field_name, result in fields.items() if not result["pass"]]


def _sum_quantity(items: Any) -> float | None:
    if not isinstance(items, list) or not items:
        return None
    total = 0.0
    found_quantity = False
    for item in items:
        if isinstance(item, dict) and item.get("quantity") not in (None, ""):
            total += _to_float(item["quantity"])
            found_quantity = True
    return total if found_quantity else None


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            continue
    return None


def _to_float(value: Any) -> float:
    try:
        return float(str(value or 0).replace(",", ""))
    except ValueError:
        return 0.0


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").lower().strip().split())
