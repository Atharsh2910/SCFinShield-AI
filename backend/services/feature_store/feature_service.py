from __future__ import annotations

from typing import Any
from datetime import datetime, timedelta


async def get_entity_history(
    supplier_id: str,
    db_client,
    lookback_days: int = 90,
) -> dict[str, Any]:
    """
    Compute entity history features from Supabase invoice data.
    Returns feature dict compatible with build_feature_vector().
    """
    try:
        cutoff_7d = (datetime.utcnow() - timedelta(days=7)).date().isoformat()
        cutoff_30d = (datetime.utcnow() - timedelta(days=30)).date().isoformat()

        # All invoices for this supplier
        all_res = (
            db_client.table("invoices")
            .select("amount,fraud_score,fraud_decision,invoice_date")
            .eq("supplier_id", supplier_id)
            .limit(500)
            .execute()
        )
        all_invoices = all_res.data or []

        if not all_invoices:
            return {
                "supplier_invoice_count": 0,
                "supplier_avg_amount": 0.0,
                "supplier_fraud_rate": 0.0,
                "supplier_age_days": 365,
                "invoices_last_7d": 0,
                "invoices_last_30d": 0,
                "amount_last_7d": 0.0,
            }

        amounts = [float(inv.get("amount", 0) or 0) for inv in all_invoices]
        fraud_count = sum(
            1 for inv in all_invoices if inv.get("fraud_decision") in ("HOLD", "REVIEW")
        )

        inv_7d = [inv for inv in all_invoices if str(inv.get("invoice_date", "")) >= cutoff_7d]
        inv_30d = [inv for inv in all_invoices if str(inv.get("invoice_date", "")) >= cutoff_30d]

        return {
            "supplier_invoice_count": len(all_invoices),
            "supplier_avg_amount": sum(amounts) / len(amounts) if amounts else 0.0,
            "supplier_fraud_rate": fraud_count / len(all_invoices) if all_invoices else 0.0,
            "supplier_age_days": 365,
            "invoices_last_7d": len(inv_7d),
            "invoices_last_30d": len(inv_30d),
            "amount_last_7d": sum(float(inv.get("amount", 0) or 0) for inv in inv_7d),
        }
    except Exception:
        return {
            "supplier_invoice_count": 0,
            "supplier_avg_amount": 0.0,
            "supplier_fraud_rate": 0.0,
            "supplier_age_days": 365,
            "invoices_last_7d": 0,
            "invoices_last_30d": 0,
            "amount_last_7d": 0.0,
        }
