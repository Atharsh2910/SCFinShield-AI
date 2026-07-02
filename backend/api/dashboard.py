from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from supabase import Client

from backend.db.supabase import get_db

router = APIRouter()


def _parse_date(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


@router.get("/dashboard/summary")
async def summary(db: Client = Depends(get_db)) -> Dict[str, Any]:
    invoices = db.table("invoices").select("id,amount,fraud_decision,fraud_score,status,invoice_date").limit(500).execute()
    rows = invoices.data or []

    total = len(rows)
    flagged = [r for r in rows if r.get("fraud_decision") in ("REVIEW", "HOLD")]
    flagged_count = len(flagged)
    total_exposure = sum(float(r.get("amount") or 0) for r in flagged)
    fraud_rate = (flagged_count / total) if total else 0.0

    return {
        "total_invoices": total,
        "flagged_count": flagged_count,
        "total_exposure": total_exposure,
        "fraud_rate": fraud_rate,
    }


@router.get("/dashboard/timeline")
async def timeline(db: Client = Depends(get_db)) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    last_90 = now - timedelta(days=90)
    invoices = db.table("invoices").select("invoice_date,fraud_decision,fraud_score,amount").limit(1000).execute()
    rows = invoices.data or []

    monthly: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        dt = _parse_date(r.get("invoice_date"))
        if not dt or dt < last_90:
            continue
        key = f"{dt.year:04d}-{dt.month:02d}"
        if key not in monthly:
            monthly[key] = {"month": key, "total": 0, "flagged": 0, "exposure": 0.0}
        monthly[key]["total"] += 1
        if r.get("fraud_decision") in ("REVIEW", "HOLD"):
            monthly[key]["flagged"] += 1
            monthly[key]["exposure"] += float(r.get("amount") or 0)

    timeline_rows = list(monthly.values())
    timeline_rows.sort(key=lambda x: x["month"])
    return {"timeline": timeline_rows[-24:]}  # keep last ~2 years/months if needed


@router.get("/dashboard/risk-heatmap")
async def risk_heatmap(db: Client = Depends(get_db)) -> Dict[str, Any]:
    # Best-effort: tier + sector derived from supplier entity.
    invoices = db.table("invoices").select("supplier_id,amount,fraud_decision").limit(500).execute()
    rows = invoices.data or []
    supplier_ids = list({r.get("supplier_id") for r in rows if r.get("supplier_id")})
    if not supplier_ids:
        return {"matrix": [], "tiers": [1, 2, 3], "sectors": []}

    entities = db.table("entities").select("id,tier,sector").in_("id", supplier_ids).limit(500).execute()
    ent_rows = entities.data or []
    ent_map = {e["id"]: e for e in ent_rows}

    matrix: Dict[str, Dict[str, float]] = {}
    sectors = set()
    for r in rows:
        if r.get("fraud_decision") not in ("REVIEW", "HOLD"):
            continue
        sup = ent_map.get(r.get("supplier_id"))
        tier = sup.get("tier") if sup else None
        sector = (sup.get("sector") or "Unknown") if sup else "Unknown"
        sectors.add(sector)
        if tier is None:
            tier = 1
        matrix.setdefault(str(tier), {}).setdefault(sector, 0.0)
        matrix[str(tier)][sector] += float(r.get("amount") or 0)

    sectors_list = sorted(sectors)
    matrix_rows: List[List[Any]] = []
    for tier in ["1", "2", "3"]:
        matrix_rows.append([matrix.get(tier, {}).get(sector, 0.0) for sector in sectors_list])

    return {"tiers": [1, 2, 3], "sectors": sectors_list, "matrix": matrix_rows}


@router.get("/dashboard/top-risks")
async def top_risks(db: Client = Depends(get_db)) -> Dict[str, Any]:
    rows = (
        db.table("invoices")
        .select("id,invoice_number,amount,fraud_decision,fraud_score,invoice_date")
        .order("fraud_score", desc=True)
        .limit(10)
        .execute()
    )
    return {"top_risks": rows.data or []}


@router.get("/dashboard/recent-alerts")
async def recent_alerts(db: Client = Depends(get_db)) -> Dict[str, Any]:
    rows = (
        db.table("fraud_cases")
        .select("id,case_number,invoice_id,decision,severity,created_at,alert_narrative")
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    return {"alerts": rows.data or []}

