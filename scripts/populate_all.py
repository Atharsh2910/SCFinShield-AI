"""
scripts/populate_all.py
------------------------
One-shot script to fully populate ALL databases from existing invoice data.

Steps:
  1. Populate fingerprint_registry from invoices
  2. Generate fraud_cases for invoices that are flagged
  3. Load fraud_cases into Pinecone (fraud_cases namespace)
  4. Rebuild and verify the NetworkX graph
  5. Print a final census of every table + Pinecone namespace
"""
from __future__ import annotations

import asyncio
import json
import uuid

from backend.db.pinecone import get_pinecone_index
from backend.db.supabase import get_supabase_client
from backend.services.graph.builder import upsert_invoice_graph
from backend.db.graph import get_graph, graph_stats
from backend.services.fingerprinting.sha_fingerprint import generate_sha256_fingerprint
from backend.services.rag.knowledge_base import upsert_fraud_cases_to_pinecone


# ---------------------------------------------------------------------------
# 1. Populate fingerprint_registry
# ---------------------------------------------------------------------------

async def populate_fingerprints() -> int:
    db = get_supabase_client()
    invoices = db.table("invoices").select(
        "id, invoice_number, supplier_id, buyer_id, lender_id, invoice_date, amount, sha256_fingerprint"
    ).execute().data or []

    entity_map: dict[str, str] = {}
    def entity_name(eid: str | None) -> str:
        if not eid:
            return ""
        if eid not in entity_map:
            r = db.table("entities").select("name").eq("id", eid).limit(1).execute()
            entity_map[eid] = (r.data or [{}])[0].get("name", "") if r.data else ""
        return entity_map[eid]

    inserted = 0
    for inv in invoices:
        sha = inv.get("sha256_fingerprint")
        if not sha:
            # Recompute fingerprint
            event = {
                "invoice_number": inv.get("invoice_number", ""),
                "supplier_name": entity_name(inv.get("supplier_id")),
                "buyer_name": entity_name(inv.get("buyer_id")),
                "amount": inv.get("amount", 0),
                "invoice_date": str(inv.get("invoice_date", "")),
                "po_number": "",
            }
            sha = generate_sha256_fingerprint(event)

        # Check if already registered
        existing = db.table("fingerprint_registry").select("id").eq("sha256_hash", sha).limit(1).execute()
        if existing.data:
            continue

        db.table("fingerprint_registry").insert({
            "sha256_hash": sha,
            "invoice_number": inv.get("invoice_number", ""),
            "supplier_id": inv.get("supplier_id"),
            "buyer_id": inv.get("buyer_id"),
            "lender_id": inv.get("lender_id"),
            "amount": float(inv.get("amount") or 0),
            "invoice_date": str(inv.get("invoice_date", "")) or None,
            "status": "active",
        }).execute()
        inserted += 1

    print(f"  fingerprint_registry: +{inserted} new fingerprints ({len(invoices)} invoices processed)")
    return inserted


# ---------------------------------------------------------------------------
# 2. Populate fraud_cases for flagged invoices
# ---------------------------------------------------------------------------

SEVERITY_MAP = {
    "HOLD":   "critical",
    "REVIEW": "medium",
    "PASS":   "low",
}

PATTERN_SCORE_MAP = {
    "duplicate_financing":  0.91,
    "phantom_invoice":      0.88,
    "carousel_trade":       0.93,
    "cascade_amplification": 0.85,
    "velocity_anomaly":     0.82,
}


async def populate_fraud_cases() -> int:
    db = get_supabase_client()
    invoices = db.table("invoices").select("*").neq("fraud_decision", "PASS").execute().data or []

    inserted = 0
    for inv in invoices:
        inv_id = str(inv["id"])
        # Skip if case already exists
        existing = db.table("fraud_cases").select("id").eq("invoice_id", inv_id).limit(1).execute()
        if existing.data:
            continue

        decision = str(inv.get("fraud_decision") or "REVIEW")
        patterns = inv.get("fraud_patterns") or []
        if isinstance(patterns, str):
            try:
                patterns = json.loads(patterns)
            except Exception:
                patterns = []

        primary_signal = patterns[0] if patterns else "velocity_anomaly"
        fraud_score = float(inv.get("fraud_score") or PATTERN_SCORE_MAP.get(primary_signal, 0.75))
        severity = SEVERITY_MAP.get(decision, "medium")
        case_number = f"CASE-{inv.get('invoice_number', inv_id[:8]).upper()}"

        # Prevent duplicate case_number
        cn_existing = db.table("fraud_cases").select("id").eq("case_number", case_number).limit(1).execute()
        if cn_existing.data:
            case_number = f"{case_number}-{str(uuid.uuid4())[:4].upper()}"

        ensemble_scores = {
            "xgboost": round(fraud_score * 0.95, 3),
            "isolation_forest": round(fraud_score * 0.88, 3),
            "siamese": round(fraud_score * 0.72, 3),
            "lsh": round(fraud_score * 0.65, 3),
        }

        alert_narrative = (
            f"Invoice {inv.get('invoice_number')} for INR {float(inv.get('amount') or 0):,.2f} "
            f"was flagged with decision {decision}. "
            f"Primary signal: {primary_signal}. "
            f"Fraud score: {fraud_score:.2%}. "
            f"Patterns detected: {', '.join(patterns) if patterns else 'anomaly'}."
        )

        db.table("fraud_cases").insert({
            "invoice_id": inv_id,
            "case_number": case_number,
            "fraud_patterns": patterns,
            "fraud_score": fraud_score,
            "decision": decision,
            "severity": severity,
            "primary_signal": primary_signal,
            "ensemble_scores": ensemble_scores,
            "shap_values": {p: round(fraud_score / max(len(patterns), 1), 3) for p in patterns},
            "cascade_path": [],
            "alert_narrative": alert_narrative,
            "regulation_citations": [],
            "is_confirmed_fraud": decision == "HOLD",
        }).execute()
        inserted += 1

    print(f"  fraud_cases: +{inserted} new cases ({len(invoices)} flagged invoices processed)")
    return inserted


# ---------------------------------------------------------------------------
# 3. Load fraud_cases → Pinecone
# ---------------------------------------------------------------------------

async def load_fraud_cases_to_pinecone() -> int:
    db = get_supabase_client()
    rows = db.table("fraud_cases").select("*").limit(1000).execute().data or []
    if not rows:
        print("  pinecone[fraud_cases]: no cases to index")
        return 0

    documents = []
    for row in rows:
        case_id = str(row.get("id") or "")
        documents.append({
            "id": case_id,
            "title": f"Fraud Case {row.get('case_number', case_id)}",
            "source": f"fraud_case:{case_id}",
            "category": "fraud_case",
            "content": (
                f"Case Number: {row.get('case_number', '')}\n"
                f"Invoice ID: {row.get('invoice_id', '')}\n"
                f"Decision: {row.get('decision', '')}\n"
                f"Severity: {row.get('severity', '')}\n"
                f"Fraud Score: {row.get('fraud_score', 0)}\n"
                f"Fraud Patterns: {row.get('fraud_patterns', [])}\n"
                f"Primary Signal: {row.get('primary_signal', '')}\n"
                f"Signal Scores: {row.get('ensemble_scores', {})}\n"
                f"Alert Narrative: {row.get('alert_narrative', '')}\n"
                f"Analyst Decision: {row.get('analyst_decision', '')}\n"
            ),
        })

    count = upsert_fraud_cases_to_pinecone(documents)
    print(f"  pinecone[fraud_cases]: +{count} vectors from {len(rows)} case(s)")
    return count


# ---------------------------------------------------------------------------
# 4. Rebuild graph with ALL invoices
# ---------------------------------------------------------------------------

async def rebuild_graph() -> None:
    db = get_supabase_client()
    invoices = db.table("invoices").select(
        "id, invoice_number, supplier_id, buyer_id, lender_id, invoice_date, amount, status"
    ).execute().data or []

    entity_rows = db.table("entities").select("id, name, entity_type").execute().data or []
    entity_map = {r["id"]: r for r in entity_rows}

    for inv in invoices:
        sid = inv.get("supplier_id")
        bid = inv.get("buyer_id")
        if not sid or not bid:
            continue
        await upsert_invoice_graph(
            invoice={
                "invoice_number": inv.get("invoice_number", ""),
                "supplier_name": entity_map.get(sid, {}).get("name", ""),
                "buyer_name": entity_map.get(bid, {}).get("name", ""),
                "lender_name": entity_map.get(inv.get("lender_id", ""), {}).get("name", ""),
                "invoice_date": str(inv.get("invoice_date") or "2024-01-01"),
                "amount": float(inv.get("amount") or 0),
                "status": inv.get("status", "pending"),
            },
            invoice_id=str(inv["id"]),
            supplier_id=str(sid),
            buyer_id=str(bid),
            lender_id=str(inv["lender_id"]) if inv.get("lender_id") else None,
        )

    stats = graph_stats()
    print(f"  graph (NetworkX): {stats['node_count']} nodes, {stats['edge_count']} edges")


# ---------------------------------------------------------------------------
# 5. Final census
# ---------------------------------------------------------------------------

async def print_census() -> None:
    db = get_supabase_client()
    idx = get_pinecone_index()

    print("\n" + "=" * 55)
    print("  DATABASE CENSUS")
    print("=" * 55)
    print("  SUPABASE")
    for table in ["entities", "invoices", "fraud_cases", "investigations",
                  "fingerprint_registry", "audit_log"]:
        try:
            r = db.table(table).select("id", count="exact").execute()
            print(f"    {table:30s}: {r.count:>5} rows")
        except Exception as e:
            print(f"    {table:30s}: ERROR - {e}")

    print("  PINECONE")
    try:
        stats = idx.describe_index_stats()
        print(f"    Total vectors               : {stats.total_vector_count:>5}")
        for ns, ns_stats in (stats.namespaces or {}).items():
            print(f"    namespace[{ns:16s}]: {ns_stats.vector_count:>5}")
    except Exception as e:
        print(f"    ERROR: {e}")

    print("  NETWORKX GRAPH")
    gstats = graph_stats()
    print(f"    nodes                       : {gstats['node_count']:>5}")
    print(f"    edges                       : {gstats['edge_count']:>5}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("\nPopulating all databases...\n")

    print("1/4  fingerprint_registry")
    await populate_fingerprints()

    print("2/4  fraud_cases")
    await populate_fraud_cases()

    print("3/4  Pinecone fraud_cases namespace")
    await load_fraud_cases_to_pinecone()

    print("4/4  NetworkX graph")
    await rebuild_graph()

    await print_census()
    print("\nDone. All databases populated.\n")


if __name__ == "__main__":
    asyncio.run(main())
