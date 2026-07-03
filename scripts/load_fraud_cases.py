from __future__ import annotations

from backend.db.supabase import get_supabase_client
from backend.services.rag.knowledge_base import upsert_fraud_cases_to_pinecone


def _to_document(case_row: dict) -> dict:
    case_id = str(case_row.get("id") or "")
    return {
        "id": case_id,
        "title": f"Fraud Case {case_row.get('case_number', case_id)}",
        "source": f"fraud_case:{case_id}",
        "category": "fraud_case",
        "content": (
            f"Case Number: {case_row.get('case_number', '')}\n"
            f"Invoice ID: {case_row.get('invoice_id', '')}\n"
            f"Decision: {case_row.get('decision', '')}\n"
            f"Severity: {case_row.get('severity', '')}\n"
            f"Fraud Score: {case_row.get('fraud_score', 0)}\n"
            f"Fraud Patterns: {case_row.get('fraud_patterns', [])}\n"
            f"Primary Signal: {case_row.get('primary_signal', '')}\n"
            f"Signal Scores: {case_row.get('ensemble_scores', {})}\n"
            f"SHAP: {case_row.get('shap_values', {})}\n"
            f"Cascade Path: {case_row.get('cascade_path', [])}\n"
            f"Alert Narrative: {case_row.get('alert_narrative', '')}\n"
            f"Analyst Decision: {case_row.get('analyst_decision', '')}\n"
            f"Analyst Notes: {case_row.get('analyst_notes', '')}\n"
            f"Citations: {case_row.get('regulation_citations', [])}\n"
        ),
    }


def load_fraud_cases() -> None:
    client = get_supabase_client()
    rows = client.table("fraud_cases").select("*").limit(1000).execute().data or []
    if not rows:
        print("No fraud_cases rows found to index.")
        return

    documents = [_to_document(row) for row in rows]
    vector_count = upsert_fraud_cases_to_pinecone(documents)
    print(f"Indexed {len(rows)} fraud case(s) with {vector_count} vector chunk(s).")


if __name__ == "__main__":
    load_fraud_cases()
