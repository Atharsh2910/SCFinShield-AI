from __future__ import annotations

from typing import Any

from backend.core.constants import AlertSeverity, FraudDecision


def _severity_to_badge(severity: str) -> str:
    # Simple mapping for frontend styling
    return {
        AlertSeverity.LOW: "bg-green-100 text-green-800",
        AlertSeverity.MEDIUM: "bg-yellow-100 text-yellow-800",
        AlertSeverity.HIGH: "bg-orange-100 text-orange-800",
        AlertSeverity.CRITICAL: "bg-red-100 text-red-800",
    }.get(severity, "bg-gray-100 text-gray-800")


def format_alert(fraud_state: dict[str, Any], invoice: dict[str, Any]) -> dict[str, Any]:
    """
    Convert raw fraud results into a structured alert payload for the frontend.
    """
    ensemble_score = float(fraud_state.get("ensemble_score", 0.0) or 0.0)
    fraud_decision = str(fraud_state.get("fraud_decision", FraudDecision.PASS) or FraudDecision.PASS)
    fraud_patterns = fraud_state.get("fraud_patterns", []) or []

    # Severity buckets (keep consistent with other parts of the backend)
    if ensemble_score >= 0.9:
        severity = AlertSeverity.CRITICAL
    elif ensemble_score >= 0.75:
        severity = AlertSeverity.HIGH
    elif ensemble_score >= 0.5:
        severity = AlertSeverity.MEDIUM
    else:
        severity = AlertSeverity.LOW

    primary_signal = fraud_patterns[0] if fraud_patterns else "UNKNOWN"

    return {
        "severity": severity,
        "badge_color": _severity_to_badge(severity),
        "decision_label": fraud_decision,
        "score_display": f"{ensemble_score * 100:.1f}%",
        "primary_signal": primary_signal,
        "signal_breakdown": fraud_state.get("ml_result", {}).get("individual_scores", {}),
        "cascade_warning": "Cascade risk detected" if fraud_state.get("graph_findings", {}).get("has_cascade") else "",
        "top_regulations": (fraud_state.get("rag_result", {}).get("regulation_citations") or [])[:3],
        "recommended_action": {
            FraudDecision.HOLD: "HOLD financing pending review",
            FraudDecision.REVIEW: "REVIEW evidence and request additional documentation",
            FraudDecision.PASS: "No action required",
        }.get(fraud_decision, "REVIEW"),
        "case_id": fraud_state.get("case_id"),
        "invoice": {
            "invoice_number": invoice.get("invoice_number"),
            "supplier_name": invoice.get("supplier_name"),
            "buyer_name": invoice.get("buyer_name"),
            "amount": float(invoice.get("amount", 0) or 0),
            "currency": invoice.get("currency") or "INR",
            "invoice_date": invoice.get("invoice_date"),
        },
    }

