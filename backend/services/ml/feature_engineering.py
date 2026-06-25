import math
from typing import Any

import numpy as np

from backend.services.ml.model_loader import ModelRegistry


DEFAULT_FEATURE_COLUMNS = [
    "amount",
    "amount_log",
    "has_po",
    "has_grn",
    "currency_inr",
    "duplicate_risk_score",
    "is_exact_duplicate",
    "lsh_candidate_count",
    "max_semantic_similarity",
    "match_score",
    "po_match_score",
    "grn_match_score",
    "anomaly_count",
    "has_carousel",
    "carousel_ring_count",
    "cascade_depth",
    "cascade_exposure",
    "supplier_degree",
    "supplier_age_days",
    "supplier_avg_amount",
    "supplier_fraud_rate",
    "amount_vs_supplier_avg",
    "invoices_last_7d",
    "invoices_last_30d",
    "amount_last_7d",
]


def build_feature_vector(
    invoice: dict[str, Any],
    dedup_result: dict[str, Any] | None = None,
    match_result: dict[str, Any] | None = None,
    graph_context: dict[str, Any] | None = None,
    entity_history: dict[str, Any] | None = None,
) -> np.ndarray:
    dedup_result = dedup_result or {}
    match_result = match_result or {}
    graph_context = graph_context or {}
    entity_history = entity_history or {}

    amount = float(invoice.get("amount", 0) or 0)
    supplier_avg_amount = float(entity_history.get("supplier_avg_amount", 0) or 0)
    similar_invoices = dedup_result.get("similar_invoices", []) or []

    features = {
        "amount": amount,
        "amount_log": math.log1p(max(amount, 0)),
        "has_po": 1.0 if invoice.get("po_number") else 0.0,
        "has_grn": 1.0 if invoice.get("grn_number") else 0.0,
        "currency_inr": 1.0 if str(invoice.get("currency", "INR")).upper() == "INR" else 0.0,
        "duplicate_risk_score": float(dedup_result.get("duplicate_risk_score", 0.0) or 0.0),
        "is_exact_duplicate": 1.0 if dedup_result.get("is_exact_duplicate") else 0.0,
        "lsh_candidate_count": float(len(dedup_result.get("lsh_candidates", []) or [])),
        "max_semantic_similarity": max(
            (float(item.get("similarity_score", 0.0) or 0.0) for item in similar_invoices),
            default=0.0,
        ),
        "match_score": float(match_result.get("overall_match_score", 0.0) or 0.0),
        "po_match_score": float(match_result.get("po_match", {}).get("score", 0.0) or 0.0),
        "grn_match_score": float(match_result.get("grn_match", {}).get("score", 0.0) or 0.0),
        "anomaly_count": float(len(match_result.get("anomalies", []) or [])),
        "has_carousel": 1.0 if graph_context.get("has_carousel") else 0.0,
        "carousel_ring_count": float(graph_context.get("ring_count", 0) or 0),
        "cascade_depth": float(graph_context.get("cascade_depth", 0) or 0),
        "cascade_exposure": float(graph_context.get("total_cascade_exposure", 0.0) or 0.0),
        "supplier_degree": float(entity_history.get("supplier_invoice_count", 0) or 0),
        "supplier_age_days": float(entity_history.get("supplier_age_days", 365) or 365),
        "supplier_avg_amount": supplier_avg_amount,
        "supplier_fraud_rate": float(entity_history.get("supplier_fraud_rate", 0.0) or 0.0),
        "amount_vs_supplier_avg": _ratio(amount, supplier_avg_amount),
        "invoices_last_7d": float(entity_history.get("invoices_last_7d", 0) or 0),
        "invoices_last_30d": float(entity_history.get("invoices_last_30d", 0) or 0),
        "amount_last_7d": float(entity_history.get("amount_last_7d", 0.0) or 0.0),
    }

    registry = ModelRegistry.get_instance()
    feature_columns = registry.get("feature_columns") or DEFAULT_FEATURE_COLUMNS
    return np.array([features.get(column, 0.0) for column in feature_columns], dtype=np.float32)


def apply_preprocessing(feature_vector: np.ndarray) -> np.ndarray:
    registry = ModelRegistry.get_instance()
    scaler = registry.get("scaler")
    pca = registry.get("pca")

    vector_2d = feature_vector.reshape(1, -1)
    if scaler is not None:
        vector_2d = scaler.transform(vector_2d)
    if pca is not None:
        vector_2d = pca.transform(vector_2d)
    return vector_2d[0]


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0
