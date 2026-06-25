from typing import Any

import numpy as np

from backend.core.config import get_settings
from backend.core.constants import FraudDecision
from backend.services.ml.feature_engineering import apply_preprocessing, build_feature_vector
from backend.services.ml.model_loader import ModelRegistry


async def run_ensemble_inference(
    invoice: dict[str, Any],
    dedup_result: dict[str, Any] | None = None,
    match_result: dict[str, Any] | None = None,
    graph_context: dict[str, Any] | None = None,
    entity_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dedup_result = dedup_result or {}
    match_result = match_result or {}
    graph_context = graph_context or {}
    entity_history = entity_history or {}

    registry = ModelRegistry.get_instance()
    raw_features = build_feature_vector(invoice, dedup_result, match_result, graph_context, entity_history)
    processed_features = apply_preprocessing(raw_features)

    scores = {
        "dnn": _run_dnn(registry.get("dnn"), processed_features, dedup_result, match_result),
        "isolation_forest": _run_isolation_forest(registry.get("isolation_forest"), processed_features, dedup_result),
        "siamese": _max_semantic_similarity(dedup_result),
        "graph_carousel": 0.9 if graph_context.get("has_carousel") else 0.0,
        "graph_cascade": min(float(graph_context.get("cascade_depth", 0) or 0) / 5.0, 1.0),
    }
    scores["match_anomaly"] = min(len(match_result.get("anomalies", []) or []) / 5.0, 1.0)

    ensemble_score = _run_xgboost(registry.get("xgboost_ensemble"), scores, dedup_result, match_result)
    shap_values = _compute_shap(registry, processed_features)

    return {
        "individual_scores": scores,
        "ensemble_score": ensemble_score,
        "shap_values": shap_values,
        "top_shap_features": _top_shap(shap_values),
        "fraud_decision": _compute_decision(ensemble_score),
    }


def _run_dnn(model: Any, features: np.ndarray, dedup_result: dict[str, Any], match_result: dict[str, Any]) -> float:
    if model is None:
        return _heuristic_score(dedup_result, match_result)
    try:
        import torch

        with torch.no_grad():
            tensor = torch.FloatTensor(features).unsqueeze(0)
            output = model(tensor)
            return float(output.item() if hasattr(output, "item") else output[0].item())
    except Exception:
        return _heuristic_score(dedup_result, match_result)


def _run_isolation_forest(model: Any, features: np.ndarray, dedup_result: dict[str, Any]) -> float:
    if model is None:
        return float(dedup_result.get("duplicate_risk_score", 0.0) or 0.0) * 0.5
    try:
        anomaly_score = model.decision_function(features.reshape(1, -1))[0]
        return float(1 / (1 + np.exp(anomaly_score)))
    except Exception:
        return 0.0


def _run_xgboost(
    model: Any,
    scores: dict[str, float],
    dedup_result: dict[str, Any],
    match_result: dict[str, Any],
) -> float:
    if model is not None:
        try:
            ensemble_input = np.array(
                [
                    [
                        scores["dnn"],
                        scores["isolation_forest"],
                        scores["siamese"],
                        scores["graph_carousel"],
                        scores["graph_cascade"],
                        float(dedup_result.get("duplicate_risk_score", 0.0) or 0.0),
                        float(match_result.get("overall_match_score", 1.0) or 1.0),
                    ]
                ]
            )
            return float(model.predict_proba(ensemble_input)[0][1])
        except Exception:
            pass

    weights = {
        "dnn": 0.3,
        "isolation_forest": 0.15,
        "siamese": 0.2,
        "graph_carousel": 0.15,
        "graph_cascade": 0.1,
        "match_anomaly": 0.1,
    }
    return min(sum(scores.get(key, 0.0) * weight for key, weight in weights.items()), 1.0)


def _compute_shap(registry: ModelRegistry, features: np.ndarray) -> dict[str, float]:
    explainer = registry.get("shap_explainer")
    if explainer is None:
        return {}
    try:
        shap_result = explainer.shap_values(features.reshape(1, -1))
        feature_names = registry.get("feature_columns") or []
        values = shap_result[0] if isinstance(shap_result, list) else shap_result
        values = values[0] if hasattr(values, "__len__") and len(values) else []
        return {name: float(value) for name, value in zip(feature_names, values)}
    except Exception:
        return {}


def _compute_decision(score: float) -> str:
    settings = get_settings()
    if score >= settings.fraud_threshold_hold:
        return FraudDecision.HOLD
    if score >= settings.fraud_threshold_review:
        return FraudDecision.REVIEW
    return FraudDecision.PASS


def _top_shap(shap_values: dict[str, float], n: int = 5) -> list[dict[str, float | str]]:
    return [
        {"feature": feature, "shap_value": value}
        for feature, value in sorted(shap_values.items(), key=lambda item: abs(item[1]), reverse=True)[:n]
    ]


def _max_semantic_similarity(dedup_result: dict[str, Any]) -> float:
    return max(
        (float(item.get("similarity_score", 0.0) or 0.0) for item in dedup_result.get("similar_invoices", []) or []),
        default=0.0,
    )


def _heuristic_score(dedup_result: dict[str, Any], match_result: dict[str, Any]) -> float:
    score = 0.0
    score += float(dedup_result.get("duplicate_risk_score", 0.0) or 0.0) * 0.5
    score += (1 - float(match_result.get("overall_match_score", 1.0) or 1.0)) * 0.3
    if dedup_result.get("is_exact_duplicate"):
        score += 0.4
    return min(score, 1.0)
