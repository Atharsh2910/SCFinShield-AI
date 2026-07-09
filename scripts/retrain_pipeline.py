#!/usr/bin/env python
"""Retraining pipeline for SCFinShield-AI XGBoost fraud classifier.

Usage
-----
    python -m scripts.retrain_pipeline            # full run
    python -m scripts.retrain_pipeline --dry-run  # dry-run (no model saved)

The script:
    1. Connects to Supabase and fetches all labelled invoices.
    2. Builds a feature DataFrame matching ``build_feature_vector()`` schema.
    3. Splits 80/20 train/test (stratified).
    4. Applies SMOTE for class balancing on the training split.
    5. Fits a StandardScaler then PCA (retaining 95 % variance).
    6. Trains an XGBoost classifier.
    7. Evaluates on the test split (F1, AUC, Recall).
    8. Saves artefacts to ``backend/services/ml/model_registry/``:
         - preprocessing/scaler.pkl
         - preprocessing/pca.pkl
         - preprocessing/feature_columns.pkl
         - ensemble/xgboost_ensemble.pkl
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Heavy optional imports — fail early with clear messages
# ---------------------------------------------------------------------------

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    logger.error("imbalanced-learn is not installed. Run: pip install imbalanced-learn")
    sys.exit(1)

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import f1_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    logger.error("scikit-learn is not installed. Run: pip install scikit-learn")
    sys.exit(1)

try:
    import xgboost as xgb
except ImportError:
    logger.error("xgboost is not installed. Run: pip install xgboost")
    sys.exit(1)

from backend.core.config import get_settings
from backend.db.supabase import get_supabase_client
from backend.services.ml.feature_engineering import DEFAULT_FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGISTRY_PATH = Path("backend/services/ml/model_registry")
PREPROCESSING_DIR = REGISTRY_PATH / "preprocessing"
ENSEMBLE_DIR = REGISTRY_PATH / "ensemble"

MIN_FRAUD_SAMPLES: int = 5  # minimum fraud rows required to proceed
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
PCA_VARIANCE: float = 0.95

XGBOOST_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 5,  # adjusted further by SMOTE output ratio
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _extract_features(row: dict[str, Any]) -> dict[str, float]:
    """Map a Supabase invoice row to the ``DEFAULT_FEATURE_COLUMNS`` schema."""
    amount = _safe_float(row.get("amount"), 0.0)
    metadata: dict[str, Any] = row.get("metadata") or {}
    fraud_patterns: list[str] = row.get("fraud_patterns") or []

    # Supplier entity data embedded in metadata (populated by seed/ingestion scripts)
    supplier_avg_amount = _safe_float(metadata.get("supplier_avg_amount"), 0.0)
    supplier_invoice_count = _safe_float(metadata.get("supplier_invoice_count"), 0.0)
    supplier_age_days = _safe_float(metadata.get("supplier_age_days"), 365.0) or 365.0
    supplier_fraud_rate = _safe_float(metadata.get("supplier_fraud_rate"), 0.0)

    amount_log = math.log1p(max(amount, 0.0))
    amount_vs_avg = amount / supplier_avg_amount if supplier_avg_amount > 0 else 0.0

    currency = str(row.get("currency", "INR")).upper()

    return {
        "amount": amount,
        "amount_log": amount_log,
        "has_po": 1.0 if row.get("po_number") else 0.0,
        "has_grn": 1.0 if row.get("grn_number") else 0.0,
        "currency_inr": 1.0 if currency == "INR" else 0.0,
        "duplicate_risk_score": _safe_float(metadata.get("duplicate_risk_score"), 0.0),
        "is_exact_duplicate": 1.0 if metadata.get("is_exact_duplicate") else 0.0,
        "lsh_candidate_count": _safe_float(metadata.get("lsh_candidate_count"), 0.0),
        "max_semantic_similarity": _safe_float(metadata.get("max_semantic_similarity"), 0.0),
        "match_score": _safe_float(metadata.get("match_score"), 0.0),
        "po_match_score": _safe_float(metadata.get("po_match_score"), 0.0),
        "grn_match_score": _safe_float(metadata.get("grn_match_score"), 0.0),
        "anomaly_count": _safe_float(metadata.get("anomaly_count"), 0.0),
        "has_carousel": 1.0 if "carousel_trade" in fraud_patterns else 0.0,
        "carousel_ring_count": _safe_float(metadata.get("carousel_ring_count"), 0.0),
        "cascade_depth": _safe_float(metadata.get("cascade_depth"), 0.0),
        "cascade_exposure": _safe_float(metadata.get("cascade_exposure"), 0.0),
        "supplier_degree": supplier_invoice_count,
        "supplier_age_days": supplier_age_days,
        "supplier_avg_amount": supplier_avg_amount,
        "supplier_fraud_rate": supplier_fraud_rate,
        "amount_vs_supplier_avg": amount_vs_avg,
        "invoices_last_7d": _safe_float(metadata.get("invoices_last_7d"), 0.0),
        "invoices_last_30d": _safe_float(metadata.get("invoices_last_30d"), 0.0),
        "amount_last_7d": _safe_float(metadata.get("amount_last_7d"), 0.0),
    }


def _infer_fraud_label(row: dict[str, Any]) -> int:
    """Derive a binary fraud label from stored Supabase columns."""
    # Primary: explicit is_fraud flag in metadata
    metadata: dict[str, Any] = row.get("metadata") or {}
    if "is_fraud" in metadata:
        return 1 if metadata["is_fraud"] else 0

    # Fallback: fraud_decision column
    decision = str(row.get("fraud_decision") or "").upper()
    if decision == "HOLD":
        return 1
    if decision in ("PASS", ""):
        return 0

    # Fallback: fraud_score threshold
    fraud_score = _safe_float(row.get("fraud_score"), 0.0)
    return 1 if fraud_score >= 0.7 else 0


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def fetch_labelled_invoices() -> list[dict[str, Any]]:
    """Fetch all invoices from Supabase that have a deterministic fraud label."""
    logger.info("Connecting to Supabase…")
    client = get_supabase_client()

    rows: list[dict[str, Any]] = []
    page_size = 1000
    offset = 0

    while True:
        result = (
            client.table("invoices")
            .select(
                "id, invoice_number, amount, currency, po_number, grn_number, "
                "fraud_score, fraud_decision, fraud_patterns, metadata, status"
            )
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch: list[dict[str, Any]] = result.data or []
        rows.extend(batch)
        logger.debug("Fetched {} rows (total so far: {})", len(batch), len(rows))
        if len(batch) < page_size:
            break
        offset += page_size

    logger.info("Total invoice rows fetched: {}", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_dataset(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.Series]:
    """Convert Supabase rows into (X, y) matching feature_engineering schema."""
    records: list[dict[str, float]] = []
    labels: list[int] = []

    for row in rows:
        features = _extract_features(row)
        label = _infer_fraud_label(row)
        records.append(features)
        labels.append(label)

    df = pd.DataFrame(records, columns=DEFAULT_FEATURE_COLUMNS)
    y = pd.Series(labels, name="is_fraud")

    logger.info(
        "Dataset built: {} rows | {} fraud ({:.1f}%)",
        len(df),
        y.sum(),
        100.0 * y.mean(),
    )
    return df, y


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------


def _save_artifact(obj: Any, path: Path, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] Would save artefact to {}", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved artefact → {}", path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(dry_run: bool = False) -> None:
    settings = get_settings()
    registry_base = Path(settings.model_registry_path)
    preprocessing_dir = registry_base / "preprocessing"
    ensemble_dir = registry_base / "ensemble"

    # 1. Fetch data
    rows = fetch_labelled_invoices()
    if not rows:
        logger.error("No invoice rows returned from Supabase. Aborting.")
        sys.exit(1)

    # 2. Build feature DataFrame
    X, y = build_dataset(rows)

    fraud_count = int(y.sum())
    benign_count = int((y == 0).sum())

    if fraud_count < MIN_FRAUD_SAMPLES:
        logger.error(
            "Only {} fraud sample(s) found (minimum required: {}). "
            "Seed more fraud data before retraining.",
            fraud_count,
            MIN_FRAUD_SAMPLES,
        )
        sys.exit(1)

    logger.info(
        "Class distribution — benign: {}, fraud: {} (ratio 1:{:.1f})",
        benign_count,
        fraud_count,
        benign_count / fraud_count if fraud_count else float("inf"),
    )

    # 3. Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y.values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y.values,
    )
    logger.info(
        "Split — train: {} rows, test: {} rows", len(X_train), len(X_test)
    )

    # 4. SMOTE oversampling (training set only)
    logger.info("Applying SMOTE…")
    smote_k = min(fraud_count - 1, 5)  # k_neighbors must be < minority class size
    smote = SMOTE(k_neighbors=smote_k, random_state=RANDOM_STATE)
    try:
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logger.info(
            "After SMOTE — total: {}, fraud: {}",
            len(X_resampled),
            int(y_resampled.sum()),
        )
    except Exception as exc:
        logger.warning("SMOTE failed ({}); continuing without oversampling.", exc)
        X_resampled, y_resampled = X_train, y_train

    # 5. StandardScaler + PCA
    logger.info("Fitting StandardScaler…")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    logger.info("Fitting PCA (retain {}% variance)…", int(PCA_VARIANCE * 100))
    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(
        "PCA: {} → {} components (explained variance: {:.3f})",
        X_scaled.shape[1],
        pca.n_components_,
        float(np.sum(pca.explained_variance_ratio_)),
    )

    # Transform test set with fitted scaler + PCA
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)

    # 6. Train XGBoost
    logger.info("Training XGBoost classifier…")
    pos_weight = max(1, int((y_resampled == 0).sum() / max((y_resampled == 1).sum(), 1)))
    params = {**XGBOOST_PARAMS, "scale_pos_weight": pos_weight}
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_pca,
        y_resampled,
        eval_set=[(X_test_pca, y_test)],
        verbose=False,
    )
    logger.info("XGBoost training complete.")

    # 7. Evaluation
    y_pred = model.predict(X_test_pca)
    y_proba = model.predict_proba(X_test_pca)[:, 1]

    f1 = f1_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # AUC requires both classes in y_test
    if len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = float("nan")
        logger.warning("Test set contains only one class — AUC is undefined.")

    logger.info("=" * 50)
    logger.info("Evaluation results on hold-out test set:")
    logger.info("  F1     : {:.4f}", f1)
    logger.info("  AUC    : {:.4f}", auc)
    logger.info("  Recall : {:.4f}", recall)
    logger.info("=" * 50)

    print("\n--- Retrain Pipeline Results ---")
    print(f"  Training samples  : {len(X_resampled)}")
    print(f"  Test samples      : {len(X_test)}")
    print(f"  PCA components    : {pca.n_components_}")
    print(f"  F1 score          : {f1:.4f}")
    print(f"  AUC-ROC           : {auc:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print()

    # 8. Save artefacts
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    feature_columns = DEFAULT_FEATURE_COLUMNS

    _save_artifact(scaler, preprocessing_dir / "scaler.pkl", dry_run)
    _save_artifact(pca, preprocessing_dir / "pca.pkl", dry_run)
    _save_artifact(feature_columns, preprocessing_dir / "feature_columns.pkl", dry_run)
    _save_artifact(model, ensemble_dir / "xgboost_ensemble.pkl", dry_run)

    if not dry_run:
        # Write lightweight metadata alongside the ensemble model
        import json

        meta = {
            "trained_at": timestamp,
            "train_rows": len(X_resampled),
            "test_rows": len(X_test),
            "fraud_train_count": int(y_resampled.sum()),
            "pca_components": int(pca.n_components_),
            "metrics": {
                "f1": round(f1, 6),
                "auc": round(auc, 6) if not math.isnan(auc) else None,
                "recall": round(recall, 6),
            },
            "feature_columns": feature_columns,
        }
        meta_path = ensemble_dir / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Metadata saved → {}", meta_path)

    if dry_run:
        logger.info("[dry-run] Pipeline complete — no artefacts written to disk.")
    else:
        logger.info("Retrain pipeline complete. Artefacts saved to {}", registry_base)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain the SCFinShield-AI XGBoost fraud detection model."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run the full pipeline but skip saving artefacts to disk.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(dry_run=args.dry_run)
