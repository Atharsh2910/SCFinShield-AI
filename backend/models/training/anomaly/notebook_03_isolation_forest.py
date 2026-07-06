# ============================================================
# SCFinShield-AI | Notebook 03: Isolation Forest
# ============================================================
# Inputs   : training/X_train_pca.npy, X_val_pca.npy,
#            training/X_test_pca.npy
#            training/y_train_bal.npy, y_val.npy, y_test.npy
# Outputs  : anomaly/isolation_forest.pkl
#            anomaly/metadata.json
# ============================================================

import os, json, pickle, datetime, warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, precision_recall_curve
)
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("anomaly", exist_ok=True)
os.makedirs("plots",   exist_ok=True)

print("=" * 60)
print("SCFinShield-AI  |  Notebook 03: Isolation Forest")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# SECTION 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading preprocessed arrays...")

X_train = np.load("training/X_train_pca.npy").astype(np.float64)
X_val   = np.load("training/X_val_pca.npy").astype(np.float64)
X_test  = np.load("training/X_test_pca.npy").astype(np.float64)
y_train = np.load("training/y_train_bal.npy").astype(int)
y_val   = np.load("training/y_val.npy").astype(int)
y_test  = np.load("training/y_test.npy").astype(int)

# Isolation Forest is an unsupervised method — train on CLEAN (legitimate) invoices only.
# This is the key design choice: the model learns the boundary of normal behaviour.
X_train_clean = X_train[y_train == 0]
print(f"  Train (clean only): {X_train_clean.shape}")
print(f"  Val               : {X_val.shape}  fraud={y_val.mean():.4f}")
print(f"  Test              : {X_test.shape}  fraud={y_test.mean():.4f}")

# Contamination estimate: expected fraud rate in production data
# Use the original (pre-SMOTE) fraud rate from metadata
try:
    with open("preprocessing/metadata.json") as f:
        meta = json.load(f)
    CONTAMINATION = float(meta.get("fraud_rate_original", 0.05))
except:
    CONTAMINATION = 0.05
print(f"  Contamination param: {CONTAMINATION:.4f}")


# ─────────────────────────────────────────────────────────────
# SECTION 2 — HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Grid-searching Isolation Forest hyperparameters...")

# Grid search on val set AUC-PR (more informative than AUC-ROC for imbalanced data)
param_grid = {
    "n_estimators":     [100, 200, 300],
    "max_samples":      ["auto", 0.8, 0.6],
    "max_features":     [1.0, 0.8, 0.6],
    "contamination":    [CONTAMINATION],
    "bootstrap":        [False, True],
}

best_auc_pr   = 0.0
best_params   = {}
results_log   = []

for params in ParameterGrid(param_grid):
    iso = IsolationForest(
        random_state=42,
        n_jobs=-1,
        **params
    )
    iso.fit(X_train_clean)

    # decision_function: lower = more anomalous
    # Negate and normalise to [0,1] probability proxy
    raw_scores = iso.decision_function(X_val)
    # Sigmoid normalisation for calibrated scores
    prob_scores = 1 / (1 + np.exp(raw_scores * 10))  # scale=10 sharpens the sigmoid

    auc_pr  = average_precision_score(y_val, prob_scores)
    auc_roc = roc_auc_score(y_val, prob_scores)

    results_log.append({**params, "auc_pr": auc_pr, "auc_roc": auc_roc})

    if auc_pr > best_auc_pr:
        best_auc_pr  = auc_pr
        best_params  = params.copy()
        best_iso_val = prob_scores.copy()

print(f"  Best AUC-PR    : {best_auc_pr:.4f}")
print(f"  Best params    : {best_params}")

# Plot search results
results_df_data = sorted(results_log, key=lambda x: x["auc_pr"], reverse=True)[:20]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(results_df_data)),
       [r["auc_pr"] for r in results_df_data], color="#3498DB")
ax.set_xlabel("Config Rank"); ax.set_ylabel("AUC-PR (Val)")
ax.set_title("Isolation Forest Grid Search — Top 20 Configurations")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/isolation_forest_grid_search.png", dpi=150)
plt.close()
print("  Saved: plots/isolation_forest_grid_search.png")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — FINAL MODEL + THRESHOLD CALIBRATION
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Training final Isolation Forest with best params...")

iso_final = IsolationForest(
    random_state=42,
    n_jobs=-1,
    **best_params
)
# Refit on ALL training data (clean + fraud both visible — but
# contamination param guides the boundary estimation)
iso_final.fit(X_train_clean)

# Score validation and test sets
def iso_to_proba(model, X):
    """Convert IF decision_function to a fraud probability [0,1]."""
    raw  = model.decision_function(X)
    prob = 1 / (1 + np.exp(raw * 10))
    return prob

val_probs  = iso_to_proba(iso_final, X_val)
test_probs = iso_to_proba(iso_final, X_test)

# Threshold calibration — maximise F1 on validation set
best_t, best_f1 = 0.5, 0.0
threshold_scan  = []
for t in np.arange(0.05, 0.95, 0.02):
    preds = (val_probs >= t).astype(int)
    f1    = f1_score(y_val, preds, zero_division=0)
    threshold_scan.append((t, f1))
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"  Optimal threshold : {best_t:.2f}")
print(f"  Val F1 at optimal : {best_f1:.4f}")

# Threshold scan plot
fig, ax = plt.subplots(figsize=(7, 4))
threshs, f1s = zip(*threshold_scan)
ax.plot(threshs, f1s, color="#E74C3C", lw=2)
ax.axvline(best_t, color="#2ECC71", ls="--", label=f"best t={best_t:.2f}")
ax.set_xlabel("Threshold"); ax.set_ylabel("F1 Score")
ax.set_title("F1 vs Threshold — Isolation Forest (Val)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/iso_threshold_scan.png", dpi=150)
plt.close()


# ─────────────────────────────────────────────────────────────
# SECTION 4 — TEST SET EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Evaluating on test set...")

test_preds = (test_probs >= best_t).astype(int)
test_auc_roc = roc_auc_score(y_test, test_probs)
test_auc_pr  = average_precision_score(y_test, test_probs)
test_f1      = f1_score(y_test, test_preds, zero_division=0)
test_recall  = (test_preds[y_test == 1]).mean() if (y_test == 1).any() else 0.0

print(f"\n  ┌──────────────────────────────────────────┐")
print(f"  │      TEST SET PERFORMANCE                 │")
print(f"  ├──────────────────────────────────────────┤")
print(f"  │  AUC-ROC  : {test_auc_roc:.4f}                   │")
print(f"  │  AUC-PR   : {test_auc_pr:.4f}                   │")
print(f"  │  F1 Score : {test_f1:.4f}                   │")
print(f"  │  Recall   : {test_recall:.4f}                   │")
print(f"  └──────────────────────────────────────────┘")

print("\n  Classification Report:")
print(classification_report(y_test, test_preds,
                             target_names=["Legitimate", "Fraud"]))

# Anomaly score distribution plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(test_probs[y_test == 0], bins=50, alpha=0.6,
        color="#2ECC71", label="Legitimate", density=True)
ax.hist(test_probs[y_test == 1], bins=50, alpha=0.6,
        color="#E74C3C", label="Fraud", density=True)
ax.axvline(best_t, color="#F39C12", ls="--", lw=2, label=f"threshold={best_t:.2f}")
ax.set_xlabel("Anomaly Probability Score")
ax.set_ylabel("Density")
ax.set_title("Isolation Forest Score Distribution (Test Set)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/iso_score_distribution.png", dpi=150)
plt.close()
print("  Saved: plots/iso_score_distribution.png")


# ─────────────────────────────────────────────────────────────
# SECTION 5 — SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Saving artifacts...")

with open("anomaly/isolation_forest.pkl", "wb") as f:
    pickle.dump(iso_final, f, protocol=pickle.HIGHEST_PROTOCOL)
print("  ✓ anomaly/isolation_forest.pkl")

metadata = {
    "created_at":        datetime.datetime.utcnow().isoformat(),
    "model_type":        "IsolationForest",
    "training_strategy": "clean_only (no fraud samples during fit)",
    "optimal_threshold": float(best_t),
    "score_transform":   "sigmoid(decision_function * 10)",
    "best_params":       best_params,
    "val_auc_pr":        float(best_auc_pr),
    "test_auc_roc":      float(test_auc_roc),
    "test_auc_pr":       float(test_auc_pr),
    "test_f1":           float(test_f1),
    "test_recall":       float(test_recall),
    "contamination":     float(CONTAMINATION),
}
with open("anomaly/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("  ✓ anomaly/metadata.json")

print("\n" + "=" * 60)
print("Notebook 03 COMPLETE — Isolation Forest saved.")
print(f"AUC-ROC  : {test_auc_roc:.4f}")
print(f"AUC-PR   : {test_auc_pr:.4f}")
print(f"F1       : {test_f1:.4f}")
print(f"Threshold: {best_t:.2f}")
print("=" * 60)
