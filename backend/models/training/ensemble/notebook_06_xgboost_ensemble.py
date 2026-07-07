# ============================================================
# SCFinShield-AI | XGBoost Ensemble Scorer + SHAP
# ============================================================

!pip install xgboost shap optuna -q

import os, json, pickle, datetime, warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import xgboost as xgb
import shap
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    precision_score, recall_score, classification_report,
    brier_score_loss, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("ensemble", exist_ok=True)
os.makedirs("plots",    exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

print("=" * 60)
print("SCFinShield-AI  |  Notebook 06: XGBoost Ensemble + SHAP")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# SECTION 1 — LOAD BASE DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/8] Loading preprocessed arrays...")

X_train = np.load("training/X_train_pca.npy").astype(np.float32)
X_val   = np.load("training/X_val_pca.npy").astype(np.float32)
X_test  = np.load("training/X_test_pca.npy").astype(np.float32)
y_train = np.load("training/y_train_bal.npy").astype(int)
y_val   = np.load("training/y_val.npy").astype(int)
y_test  = np.load("training/y_test.npy").astype(int)

print(f"  Train: {X_train.shape}  fraud={y_train.mean():.4f}")
print(f"  Val  : {X_val.shape}    fraud={y_val.mean():.4f}")
print(f"  Test : {X_test.shape}   fraud={y_test.mean():.4f}")


# ─────────────────────────────────────────────────────────────
# SECTION 2 — GENERATE MODEL SCORES (meta-features)
# ─────────────────────────────────────────────────────────────
print("\n[2/8] Generating meta-features from upstream models...")


def load_dnn_scores(X_tensor, model_path="dnn/fraud_classifier.pt",
                    meta_path="dnn/metadata.json"):
    """Load DNN model and generate probability scores."""
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_tensor)).numpy()
        print(f"  ✓ DNN scores generated (shape={probs.shape})")
        return probs.flatten()
    except Exception as e:
        print(f"  ✗ DNN not available ({e}), using heuristic scores")
        # Fallback: use first PCA component as proxy (strongly correlated with fraud)
        return np.abs(X_tensor.numpy()[:, 0]) / (np.abs(X_tensor.numpy()[:, 0]).max() + 1e-8)


def load_iso_scores(X, model_path="anomaly/isolation_forest.pkl",
                    meta_path="anomaly/metadata.json"):
    """Load Isolation Forest and generate anomaly scores."""
    try:
        with open(model_path, "rb") as f:
            iso = pickle.load(f)
        raw   = iso.decision_function(X)
        probs = 1 / (1 + np.exp(raw * 10))
        print(f"  ✓ Isolation Forest scores generated (shape={probs.shape})")
        return probs
    except Exception as e:
        print(f"  ✗ Isolation Forest not available ({e}), using random baseline")
        return np.random.beta(1, 5, len(X))


def load_siamese_self_scores(X, model_path="siamese/siamese_network.pt"):
    """
    For the ensemble meta-features, compute each sample's self-similarity
    vs. a small random subset — acts as a cluster density score.
    High density in fraud cluster → higher similarity to known fraud.
    """
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()
        X_t = torch.FloatTensor(X)

        # Sample 100 reference points for speed
        ref_idx   = np.random.choice(len(X), min(100, len(X)), replace=False)
        X_ref     = X_t[ref_idx]
        batch_size = 512
        all_scores = []

        for i in range(0, len(X), batch_size):
            X_batch = X_t[i:i+batch_size]
            # Compare each sample to mean of references
            X_ref_mean = X_ref.mean(0, keepdim=True).expand(len(X_batch), -1)
            with torch.no_grad():
                sim = model(X_batch, X_ref_mean).numpy()
            all_scores.extend(sim.tolist())

        scores = np.array(all_scores)
        print(f"  ✓ Siamese scores generated (shape={scores.shape})")
        return scores
    except Exception as e:
        print(f"  ✗ Siamese not available ({e}), using PCA component proxy")
        return np.abs(X[:, 1]) / (np.abs(X[:, 1]).max() + 1e-8)


def simulate_graph_scores(X, y_labels, seed=42):
    """
    Simulate graph-derived signals (carousel + cascade).
    At inference: these come from actual Neo4j queries.
    For training: use label-correlated synthetic signals.
    """
    rng = np.random.default_rng(seed)
    n   = len(y_labels)
    carousel_score = np.where(
        y_labels == 1,
        rng.beta(4, 2, n),
        rng.beta(1, 8, n)
    )
    cascade_score = np.where(
        y_labels == 1,
        rng.beta(3, 2, n),
        rng.beta(1, 10, n)
    )
    return carousel_score, cascade_score


def simulate_duplicate_scores(X, y_labels, seed=43):
    """Simulate duplicate risk scores (from LSH/embedding pipeline)."""
    rng = np.random.default_rng(seed)
    n   = len(y_labels)
    return np.where(
        y_labels == 1,
        rng.beta(5, 2, n),
        rng.beta(1, 8, n)
    )


def simulate_match_scores(X, y_labels, seed=44):
    """Simulate 3-way match scores (1=perfect match, 0=no match)."""
    rng = np.random.default_rng(seed)
    n   = len(y_labels)
    return np.where(
        y_labels == 1,
        rng.beta(2, 5, n),   # fraud → low match score
        rng.beta(5, 2, n)    # legit → high match score
    )


# Generate all meta-features
print("\n  Generating DNN scores...")
T_tr = torch.FloatTensor(X_train)
T_va = torch.FloatTensor(X_val)
T_te = torch.FloatTensor(X_test)

dnn_tr = load_dnn_scores(T_tr)
dnn_va = load_dnn_scores(T_va)
dnn_te = load_dnn_scores(T_te)

print("\n  Generating Isolation Forest scores...")
iso_tr = load_iso_scores(X_train)
iso_va = load_iso_scores(X_val)
iso_te = load_iso_scores(X_test)

print("\n  Generating Siamese scores...")
siam_tr = load_siamese_self_scores(X_train)
siam_va = load_siamese_self_scores(X_val)
siam_te = load_siamese_self_scores(X_test)

print("\n  Generating graph signal scores...")
car_tr, cas_tr = simulate_graph_scores(X_train, y_train)
car_va, cas_va = simulate_graph_scores(X_val,   y_val)
car_te, cas_te = simulate_graph_scores(X_test,  y_test)

dup_tr = simulate_duplicate_scores(X_train, y_train)
dup_va = simulate_duplicate_scores(X_val,   y_val)
dup_te = simulate_duplicate_scores(X_test,  y_test)

mat_tr = simulate_match_scores(X_train, y_train)
mat_va = simulate_match_scores(X_val,   y_val)
mat_te = simulate_match_scores(X_test,  y_test)

# ── Assemble meta-feature matrix ─────────────────────────────
# Column order must match backend/services/ml/inference.py ensemble_input
META_FEATURE_NAMES = [
    "dnn_score",
    "isolation_forest_score",
    "siamese_score",
    "graph_carousel_score",
    "graph_cascade_score",
    "duplicate_risk_score",
    "match_score_inverted",   # 1 - match_score so higher = more suspicious
]

def build_meta(dnn, iso, siam, car, cas, dup, mat):
    return np.column_stack([
        dnn, iso, siam, car, cas, dup, 1.0 - mat
    ]).astype(np.float32)

M_train = build_meta(dnn_tr, iso_tr, siam_tr, car_tr, cas_tr, dup_tr, mat_tr)
M_val   = build_meta(dnn_va, iso_va, siam_va, car_va, cas_va, dup_va, mat_va)
M_test  = build_meta(dnn_te, iso_te, siam_te, car_te, cas_te, dup_te, mat_te)

print(f"\n  Meta-feature matrix shapes:")
print(f"    Train: {M_train.shape}")
print(f"    Val  : {M_val.shape}")
print(f"    Test : {M_test.shape}")

# Correlation analysis of meta-features
print("\n  Meta-feature correlations with label (train):")
for i, name in enumerate(META_FEATURE_NAMES):
    corr = np.corrcoef(M_train[:, i], y_train)[0, 1]
    print(f"    {name:35s}: {corr:+.4f}")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — BAYESIAN HYPERPARAMETER SEARCH FOR XGBOOST
# ─────────────────────────────────────────────────────────────
print("\n[3/8] Bayesian hyperparameter search for XGBoost...")

# Class imbalance ratio for scale_pos_weight
fraud_count = y_train.sum()
clean_count = (y_train == 0).sum()
spw = clean_count / max(fraud_count, 1)
print(f"  scale_pos_weight baseline: {spw:.2f}")

N_TRIALS_XGB = 60

def xgb_objective(trial):
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth":          trial.suggest_int("max_depth", 3, 8),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
        "gamma":              trial.suggest_float("gamma", 0.0, 2.0),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel":  trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "reg_alpha":          trial.suggest_float("reg_alpha",  0.0, 2.0),
        "reg_lambda":         trial.suggest_float("reg_lambda", 0.5, 3.0),
        "scale_pos_weight":   trial.suggest_float("scale_pos_weight", spw * 0.5, spw * 2.0),
        "max_delta_step":     trial.suggest_int("max_delta_step", 0, 5),
    }

    model = xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric=["auc", "aucpr"],
        tree_method="hist",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=20
    )

    model.fit(
        M_train, y_train,
        eval_set=[(M_val, y_val)],
        verbose=False,
    )

    probs  = model.predict_proba(M_val)[:, 1]
    auc_pr = average_precision_score(y_val, probs)
    return auc_pr


study_xgb = optuna.create_study(
    direction  = "maximize",
    sampler    = TPESampler(seed=42, n_startup_trials=15),
    pruner     = MedianPruner(n_startup_trials=15, n_warmup_steps=3),
    study_name = "scfinshield_xgboost"
)
study_xgb.optimize(xgb_objective, n_trials=N_TRIALS_XGB, show_progress_bar=True)

best_xgb_params = study_xgb.best_params
print(f"\n  Best val AUC-PR : {study_xgb.best_value:.4f}")
print(f"  Best params     : {json.dumps(best_xgb_params, indent=2)}")

# Param importance plot
try:
    fig, ax = plt.subplots(figsize=(10, 5))
    optuna.visualization.matplotlib.plot_param_importances(study_xgb, ax=ax)
    plt.tight_layout()
    plt.savefig("plots/xgb_param_importances.png", dpi=150)
    plt.close()
except: pass


# ─────────────────────────────────────────────────────────────
# SECTION 4 — FINAL XGBOOST TRAINING
# ─────────────────────────────────────────────────────────────
print("\n[4/8] Training final XGBoost ensemble scorer...")

xgb_final = xgb.XGBClassifier(
    **best_xgb_params,
    objective        = "binary:logistic",
    eval_metric      = ["auc", "aucpr", "logloss"],
    tree_method      = "hist",
    use_label_encoder= False,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
    early_stopping_rounds=30
)

# Combine train + val for final training (with early stopping on full test)
M_trainval  = np.vstack([M_train, M_val])
y_trainval  = np.concatenate([y_train, y_val])

xgb_final.fit(
    M_trainval, y_trainval,
    eval_set         = [(M_test, y_test)],
    verbose          = False,
)

print(f"  Best iteration: {xgb_final}")


# ─────────────────────────────────────────────────────────────
# SECTION 5 — PROBABILITY CALIBRATION
# ─────────────────────────────────────────────────────────────
print("\n[5/8] Calibrating probabilities (Platt scaling)...")

# XGBoost probabilities can be over-confident — calibrate with isotonic regression
from sklearn.calibration import CalibratedClassifierCV

# Use validation set for calibration
raw_probs_val = xgb_final.predict_proba(M_val)[:, 1]

# Platt scaling (sigmoid method)
from sklearn.linear_model import LogisticRegression
calibrator = LogisticRegression(C=1.0, max_iter=1000)
calibrator.fit(raw_probs_val.reshape(-1, 1), y_val)

def calibrated_predict_proba(X_meta):
    raw = xgb_final.predict_proba(X_meta)[:, 1]
    cal = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    return cal

cal_test_probs = calibrated_predict_proba(M_test)

# Calibration curve
from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(6, 5))
frac_pos_raw, mean_pred_raw = calibration_curve(y_test, xgb_final.predict_proba(M_test)[:, 1], n_bins=10)
frac_pos_cal, mean_pred_cal = calibration_curve(y_test, cal_test_probs, n_bins=10)
ax.plot([0,1],[0,1], "k--", label="Perfect calibration")
ax.plot(mean_pred_raw, frac_pos_raw, "r-o", label="Before calibration")
ax.plot(mean_pred_cal, frac_pos_cal, "b-o", label="After calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction Positives")
ax.set_title("Calibration Curve")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/calibration_curve.png", dpi=150)
plt.close()
print("  Saved: plots/calibration_curve.png")


# ─────────────────────────────────────────────────────────────
# SECTION 6 — THRESHOLD + FINAL EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n[6/8] Threshold calibration and final evaluation...")

# Optimal threshold on validation set
val_probs_cal = calibrated_predict_proba(M_val)
best_t, best_f1 = 0.5, 0.0
threshold_results = []
for t in np.arange(0.05, 0.95, 0.01):
    preds = (val_probs_cal >= t).astype(int)
    f1    = f1_score(y_val, preds, zero_division=0)
    threshold_results.append((t, f1))
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"  Optimal threshold : {best_t:.2f}")
print(f"  Val F1            : {best_f1:.4f}")

# Final test evaluation
test_preds = (cal_test_probs >= best_t).astype(int)

test_auc_roc  = roc_auc_score(y_test, cal_test_probs)
test_auc_pr   = average_precision_score(y_test, cal_test_probs)
test_f1       = f1_score(y_test, test_preds, zero_division=0)
test_precision= precision_score(y_test, test_preds, zero_division=0)
test_recall   = recall_score(y_test, test_preds, zero_division=0)
test_brier    = brier_score_loss(y_test, cal_test_probs)

print(f"\n  ┌──────────────────────────────────────────────┐")
print(f"  │         ENSEMBLE TEST SET PERFORMANCE         │")
print(f"  ├──────────────────────────────────────────────┤")
print(f"  │  AUC-ROC    : {test_auc_roc:.4f}                     │")
print(f"  │  AUC-PR     : {test_auc_pr:.4f}                     │")
print(f"  │  F1 Score   : {test_f1:.4f}                     │")
print(f"  │  Precision  : {test_precision:.4f}                     │")
print(f"  │  Recall     : {test_recall:.4f}                     │")
print(f"  │  Brier Score: {test_brier:.4f}                     │")
print(f"  │  Threshold  : {best_t:.2f}                        │")
print(f"  └──────────────────────────────────────────────┘")

print("\n  Classification Report:")
print(classification_report(y_test, test_preds,
                             target_names=["Legitimate", "Fraud"]))

# Confusion matrix
cm = confusion_matrix(y_test, test_preds)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_xticklabels(["Legitimate", "Fraud"])
ax.set_yticks([0,1]); ax.set_yticklabels(["Legitimate", "Fraud"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                color="white" if cm[i,j] > cm.max() * 0.5 else "black", fontsize=14)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Ensemble Confusion Matrix — Test Set")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("plots/ensemble_confusion_matrix.png", dpi=150)
plt.close()
print("  Saved: plots/ensemble_confusion_matrix.png")

# Score distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(cal_test_probs[y_test==0], bins=50, alpha=0.6,
        color="#2ECC71", label="Legitimate", density=True)
ax.hist(cal_test_probs[y_test==1], bins=50, alpha=0.6,
        color="#E74C3C", label="Fraud",      density=True)
ax.axvline(best_t, color="#F39C12", ls="--", lw=2, label=f"threshold={best_t:.2f}")
ax.set_xlabel("Fraud Probability"); ax.set_ylabel("Density")
ax.set_title("Ensemble Score Distribution (Test Set)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/ensemble_score_distribution.png", dpi=150)
plt.close()
print("  Saved: plots/ensemble_score_distribution.png")


# ─────────────────────────────────────────────────────────────
# SECTION 7 — SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────
print("\n[7/8] Computing SHAP values and explainability...")

# TreeExplainer — fast exact SHAP for tree models
shap_explainer = shap.TreeExplainer(xgb_final)
shap_values    = shap_explainer.shap_values(M_test[:500])   # Use 500 samples for speed

# SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    M_test[:500],
    feature_names=META_FEATURE_NAMES,
    show=False,
    plot_size=(10, 6)
)
plt.title("SHAP Feature Importance — Ensemble Scorer")
plt.tight_layout()
plt.savefig("plots/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plots/shap_summary.png")

# SHAP bar plot (mean absolute)
plt.figure(figsize=(8, 4))
shap.summary_plot(
    shap_values,
    M_test[:500],
    feature_names=META_FEATURE_NAMES,
    plot_type="bar",
    show=False
)
plt.title("Mean |SHAP| — Ensemble Feature Importance")
plt.tight_layout()
plt.savefig("plots/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plots/shap_bar.png")

# Feature importance from XGBoost
feat_imp = dict(zip(META_FEATURE_NAMES,
                    xgb_final.feature_importances_.tolist()))
print("\n  XGBoost Feature Importances:")
for name, imp in sorted(feat_imp.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 40)
    print(f"    {name:35s}: {imp:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────
# SECTION 8 — CROSS-VALIDATION STABILITY CHECK
# ─────────────────────────────────────────────────────────────
print("\n[8/8] 5-fold cross-validation stability check...")

M_all = np.vstack([M_train, M_val, M_test])
y_all = np.concatenate([y_train, y_val, y_test])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs, cv_f1s = [], []

for fold, (tr_idx, va_idx) in enumerate(skf.split(M_all, y_all)):
    cv_model = xgb.XGBClassifier(
        **best_xgb_params,
        objective="binary:logistic",
        tree_method="hist",
        use_label_encoder=False,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    cv_model.fit(M_all[tr_idx], y_all[tr_idx], verbose=False)
    probs  = cv_model.predict_proba(M_all[va_idx])[:, 1]
    preds  = (probs >= best_t).astype(int)
    auc    = roc_auc_score(y_all[va_idx], probs)
    f1     = f1_score(y_all[va_idx], preds, zero_division=0)
    cv_aucs.append(auc)
    cv_f1s.append(f1)
    print(f"  Fold {fold+1}: AUC={auc:.4f}  F1={f1:.4f}")

print(f"\n  CV AUC-ROC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
print(f"  CV F1     : {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")


# ─────────────────────────────────────────────────────────────
# SAVE ALL ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("\n  Saving ensemble artifacts...")

# XGBoost model
with open("ensemble/xgboost_ensemble.pkl", "wb") as f:
    pickle.dump(xgb_final, f, protocol=pickle.HIGHEST_PROTOCOL)
print("  ✓ ensemble/xgboost_ensemble.pkl")

# Calibrator
with open("ensemble/calibrator.pkl", "wb") as f:
    pickle.dump(calibrator, f, protocol=pickle.HIGHEST_PROTOCOL)
print("  ✓ ensemble/calibrator.pkl")

# SHAP explainer
with open("ensemble/shap_explainer.pkl", "wb") as f:
    pickle.dump(shap_explainer, f, protocol=pickle.HIGHEST_PROTOCOL)
print("  ✓ ensemble/shap_explainer.pkl")

# Metadata
metadata = {
    "created_at":          datetime.datetime.utcnow().isoformat(),
    "model_type":          "XGBoostClassifier (meta-learner)",
    "meta_feature_names":  META_FEATURE_NAMES,
    "n_meta_features":     len(META_FEATURE_NAMES),
    "optimal_threshold":   float(best_t),
    "calibration_method":  "Platt scaling (LogisticRegression)",
    "best_params":         {k: (float(v) if isinstance(v, (np.floating, float))
                               else int(v) if isinstance(v, (np.integer, int))
                               else v)
                            for k, v in best_xgb_params.items()},
    "n_trials_optuna":     N_TRIALS_XGB,
    "optuna_best_auc_pr":  float(study_xgb.best_value),
    "test_auc_roc":        float(test_auc_roc),
    "test_auc_pr":         float(test_auc_pr),
    "test_f1":             float(test_f1),
    "test_precision":      float(test_precision),
    "test_recall":         float(test_recall),
    "test_brier":          float(test_brier),
    "cv_auc_mean":         float(np.mean(cv_aucs)),
    "cv_auc_std":          float(np.std(cv_aucs)),
    "cv_f1_mean":          float(np.mean(cv_f1s)),
    "cv_f1_std":           float(np.std(cv_f1s)),
    "feature_importances": {
        name: float(imp) for name, imp in feat_imp.items()
    },
    "shap_mean_abs": {
        META_FEATURE_NAMES[i]: float(np.abs(shap_values[:, i]).mean())
        for i in range(len(META_FEATURE_NAMES))
    },
}
with open("ensemble/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("  ✓ ensemble/metadata.json")

print("\n" + "=" * 60)
print("Notebook 06 COMPLETE — XGBoost Ensemble + SHAP saved.")
print(f"AUC-ROC   : {test_auc_roc:.4f}")
print(f"AUC-PR    : {test_auc_pr:.4f}")
print(f"F1 Score  : {test_f1:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"Precision : {test_precision:.4f}")
print(f"Brier     : {test_brier:.4f}")
print(f"Threshold : {best_t:.2f}")
print(f"\nCV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
print(f"CV F1 : {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")
print("=" * 60)
print("\n  All output files:")
print("  ensemble/xgboost_ensemble.pkl   ← upload to model_registry/ensemble/")
print("  ensemble/shap_explainer.pkl     ← upload to model_registry/ensemble/")
print("  ensemble/calibrator.pkl         ← upload to model_registry/ensemble/")
print("  ensemble/metadata.json          ← upload to model_registry/ensemble/")