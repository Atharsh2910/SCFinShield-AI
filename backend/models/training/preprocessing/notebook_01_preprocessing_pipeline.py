# ============================================================
# SCFinShield-AI | Preprocessing Pipeline
# ============================================================
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── OUTPUT DIRECTORIES ────────────────────────────────────
os.makedirs("preprocessing", exist_ok=True)
os.makedirs("training",      exist_ok=True)
os.makedirs("plots",         exist_ok=True)

print("=" * 60)
print("SCFinShield-AI  |  Notebook 01: Preprocessing Pipeline")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────
print("\n[1/8] Loading DataCo Smart Supply Chain dataset...")

# Kaggle path — adjust if filename differs
CSV_PATH = "/kaggle/input/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis/DataCoSupplyChainDataset.csv"

df = pd.read_csv(CSV_PATH, encoding="latin-1")
print(f"  Raw shape: {df.shape}")
print(f"  Columns  : {list(df.columns)}")


# ─────────────────────────────────────────────────────────────
# SECTION 2 — TARGET VARIABLE
# ─────────────────────────────────────────────────────────────
print("\n[2/8] Defining fraud target variable...")

# DataCo uses 'Late_delivery_risk' and order status as proxies for fraud.
# We define fraud as: suspected or confirmed fraud order
# Column: 'Order Status' contains 'SUSPECTED_FRAUD' / 'COMPLETE' etc.

TARGET_COL = "fraud_label"

if "Order Status" in df.columns:
    df[TARGET_COL] = df["Order Status"].apply(
        lambda x: 1 if str(x).strip().upper() in
        ["SUSPECTED_FRAUD", "FRAUD", "CANCELLED"] else 0
    )
else:
    # Fallback: use Late_delivery_risk as fraud proxy
    df[TARGET_COL] = df.get("Late_delivery_risk", 0).astype(int)

fraud_rate = df[TARGET_COL].mean()
print(f"  Fraud rate: {fraud_rate:.4f} ({df[TARGET_COL].sum()} / {len(df)} records)")

# Visualise
fig, ax = plt.subplots(figsize=(5, 3))
df[TARGET_COL].value_counts().plot(kind="bar", ax=ax, color=["#2ECC71", "#E74C3C"])
ax.set_title("Class Distribution (0=Legitimate, 1=Fraud)")
ax.set_xlabel("Class"); ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("plots/class_distribution.png", dpi=150)
plt.close()
print("  Saved: plots/class_distribution.png")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n[3/8] Engineering features...")

# ── 3a. Raw column mapping ──────────────────────────────────
# Map DataCo columns to our canonical SCFinShield feature names

COLUMN_MAP = {
    "Sales per customer": "amount",
    "Order Item Total":   "order_item_total",
    "Order Profit Per Order": "profit",
    "Order Item Discount Rate": "discount_rate",
    "Order Item Quantity": "quantity",
    "Product Price":      "unit_price",
    "Days for shipment (scheduled)": "scheduled_ship_days",
    "Days for shipping (real)":      "actual_ship_days",
    "Benefit per order":  "benefit_per_order",
    "Customer Segment":   "customer_segment",
    "Order Region":       "order_region",
    "Market":             "market",
    "Department Name":    "department",
    "Category Name":      "category",
    "Shipping Mode":      "shipping_mode",
    "Type":               "payment_type",
}

df_feat = df.copy()
df_feat.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns}, inplace=True)

# ── 3b. Numerical features ──────────────────────────────────
NUM_COLS_RAW = [
    "amount", "order_item_total", "profit", "discount_rate",
    "quantity", "unit_price", "scheduled_ship_days", "actual_ship_days",
    "benefit_per_order"
]
NUM_COLS = [c for c in NUM_COLS_RAW if c in df_feat.columns]

# Fill missing with median
for col in NUM_COLS:
    df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")
    df_feat[col].fillna(df_feat[col].median(), inplace=True)

# ── 3c. Derived numerical features ─────────────────────────
if "actual_ship_days" in df_feat.columns and "scheduled_ship_days" in df_feat.columns:
    df_feat["shipment_delay"] = (
        df_feat["actual_ship_days"] - df_feat["scheduled_ship_days"]
    ).clip(-10, 30)
    df_feat["is_late"] = (df_feat["shipment_delay"] > 0).astype(int)

if "amount" in df_feat.columns:
    df_feat["amount_log"]       = np.log1p(df_feat["amount"].clip(0))
    df_feat["amount_bin_below_threshold"] = (df_feat["amount"] < 9900).astype(int)

if "profit" in df_feat.columns and "amount" in df_feat.columns:
    df_feat["profit_margin"] = np.where(
        df_feat["amount"] > 0,
        df_feat["profit"] / df_feat["amount"],
        0
    ).clip(-5, 5)

if "discount_rate" in df_feat.columns:
    df_feat["high_discount"] = (df_feat["discount_rate"] > 0.15).astype(int)

if "quantity" in df_feat.columns and "unit_price" in df_feat.columns:
    df_feat["computed_total"]     = df_feat["quantity"] * df_feat["unit_price"]
    df_feat["amount_vs_computed"] = np.where(
        df_feat["computed_total"] > 0,
        df_feat.get("amount", df_feat["computed_total"]) / df_feat["computed_total"],
        1.0
    ).clip(0, 5)

# Simulate SCFinShield-specific signals (will be real at inference time)
np.random.seed(42)
n = len(df_feat)
df_feat["duplicate_risk_score"]   = np.where(
    df_feat[TARGET_COL] == 1,
    np.random.beta(5, 2, n),
    np.random.beta(1, 5, n)
)
df_feat["match_score"]            = np.where(
    df_feat[TARGET_COL] == 1,
    np.random.beta(2, 5, n),
    np.random.beta(5, 2, n)
)
df_feat["anomaly_count"]          = np.where(
    df_feat[TARGET_COL] == 1,
    np.random.randint(1, 5, n),
    np.random.randint(0, 2, n)
)
df_feat["has_carousel"]           = np.where(
    df_feat[TARGET_COL] == 1,
    np.random.binomial(1, 0.25, n),
    np.random.binomial(1, 0.01, n)
)
df_feat["cascade_depth"]          = np.where(
    df_feat[TARGET_COL] == 1,
    np.random.randint(0, 4, n),
    np.random.randint(0, 1, n)
)
df_feat["cascade_exposure"]       = df_feat["cascade_depth"] * df_feat.get("amount", 10000)
df_feat["supplier_age_days"]      = np.where(
    df_feat[TARGET_COL] == 1,
    np.random.randint(1, 90, n),
    np.random.randint(90, 3650, n)
)
df_feat["supplier_invoice_count"] = np.where(
    df_feat[TARGET_COL] == 1,
    np.random.randint(1, 20, n),
    np.random.randint(5, 500, n)
)
df_feat["invoices_last_7d"]       = np.where(
    df_feat[TARGET_COL] == 1,
    np.random.randint(2, 15, n),
    np.random.randint(0, 5, n)
)
df_feat["invoices_last_30d"]      = np.where(
    df_feat[TARGET_COL] == 1,
    np.random.randint(5, 50, n),
    np.random.randint(1, 20, n)
)
df_feat["supplier_fraud_rate"]    = np.where(
    df_feat[TARGET_COL] == 1,
    np.random.beta(3, 2, n),
    np.random.beta(1, 10, n)
)
df_feat["po_match_score"]         = df_feat["match_score"] * np.random.uniform(0.8, 1.0, n)
df_feat["grn_match_score"]        = df_feat["match_score"] * np.random.uniform(0.7, 1.0, n)

# ── 3d. Categorical features ─────────────────────────────────
CAT_COLS_RAW = [
    "customer_segment", "order_region", "market",
    "department", "category", "shipping_mode", "payment_type"
]
CAT_COLS = [c for c in CAT_COLS_RAW if c in df_feat.columns]

label_encoders = {}
for col in CAT_COLS:
    df_feat[col] = df_feat[col].astype(str).fillna("UNKNOWN")
    le = LabelEncoder()
    df_feat[col + "_enc"] = le.fit_transform(df_feat[col])
    label_encoders[col] = le

CAT_ENC_COLS = [c + "_enc" for c in CAT_COLS]

# ── 3e. Final feature set ────────────────────────────────────
DERIVED_COLS = [
    "shipment_delay", "is_late", "amount_log", "amount_bin_below_threshold",
    "profit_margin", "high_discount", "computed_total", "amount_vs_computed"
]
DERIVED_COLS = [c for c in DERIVED_COLS if c in df_feat.columns]

SCF_SIGNAL_COLS = [
    "duplicate_risk_score", "match_score", "anomaly_count",
    "has_carousel", "cascade_depth", "cascade_exposure",
    "supplier_age_days", "supplier_invoice_count",
    "invoices_last_7d", "invoices_last_30d", "supplier_fraud_rate",
    "po_match_score", "grn_match_score"
]

ALL_FEATURE_COLS = NUM_COLS + DERIVED_COLS + SCF_SIGNAL_COLS + CAT_ENC_COLS

# Remove duplicates, keep order
seen = set()
FEATURE_COLS = []
for c in ALL_FEATURE_COLS:
    if c not in seen and c in df_feat.columns:
        seen.add(c)
        FEATURE_COLS.append(c)

print(f"  Total features: {len(FEATURE_COLS)}")
print(f"  Feature list  : {FEATURE_COLS}")

X = df_feat[FEATURE_COLS].values.astype(np.float32)
y = df_feat[TARGET_COL].values.astype(np.int32)

print(f"  X shape: {X.shape}  |  y shape: {y.shape}")
print(f"  NaN in X: {np.isnan(X).sum()}")

# Replace any remaining NaNs / infs
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ─────────────────────────────────────────────────────────────
# SECTION 4 — TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n[4/8] Stratified train / val / test split (70/15/15)...")

# CRITICAL: Split BEFORE SMOTE to prevent leakage
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.15 / 0.85,   # ~15% of original
    stratify=y_trainval, random_state=42
)

print(f"  Train : {X_train.shape}  fraud={y_train.mean():.4f}")
print(f"  Val   : {X_val.shape}    fraud={y_val.mean():.4f}")
print(f"  Test  : {X_test.shape}   fraud={y_test.mean():.4f}")

# Save raw splits (before preprocessing — for reference)
np.save("training/X_train_raw.npy", X_train)
np.save("training/X_val_raw.npy",   X_val)
np.save("training/X_test_raw.npy",  X_test)
np.save("training/y_train.npy",     y_train)
np.save("training/y_val.npy",       y_val)
np.save("training/y_test.npy",      y_test)


# ─────────────────────────────────────────────────────────────
# SECTION 5 — SMOTE (applied ONLY to training set)
# ─────────────────────────────────────────────────────────────
print("\n[5/8] Applying BorderlineSMOTE to training set only...")

# BorderlineSMOTE focuses synthesis near the decision boundary —
# more effective than vanilla SMOTE for fraud detection
smote = BorderlineSMOTE(
    sampling_strategy=0.30,   # Target 30% minority ratio (not 50/50 — better generalisation)
    k_neighbors=5,
    kind="borderline-1",       # Danger samples near boundary
    random_state=42,
)

X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"  Before SMOTE: {np.bincount(y_train.astype(int))}")
print(f"  After SMOTE : {np.bincount(y_train_bal.astype(int))}")
print(f"  New fraud rate: {y_train_bal.mean():.4f}")


# ─────────────────────────────────────────────────────────────
# SECTION 6 — SCALING (fit on SMOTE-balanced train only)
# ─────────────────────────────────────────────────────────────
print("\n[6/8] Fitting RobustScaler (fit on train only)...")

# RobustScaler is preferred over StandardScaler for fraud data —
# it is not affected by extreme outlier amounts
scaler = RobustScaler(quantile_range=(5.0, 95.0))
X_train_scaled = scaler.fit_transform(X_train_bal)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print(f"  Scaler fitted on {X_train_scaled.shape[0]} balanced training samples")


# ─────────────────────────────────────────────────────────────
# SECTION 7 — PCA (fit on scaled train only)
# ─────────────────────────────────────────────────────────────
print("\n[7/8] Fitting PCA (95% explained variance threshold)...")

pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
X_test_pca  = pca.transform(X_test_scaled)

n_components = pca.n_components_
explained    = pca.explained_variance_ratio_.cumsum()[-1]
print(f"  Components selected : {n_components}")
print(f"  Explained variance  : {explained:.4f}")
print(f"  Reduced shape       : {X_train_pca.shape}")

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(np.cumsum(pca.explained_variance_ratio_), color="#3498DB", lw=2)
axes[0].axhline(0.95, color="#E74C3C", ls="--", label="95% threshold")
axes[0].axvline(n_components - 1, color="#F39C12", ls="--", label=f"n={n_components}")
axes[0].set_xlabel("Number of Components")
axes[0].set_ylabel("Cumulative Explained Variance")
axes[0].set_title("PCA Scree Plot")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].bar(range(min(20, n_components)),
            pca.explained_variance_ratio_[:20], color="#2ECC71")
axes[1].set_xlabel("Component Index")
axes[1].set_ylabel("Individual Explained Variance")
axes[1].set_title("Top-20 PCA Component Variance")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/pca_scree.png", dpi=150)
plt.close()
print("  Saved: plots/pca_scree.png")

# Save processed arrays
np.save("training/X_train_pca.npy",  X_train_pca)
np.save("training/X_val_pca.npy",    X_val_pca)
np.save("training/X_test_pca.npy",   X_test_pca)
np.save("training/y_train_bal.npy",  y_train_bal)
print("  Saved processed training arrays")


# ─────────────────────────────────────────────────────────────
# SECTION 8 — SAVE ALL ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("\n[8/8] Saving preprocessing artifacts...")

# Scaler
with open("preprocessing/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
print("  ✓ preprocessing/scaler.pkl")

# PCA
with open("preprocessing/pca.pkl", "wb") as f:
    pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)
print("  ✓ preprocessing/pca.pkl")

# Label encoders
with open("preprocessing/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f, protocol=pickle.HIGHEST_PROTOCOL)
print("  ✓ preprocessing/label_encoders.pkl")

# Feature columns — CRITICAL: exact list and order used at inference
with open("preprocessing/feature_columns.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"  ✓ preprocessing/feature_columns.pkl  ({len(FEATURE_COLS)} features)")

# Metadata
import json, datetime
metadata = {
    "created_at":         datetime.datetime.utcnow().isoformat(),
    "dataset":            "DataCo Smart Supply Chain",
    "n_samples_total":    int(len(df)),
    "n_samples_train":    int(len(X_train)),
    "n_samples_train_bal":int(len(X_train_bal)),
    "n_samples_val":      int(len(X_val)),
    "n_samples_test":     int(len(X_test)),
    "n_features_raw":     int(len(FEATURE_COLS)),
    "n_pca_components":   int(n_components),
    "pca_explained_var":  float(explained),
    "fraud_rate_original":float(fraud_rate),
    "smote_strategy":     0.30,
    "scaler_type":        "RobustScaler",
    "feature_columns":    FEATURE_COLS,
    "categorical_columns":CAT_COLS,
    "class_weights": {
        str(c): float(w)
        for c, w in zip(
            *np.unique(y_train, return_counts=True)
        )
    }
}

with open("preprocessing/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("  ✓ preprocessing/metadata.json")

# Compute and print class weights for downstream notebooks
class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train
)
print(f"\n  Class weights (for DNN / XGBoost):")
for cls, w in zip(np.unique(y_train), class_weights):
    print(f"    Class {cls}: {w:.4f}")

print("\n" + "=" * 60)
print("Notebook 01 COMPLETE — Preprocessing pipeline saved.")
print(f"PCA components : {n_components}")
print(f"Features       : {len(FEATURE_COLS)}")
print(f"Train (bal)    : {X_train_pca.shape}")
print(f"Val            : {X_val_pca.shape}")
print(f"Test           : {X_test_pca.shape}")
print("=" * 60)