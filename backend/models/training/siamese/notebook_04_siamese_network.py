# ============================================================
# SCFinShield-AI | Siamese Network
# ============================================================
import os, json, pickle, datetime, warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from itertools import combinations

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

os.makedirs("siamese", exist_ok=True)
os.makedirs("plots",   exist_ok=True)

print("=" * 60)
print("SCFinShield-AI  |  Notebook 04: Siamese Network")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# SECTION 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Loading preprocessed arrays...")

X_train = np.load("training/X_train_pca.npy").astype(np.float32)
X_val   = np.load("training/X_val_pca.npy").astype(np.float32)
X_test  = np.load("training/X_test_pca.npy").astype(np.float32)
y_train = np.load("training/y_train_bal.npy").astype(int)
y_val   = np.load("training/y_val.npy").astype(int)
y_test  = np.load("training/y_test.npy").astype(int)

INPUT_DIM = X_train.shape[1]
print(f"  Input dim: {INPUT_DIM}")


# ─────────────────────────────────────────────────────────────
# SECTION 2 — PAIR GENERATION
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Generating training pairs...")

def generate_pairs(X, y, max_pairs=200_000, noise_std=0.03, seed=42):
    """
    Create (anchor, pair, label) triplets where:
    - Positive pairs (label=1): same class, with small Gaussian noise added to
      one member to simulate near-duplicate invoices with slight amount/field variation
    - Negative pairs (label=0): different class samples, or same class far apart

    Strategy:
    1. Fraud-fraud pairs (positive)  → simulate same invoice re-submitted to another lender
    2. Clean-clean pairs (positive)  → simulate recurring legitimate invoices
    3. Fraud-clean pairs (negative)  → definitively different
    4. Random same-class distant pairs (negative)
    """
    rng = np.random.default_rng(seed)

    fraud_idx = np.where(y == 1)[0]
    clean_idx = np.where(y == 0)[0]

    pairs_X1, pairs_X2, labels = [], [], []

    # ── Positive pairs: fraud-fraud (near-duplicate simulation) ──
    n_fraud_pairs = min(len(fraud_idx) ** 2 // 2, max_pairs // 4)
    idx1 = rng.choice(fraud_idx, size=n_fraud_pairs, replace=True)
    idx2 = rng.choice(fraud_idx, size=n_fraud_pairs, replace=True)
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    # Add noise to simulate slight modifications
    noise = rng.normal(0, noise_std, (len(idx1), INPUT_DIM)).astype(np.float32)
    pairs_X1.append(X[idx1])
    pairs_X2.append(X[idx2] + noise)
    labels.extend([1] * len(idx1))

    # ── Positive pairs: same-invoice with noise (strongest duplicate signal) ──
    n_self = min(len(fraud_idx) * 3, max_pairs // 4)
    idx_self = rng.choice(fraud_idx, size=n_self, replace=True)
    noise2   = rng.normal(0, noise_std * 0.5, (n_self, INPUT_DIM)).astype(np.float32)
    pairs_X1.append(X[idx_self])
    pairs_X2.append(X[idx_self] + noise2)
    labels.extend([1] * n_self)

    # ── Negative pairs: fraud-clean ──
    n_neg = min(len(fraud_idx) * 4, max_pairs // 4)
    f_idx = rng.choice(fraud_idx, size=n_neg, replace=True)
    c_idx = rng.choice(clean_idx, size=n_neg, replace=True)
    pairs_X1.append(X[f_idx])
    pairs_X2.append(X[c_idx])
    labels.extend([0] * n_neg)

    # ── Negative pairs: clean-clean (different, hard negatives) ──
    n_hard_neg = min(len(clean_idx) * 2, max_pairs // 4)
    c1 = rng.choice(clean_idx, size=n_hard_neg, replace=True)
    c2 = rng.choice(clean_idx, size=n_hard_neg, replace=True)
    mask2 = c1 != c2
    c1, c2 = c1[mask2], c2[mask2]
    pairs_X1.append(X[c1])
    pairs_X2.append(X[c2])
    labels.extend([0] * len(c1))

    X1_arr = np.concatenate(pairs_X1, axis=0)
    X2_arr = np.concatenate(pairs_X2, axis=0)
    y_arr  = np.array(labels, dtype=np.float32)

    # Shuffle
    perm    = rng.permutation(len(y_arr))
    return X1_arr[perm], X2_arr[perm], y_arr[perm]

X1_tr, X2_tr, y_tr = generate_pairs(X_train, y_train, max_pairs=200_000, noise_std=0.03)
X1_va, X2_va, y_va = generate_pairs(X_val,   y_val,   max_pairs=20_000,  noise_std=0.03)
X1_te, X2_te, y_te = generate_pairs(X_test,  y_test,  max_pairs=20_000,  noise_std=0.03)

print(f"  Train pairs: {len(y_tr)}  pos_rate={y_tr.mean():.3f}")
print(f"  Val pairs  : {len(y_va)}  pos_rate={y_va.mean():.3f}")
print(f"  Test pairs : {len(y_te)}  pos_rate={y_te.mean():.3f}")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — DATASET & MODEL
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Building dataset and Siamese architecture...")

class PairDataset(Dataset):
    def __init__(self, X1, X2, y):
        self.X1 = torch.FloatTensor(X1)
        self.X2 = torch.FloatTensor(X2)
        self.y  = torch.FloatTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]


class Encoder(nn.Module):
    """
    Shared encoder branch — maps invoice feature vector to a
    normalised embedding space. L2 normalisation ensures cosine
    similarity is equivalent to dot product (cleaner distance metric).
    """
    def __init__(self, input_dim: int, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, embed_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.net(x)
        return nn.functional.normalize(emb, p=2, dim=1)   # L2 normalise


class SiameseNetwork(nn.Module):
    """
    Siamese network:
    - Shared encoder for both invoice branches
    - Cosine similarity of embeddings → similarity score [0,1]
    - Optional: also accepts absolute difference for richer comparison
    """
    def __init__(self, input_dim: int, embed_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(input_dim, embed_dim)
        # Similarity head: takes [cos_sim, |e1-e2|] → scalar
        self.sim_head = nn.Sequential(
            nn.Linear(embed_dim + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        cos_sim  = nn.functional.cosine_similarity(e1, e2, dim=1, eps=1e-8).unsqueeze(1)
        abs_diff = torch.abs(e1 - e2)  # shape: (B, embed_dim)
        # Pool absolute difference to a single statistic
        diff_mean = abs_diff.mean(dim=1, keepdim=True)  # (B, 1)
        combined  = torch.cat([cos_sim, diff_mean], dim=1)
        return self.sim_head(combined).squeeze(-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding for a single invoice (for Pinecone upsert)."""
        return self.encoder(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss:
    - Pulls positive pairs (similar invoices) together
    - Pushes negative pairs (dissimilar) apart beyond margin m
    Loss = (1-y)*D² + y*max(0, m-D)²
    where D = Euclidean distance between L2-normalised embeddings
    (equivalent to 1 - cosine_similarity when embeddings are L2-normalised)
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, pred_similarity: torch.Tensor, labels: torch.Tensor,
                e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        # Primary: BCE on similarity head output
        bce = nn.functional.binary_cross_entropy(pred_similarity, labels)
        # Secondary: contrastive on embedding distance
        dist = torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1) + 1e-8)
        pos_loss = labels * dist ** 2
        neg_loss = (1 - labels) * torch.clamp(self.margin - dist, min=0) ** 2
        contrastive = (pos_loss + neg_loss).mean()
        # Combined loss
        return 0.6 * bce + 0.4 * contrastive


# ─────────────────────────────────────────────────────────────
# SECTION 4 — TRAINING
# ─────────────────────────────────────────────────────────────
print("\n[4/7] Training Siamese Network...")

EMBED_DIM   = 128
BATCH_SIZE  = 512
EPOCHS      = 60
LR          = 3e-4
WEIGHT_DECAY= 1e-5
PATIENCE    = 12

train_ds = PairDataset(X1_tr, X2_tr, y_tr)
val_ds   = PairDataset(X1_va, X2_va, y_va)
test_ds  = PairDataset(X1_te, X2_te, y_te)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=1024, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=1024, shuffle=False, num_workers=0)

model     = SiameseNetwork(INPUT_DIM, EMBED_DIM).to(DEVICE)
criterion = ContrastiveLoss(margin=1.0)
optimiser = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimiser, T_0=20, T_mult=2
)

best_val_auc  = 0.0
best_state    = None
no_improve    = 0
train_losses  = []
val_auc_hist  = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for X1b, X2b, yb in train_loader:
        X1b, X2b, yb = X1b.to(DEVICE), X2b.to(DEVICE), yb.to(DEVICE)
        optimiser.zero_grad()
        sim   = model(X1b, X2b)
        e1    = model.encoder(X1b)
        e2    = model.encoder(X2b)
        loss  = criterion(sim, yb, e1, e2)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        epoch_loss += loss.item()
    scheduler.step(epoch)

    # Validation
    model.eval()
    all_sims, all_labels = [], []
    with torch.no_grad():
        for X1b, X2b, yb in val_loader:
            X1b, X2b = X1b.to(DEVICE), X2b.to(DEVICE)
            sim = model(X1b, X2b).cpu().numpy()
            all_sims.extend(sim)
            all_labels.extend(yb.numpy())

    val_auc   = roc_auc_score(all_labels, all_sims)
    train_losses.append(epoch_loss / len(train_loader))
    val_auc_hist.append(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve   = 0
    else:
        no_improve += 1

    if epoch % 10 == 0 or epoch == 1:
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | val_auc={val_auc:.4f}")

    if no_improve >= PATIENCE:
        print(f"  Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)
print(f"\n  Best val AUC: {best_val_auc:.4f}")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, color="#E74C3C"); axes[0].set_title("Train Loss")
axes[1].plot(val_auc_hist, color="#3498DB"); axes[1].set_title("Val AUC-ROC")
for ax in axes:
    ax.grid(alpha=0.3); ax.set_xlabel("Epoch")
plt.tight_layout()
plt.savefig("plots/siamese_training.png", dpi=150)
plt.close()


# ─────────────────────────────────────────────────────────────
# SECTION 5 — THRESHOLD CALIBRATION
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Calibrating similarity threshold...")

model.eval()
val_sims, val_labels = [], []
with torch.no_grad():
    for X1b, X2b, yb in val_loader:
        sim = model(X1b.to(DEVICE), X2b.to(DEVICE)).cpu().numpy()
        val_sims.extend(sim)
        val_labels.extend(yb.numpy())

val_sims   = np.array(val_sims)
val_labels = np.array(val_labels).astype(int)

best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.1, 0.95, 0.01):
    preds = (val_sims >= t).astype(int)
    f1    = f1_score(val_labels, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"  Optimal threshold : {best_t:.2f}")
print(f"  Val F1            : {best_f1:.4f}")


# ─────────────────────────────────────────────────────────────
# SECTION 6 — TEST EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n[6/7] Evaluating on test set...")

model.eval()
test_sims, test_labels = [], []
with torch.no_grad():
    for X1b, X2b, yb in test_loader:
        sim = model(X1b.to(DEVICE), X2b.to(DEVICE)).cpu().numpy()
        test_sims.extend(sim)
        test_labels.extend(yb.numpy())

test_sims   = np.array(test_sims)
test_labels = np.array(test_labels).astype(int)
test_preds  = (test_sims >= best_t).astype(int)

t_auc_roc = roc_auc_score(test_labels, test_sims)
t_auc_pr  = average_precision_score(test_labels, test_sims)
t_f1      = f1_score(test_labels, test_preds, zero_division=0)
t_recall  = (test_preds[test_labels == 1]).mean() if (test_labels==1).any() else 0.0

print(f"\n  ┌──────────────────────────────────────────┐")
print(f"  │      TEST SET PERFORMANCE                 │")
print(f"  ├──────────────────────────────────────────┤")
print(f"  │  AUC-ROC  : {t_auc_roc:.4f}                   │")
print(f"  │  AUC-PR   : {t_auc_pr:.4f}                   │")
print(f"  │  F1       : {t_f1:.4f}                   │")
print(f"  │  Recall   : {t_recall:.4f}                   │")
print(f"  └──────────────────────────────────────────┘")

# Similarity score distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(test_sims[test_labels==0], bins=50, alpha=0.6,
        color="#2ECC71", label="Non-duplicate", density=True)
ax.hist(test_sims[test_labels==1], bins=50, alpha=0.6,
        color="#E74C3C", label="Duplicate",     density=True)
ax.axvline(best_t, color="#F39C12", ls="--", lw=2,
           label=f"threshold={best_t:.2f}")
ax.set_xlabel("Similarity Score"); ax.set_ylabel("Density")
ax.set_title("Siamese Similarity Distribution (Test Set)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/siamese_score_distribution.png", dpi=150)
plt.close()
print("  Saved: plots/siamese_score_distribution.png")


# ─────────────────────────────────────────────────────────────
# SECTION 7 — SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("\n[7/7] Saving model artifacts...")

torch.save(model, "siamese/siamese_network.pt")
print("  ✓ siamese/siamese_network.pt")

torch.save(model.state_dict(), "siamese/siamese_state_dict.pt")
print("  ✓ siamese/siamese_state_dict.pt")

metadata = {
    "created_at":         datetime.datetime.utcnow().isoformat(),
    "model_type":         "SiameseNetwork",
    "input_dim":          INPUT_DIM,
    "embed_dim":          EMBED_DIM,
    "optimal_threshold":  float(best_t),
    "test_auc_roc":       float(t_auc_roc),
    "test_auc_pr":        float(t_auc_pr),
    "test_f1":            float(t_f1),
    "test_recall":        float(t_recall),
    "val_auc_roc":        float(best_val_auc),
    "loss_function":      "ContrastiveLoss (BCE + Euclidean contrastive)",
    "pair_noise_std":     0.03,
    "batch_size":         BATCH_SIZE,
    "epochs_trained":     epoch,
    "optimizer":          "AdamW",
    "lr":                 LR,
}
with open("siamese/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("  ✓ siamese/metadata.json")

print("\n" + "=" * 60)
print("Notebook 04 COMPLETE — Siamese Network saved.")
print(f"AUC-ROC  : {t_auc_roc:.4f}")
print(f"F1       : {t_f1:.4f}")
print(f"Threshold: {best_t:.2f}")
print("=" * 60)
