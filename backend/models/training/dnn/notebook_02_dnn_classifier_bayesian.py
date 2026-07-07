# ============================================================
# SCFinShield-AI | DNN Classifier + Bayesian Opt
# ============================================================
!pip install optuna -q

import os, json, pickle, datetime, warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ── DEVICE ───────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

os.makedirs("dnn",   exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("=" * 60)
print("SCFinShield-AI  |  Notebook 02: DNN + Bayesian Optimisation")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# SECTION 1 — LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Loading preprocessed arrays...")

X_train = np.load("training/X_train_pca.npy").astype(np.float32)
X_val   = np.load("training/X_val_pca.npy").astype(np.float32)
X_test  = np.load("training/X_test_pca.npy").astype(np.float32)
y_train = np.load("training/y_train_bal.npy").astype(np.float32)
y_val   = np.load("training/y_val.npy").astype(np.float32)
y_test  = np.load("training/y_test.npy").astype(np.float32)

INPUT_DIM = X_train.shape[1]
print(f"  X_train: {X_train.shape}  fraud rate: {y_train.mean():.4f}")
print(f"  X_val  : {X_val.shape}    fraud rate: {y_val.mean():.4f}")
print(f"  X_test : {X_test.shape}   fraud rate: {y_test.mean():.4f}")
print(f"  Input dim (PCA): {INPUT_DIM}")

# Tensors
T_Xtr = torch.FloatTensor(X_train).to(DEVICE)
T_ytr = torch.FloatTensor(y_train).to(DEVICE)
T_Xva = torch.FloatTensor(X_val).to(DEVICE)
T_yva = torch.FloatTensor(y_val).to(DEVICE)
T_Xte = torch.FloatTensor(X_test).to(DEVICE)
T_yte = torch.FloatTensor(y_test).to(DEVICE)

# Weighted sampler for DataLoader (handles residual imbalance after SMOTE)
pos_weight_val = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
sample_weights = np.where(y_train == 1, float(pos_weight_val), 1.0)
sampler = WeightedRandomSampler(
    torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)


# ─────────────────────────────────────────────────────────────
# SECTION 2 — MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Defining FraudClassifier architecture...")

class FraudClassifier(nn.Module):
    """
    Deep fully-connected classifier with:
    - BatchNorm after each linear layer (faster convergence, implicit regularisation)
    - GELU activation (smoother than ReLU for fraud probability outputs)
    - Dropout for regularisation
    - Skip connection if consecutive layer dims match (residual learning)
    """
    def __init__(self, input_dim: int, layer_dims: list, dropout_rates: list):
        super().__init__()
        assert len(layer_dims) == len(dropout_rates), \
            "layer_dims and dropout_rates must have the same length"

        self.blocks = nn.ModuleList()
        in_dim = input_dim

        for out_dim, drop_rate in zip(layer_dims, dropout_rates):
            block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(p=drop_rate)
            )
            self.blocks.append(block)
            in_dim = out_dim

        # Output head — single logit (no sigmoid; handled in loss)
        self.output_layer = nn.Linear(in_dim, 1)

        # Weight initialisation — Xavier uniform for stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x).squeeze(-1)   # shape: (batch,)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class FocalLoss(nn.Module):
    """
    Focal Loss — addresses class imbalance at inference time.
    Down-weights easy negatives so the model focuses on hard fraud cases.
    alpha: weight for positive (fraud) class
    gamma: focusing parameter — higher = more focus on hard examples
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs    = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # Probability of the true class
        p_t      = torch.where(targets == 1, probs, 1 - probs)
        alpha_t  = torch.where(targets == 1,
                               torch.tensor(self.alpha, device=logits.device),
                               torch.tensor(1 - self.alpha, device=logits.device))
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────
# SECTION 3 — TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, X_tensor, y_np, threshold=0.5, device=DEVICE):
    model.eval()
    probs = torch.sigmoid(model(X_tensor)).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    return {
        "auc_roc":  roc_auc_score(y_np, probs),
        "auc_pr":   average_precision_score(y_np, probs),
        "f1":       f1_score(y_np, preds, zero_division=0),
        "recall":   recall_score(y_np, preds, zero_division=0),
        "precision":precision_score(y_np, preds, zero_division=0),
        "probs":    probs,
        "preds":    preds,
    }


def find_optimal_threshold(model, X_val_tensor, y_val_np, device=DEVICE):
    """Sweep thresholds 0.1–0.9, return the one maximising F1 on val set."""
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_val_tensor)).cpu().numpy()
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        f1    = f1_score(y_val_np, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ─────────────────────────────────────────────────────────────
# SECTION 4 — BAYESIAN HYPERPARAMETER OPTIMISATION
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Running Bayesian hyperparameter search (Optuna)...")

N_TRIALS     = 50          # Increase to 100 for better coverage
SEARCH_EPOCHS = 20         # Epochs per trial (fast proxy)
FINAL_EPOCHS  = 100        # Full training epochs for best config
PATIENCE      = 15         # Early stopping patience


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective: maximise validation F1.
    Search space covers architecture depth, width, dropout, LR, batch size.
    """
    # Architecture
    n_layers  = trial.suggest_int("n_layers", 2, 5)
    layer_dims = [
        trial.suggest_categorical(f"dim_l{i}", [64, 128, 256, 512])
        for i in range(n_layers)
    ]
    # Enforce monotonically decreasing width (bottleneck structure)
    layer_dims = sorted(layer_dims, reverse=True)
    dropout_rates = [
        trial.suggest_float(f"dropout_l{i}", 0.1, 0.5, step=0.05)
        for i in range(n_layers)
    ]

    # Optimiser
    lr         = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Focal loss params
    alpha = trial.suggest_float("focal_alpha", 0.5, 0.9, step=0.05)
    gamma = trial.suggest_float("focal_gamma", 1.0, 3.0, step=0.5)

    # Build model
    model     = FraudClassifier(INPUT_DIM, layer_dims, dropout_rates).to(DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=SEARCH_EPOCHS)

    dataset = TensorDataset(T_Xtr, T_ytr)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         sampler=sampler, num_workers=0, pin_memory=False)

    best_val_f1 = 0.0
    for epoch in range(SEARCH_EPOCHS):
        train_one_epoch(model, loader, optimiser, criterion, DEVICE)
        scheduler.step()

        metrics = evaluate(model, T_Xva, y_val.astype(int), device=DEVICE)
        val_f1  = metrics["f1"]

        # Pruning — kill unpromising trials early
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        best_val_f1 = max(best_val_f1, val_f1)

    return best_val_f1


# Run optimisation
study = optuna.create_study(
    direction  = "maximize",
    sampler    = TPESampler(seed=42, n_startup_trials=10),
    pruner     = MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    study_name = "scfinshield_dnn"
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
print(f"\n  Best val F1   : {study.best_value:.4f}")
print(f"  Best params   : {best_params}")

# Visualise optimisation history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0])
optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[1])
plt.tight_layout()
plt.savefig("plots/optuna_history.png", dpi=150)
plt.close()
print("  Saved: plots/optuna_history.png")


# ─────────────────────────────────────────────────────────────
# SECTION 5 — FINAL MODEL TRAINING WITH BEST PARAMS
# ─────────────────────────────────────────────────────────────
print(f"\n[4/7] Training final model for {FINAL_EPOCHS} epochs...")

# Reconstruct best architecture
bp = best_params
n_layers_best   = bp["n_layers"]
layer_dims_best = sorted(
    [bp[f"dim_l{i}"] for i in range(n_layers_best)], reverse=True
)
dropout_best = [bp[f"dropout_l{i}"] for i in range(n_layers_best)]

model_final = FraudClassifier(INPUT_DIM, layer_dims_best, dropout_best).to(DEVICE)
criterion   = FocalLoss(alpha=bp.get("focal_alpha", 0.75), gamma=bp.get("focal_gamma", 2.0))
optimiser   = optim.AdamW(
    model_final.parameters(),
    lr=bp["lr"],
    weight_decay=bp.get("weight_decay", 1e-5)
)
# Warmup + Cosine Annealing schedule
scheduler = optim.lr_scheduler.OneCycleLR(
    optimiser,
    max_lr=bp["lr"],
    steps_per_epoch=len(X_train) // bp["batch_size"] + 1,
    epochs=FINAL_EPOCHS,
    pct_start=0.1,       # 10% warmup
    anneal_strategy="cos"
)

dataset = TensorDataset(T_Xtr, T_ytr)
loader  = DataLoader(dataset, batch_size=bp["batch_size"],
                     sampler=sampler, num_workers=0, pin_memory=False)

# Training loop with early stopping on val F1
best_val_f1   = 0.0
no_improve    = 0
train_losses  = []
val_f1_hist   = []
val_auc_hist  = []
best_state    = None

for epoch in range(1, FINAL_EPOCHS + 1):
    train_loss = train_one_epoch(model_final, loader, optimiser, criterion, DEVICE)
    scheduler.step()

    metrics   = evaluate(model_final, T_Xva, y_val.astype(int), device=DEVICE)
    val_f1    = metrics["f1"]
    val_auc   = metrics["auc_roc"]

    train_losses.append(train_loss)
    val_f1_hist.append(val_f1)
    val_auc_hist.append(val_auc)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state  = {k: v.cpu().clone() for k, v in model_final.state_dict().items()}
        no_improve  = 0
    else:
        no_improve += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{FINAL_EPOCHS} | "
              f"loss={train_loss:.4f} | val_f1={val_f1:.4f} | val_auc={val_auc:.4f}")

    if no_improve >= PATIENCE:
        print(f"  Early stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

# Restore best checkpoint
model_final.load_state_dict(best_state)
print(f"\n  Best val F1: {best_val_f1:.4f}")

# Training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(train_losses, color="#E74C3C"); axes[0].set_title("Train Loss (Focal)")
axes[1].plot(val_f1_hist,  color="#2ECC71"); axes[1].set_title("Val F1")
axes[2].plot(val_auc_hist, color="#3498DB"); axes[2].set_title("Val AUC-ROC")
for ax in axes:
    ax.grid(alpha=0.3); ax.set_xlabel("Epoch")
plt.tight_layout()
plt.savefig("plots/dnn_training_curves.png", dpi=150)
plt.close()
print("  Saved: plots/dnn_training_curves.png")


# ─────────────────────────────────────────────────────────────
# SECTION 6 — THRESHOLD CALIBRATION
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Calibrating classification threshold on validation set...")

opt_threshold, opt_val_f1 = find_optimal_threshold(model_final, T_Xva, y_val.astype(int))
print(f"  Optimal threshold : {opt_threshold:.2f}")
print(f"  Val F1 at optimal : {opt_val_f1:.4f}")

# Precision-Recall curve
model_final.eval()
with torch.no_grad():
    val_probs = torch.sigmoid(model_final(T_Xva)).cpu().numpy()
prec, rec, thresholds_pr = precision_recall_curve(y_val.astype(int), val_probs)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rec, prec, color="#9B59B6", lw=2, label=f"AP={average_precision_score(y_val.astype(int), val_probs):.3f}")
ax.axvline(
    rec[np.argmin(np.abs(thresholds_pr - opt_threshold))] if len(thresholds_pr) > 0 else 0.5,
    color="#E74C3C", ls="--", label=f"threshold={opt_threshold:.2f}"
)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve (Validation)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/precision_recall_curve.png", dpi=150)
plt.close()
print("  Saved: plots/precision_recall_curve.png")


# ─────────────────────────────────────────────────────────────
# SECTION 7 — TEST SET EVALUATION & SAVE
# ─────────────────────────────────────────────────────────────
print("\n[6/7] Evaluating on held-out test set...")

test_metrics = evaluate(
    model_final, T_Xte, y_test.astype(int),
    threshold=opt_threshold, device=DEVICE
)
print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │         TEST SET PERFORMANCE             │")
print(f"  ├─────────────────────────────────────────┤")
print(f"  │  AUC-ROC  : {test_metrics['auc_roc']:.4f}                  │")
print(f"  │  AUC-PR   : {test_metrics['auc_pr']:.4f}                  │")
print(f"  │  F1 Score : {test_metrics['f1']:.4f}                  │")
print(f"  │  Recall   : {test_metrics['recall']:.4f}                  │")
print(f"  │  Precision: {test_metrics['precision']:.4f}                  │")
print(f"  │  Threshold: {opt_threshold:.2f}                     │")
print(f"  └─────────────────────────────────────────┘")

print("\n  Classification Report:")
print(classification_report(y_test.astype(int), test_metrics["preds"],
                             target_names=["Legitimate", "Fraud"]))

# Confusion matrix
cm = confusion_matrix(y_test.astype(int), test_metrics["preds"])
fig, ax = plt.subplots(figsize=(5, 4))
sns_cm = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                     xticklabels=["Legitimate","Fraud"],
                     yticklabels=["Legitimate","Fraud"],
                     ax=ax) if False else None
# Manual heatmap
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_xticklabels(["Legitimate","Fraud"])
ax.set_yticks([0,1]); ax.set_yticklabels(["Legitimate","Fraud"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix — Test Set")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("plots/confusion_matrix_dnn.png", dpi=150)
plt.close()
print("  Saved: plots/confusion_matrix_dnn.png")


print("\n[7/7] Saving model artifacts...")

# Save full model (weights + architecture — safe for inference)
torch.save(model_final, "dnn/fraud_classifier.pt")
print("  ✓ dnn/fraud_classifier.pt")

# Save state dict separately (lighter, for fine-tuning)
torch.save(model_final.state_dict(), "dnn/fraud_classifier_state_dict.pt")
print("  ✓ dnn/fraud_classifier_state_dict.pt")

# Metadata
dnn_metadata = {
    "created_at":         datetime.datetime.utcnow().isoformat(),
    "model_type":         "FraudClassifier (FCN)",
    "input_dim":          INPUT_DIM,
    "layer_dims":         layer_dims_best,
    "dropout_rates":      dropout_best,
    "activation":         "GELU",
    "loss_function":      "FocalLoss",
    "focal_alpha":        float(bp.get("focal_alpha", 0.75)),
    "focal_gamma":        float(bp.get("focal_gamma", 2.0)),
    "optimal_threshold":  float(opt_threshold),
    "optuna_best_val_f1": float(study.best_value),
    "n_trials":           N_TRIALS,
    "final_epochs":       FINAL_EPOCHS,
    "test_auc_roc":       float(test_metrics["auc_roc"]),
    "test_auc_pr":        float(test_metrics["auc_pr"]),
    "test_f1":            float(test_metrics["f1"]),
    "test_recall":        float(test_metrics["recall"]),
    "test_precision":     float(test_metrics["precision"]),
    "best_params":        {k: (v if not isinstance(v, np.generic) else v.item())
                           for k, v in best_params.items()},
}
with open("dnn/metadata.json", "w") as f:
    json.dump(dnn_metadata, f, indent=2)
print("  ✓ dnn/metadata.json")

print("\n" + "=" * 60)
print("Notebook 02 COMPLETE — DNN Classifier saved.")
print(f"AUC-ROC  : {test_metrics['auc_roc']:.4f}")
print(f"F1 Score : {test_metrics['f1']:.4f}")
print(f"Recall   : {test_metrics['recall']:.4f}")
print(f"Threshold: {opt_threshold:.2f}")
print("=" * 60)