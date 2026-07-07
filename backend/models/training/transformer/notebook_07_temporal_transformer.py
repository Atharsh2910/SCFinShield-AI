# ============================================================
# SCFinShield-AI | Notebook 07: Temporal Transformer
# ============================================================
# Purpose  : Detect sequencing anomalies in invoice streams.
#            Each supplier has a historical sequence of invoices.
#            The Transformer learns the "language" of legitimate
#            invoice sequences. Anomalous sequences (end-of-month
#            spikes, non-sequential invoice numbers, uncharacteristic
#            billing cycles, velocity bursts) produce high
#            reconstruction loss → anomaly score.
#
# Architecture : BERT-style masked invoice sequence model
#                (Masked Invoice Modelling = MIM)
#                + supervised classification head fine-tuned
#                  on labelled sequences
#
# Inputs   : training/X_train_pca.npy, X_val_pca.npy, X_test_pca.npy
#            training/y_train_bal.npy, y_val.npy, y_test.npy
#            preprocessing/metadata.json
# Outputs  : transformer/temporal_transformer.pt
#            transformer/metadata.json
# ============================================================

# ── KAGGLE INSTALL ────────────────────────────────────────
# torch and sklearn already available on Kaggle — no extra install

import os, json, pickle, datetime, warnings, math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    precision_score, recall_score, classification_report
)

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

os.makedirs("transformer", exist_ok=True)
os.makedirs("plots",       exist_ok=True)

print("=" * 60)
print("SCFinShield-AI  |  Notebook 07: Temporal Transformer")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# SECTION 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/8] Loading preprocessed arrays...")

X_train = np.load("training/X_train_pca.npy").astype(np.float32)
X_val   = np.load("training/X_val_pca.npy").astype(np.float32)
X_test  = np.load("training/X_test_pca.npy").astype(np.float32)
y_train = np.load("training/y_train_bal.npy").astype(int)
y_val   = np.load("training/y_val.npy").astype(int)
y_test  = np.load("training/y_test.npy").astype(int)

INPUT_DIM = X_train.shape[1]  # PCA-reduced feature dimension

print(f"  Input dim (PCA features): {INPUT_DIM}")
print(f"  Train: {X_train.shape}  fraud={y_train.mean():.4f}")
print(f"  Val  : {X_val.shape}    fraud={y_val.mean():.4f}")
print(f"  Test : {X_test.shape}   fraud={y_test.mean():.4f}")


# ─────────────────────────────────────────────────────────────
# SECTION 2 — SEQUENCE CONSTRUCTION
# ─────────────────────────────────────────────────────────────
print("\n[2/8] Constructing invoice sequences per supplier...")

# Strategy: group consecutive invoices into fixed-length windows.
# Each "sequence" = SEQ_LEN consecutive invoices treated as coming
# from the same supplier entity. The label of the sequence is the
# label of the last invoice in the window (the one being evaluated).
#
# In production: sequences are built from actual supplier invoice
# history stored in Supabase / Neo4j. Here we simulate sequences
# by creating sliding windows over the sorted dataset.

SEQ_LEN  = 8    # number of invoices in each context window
STRIDE   = 1    # sliding window stride

def build_sequences(X, y, seq_len=SEQ_LEN, stride=STRIDE):
    """
    Create (sequence, label) pairs using a sliding window.
    sequence shape: (seq_len, input_dim)
    label: label of the last invoice in the window (the target)

    For fraud detection, a sequence is labelled fraud if ANY
    invoice in the last 3 positions is fraudulent (temporal leakage
    of fraud signals into the context).
    """
    sequences, labels = [], []
    n = len(X)
    for start in range(0, n - seq_len, stride):
        end = start + seq_len
        seq = X[start:end]        # (seq_len, input_dim)
        # Label: fraud if the last invoice or any of last 3 are fraud
        window_labels = y[max(start, end-3):end]
        label = int(window_labels.max())
        sequences.append(seq)
        labels.append(label)

    sequences = np.array(sequences, dtype=np.float32)
    labels    = np.array(labels, dtype=np.int32)
    return sequences, labels


S_train, ys_train = build_sequences(X_train, y_train, SEQ_LEN, stride=2)
S_val,   ys_val   = build_sequences(X_val,   y_val,   SEQ_LEN, stride=1)
S_test,  ys_test  = build_sequences(X_test,  y_test,  SEQ_LEN, stride=1)

print(f"  Train sequences: {S_train.shape}  fraud={ys_train.mean():.4f}")
print(f"  Val   sequences: {S_val.shape}    fraud={ys_val.mean():.4f}")
print(f"  Test  sequences: {S_test.shape}   fraud={ys_test.mean():.4f}")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — DATASET
# ─────────────────────────────────────────────────────────────

class InvoiceSequenceDataset(Dataset):
    """
    Dataset of invoice sequences.
    Returns: (sequence_tensor, label, mask_indices) for MIM pre-training,
             or (sequence_tensor, label) for supervised fine-tuning.
    """
    def __init__(self, sequences: np.ndarray, labels: np.ndarray,
                 mask_prob: float = 0.15, mode: str = "supervised"):
        """
        mode: 'pretrain' for masked invoice modelling
              'supervised' for classification fine-tuning
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels    = torch.LongTensor(labels)
        self.mask_prob = mask_prob
        self.mode      = mode
        self.seq_len   = sequences.shape[1]
        self.feat_dim  = sequences.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq   = self.sequences[idx].clone()    # (seq_len, feat_dim)
        label = self.labels[idx]

        if self.mode == "pretrain":
            # Randomly mask 15% of invoice positions (BERT-style)
            mask = torch.zeros(self.seq_len, dtype=torch.bool)
            for pos in range(self.seq_len):
                if torch.rand(1).item() < self.mask_prob:
                    mask[pos] = True
            original = seq.clone()
            # Replace masked positions with zeros (MASK token)
            seq[mask] = 0.0
            return seq, original, mask, label
        else:
            return seq, label


# ─────────────────────────────────────────────────────────────
# SECTION 4 — MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────
print("\n[3/8] Defining Temporal Transformer architecture...")


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding — injects temporal position
    information so the Transformer knows invoice ordering.
    Crucial for detecting sequencing anomalies (e.g. invoice #500
    arriving before invoice #498 from the same supplier).
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for invoice sequence modelling.

    Architecture:
    1. Input projection: invoice features → d_model
    2. Positional encoding: inject position/temporal information
    3. Transformer encoder: n_layers of multi-head self-attention
       - Each layer: MHA → Add&Norm → FFN → Add&Norm
       - Pre-LayerNorm (more stable training than Post-LN)
    4. [CLS] token pooling: aggregate sequence representation
    5. Dual heads:
       a. Reconstruction head (pre-training): reconstruct masked invoices
       b. Classification head (fine-tuning): fraud probability

    The pre-training phase on clean sequences teaches the model
    the "grammar" of legitimate invoice streams.
    The fine-tuning phase on labelled data sharpens fraud detection.
    """
    def __init__(
        self,
        input_dim:   int,
        d_model:     int   = 128,
        n_heads:     int   = 4,
        n_layers:    int   = 3,
        d_ff:        int   = 256,
        dropout:     float = 0.2,
        max_seq_len: int   = 64,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model   = d_model
        self.input_dim = input_dim

        # ── Input projection ──────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Learnable [CLS] token ─────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Positional encoding ───────────────────────────
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        # ── Transformer encoder (Pre-LayerNorm for stability) ─
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,        # (batch, seq, dim)
            norm_first      = True,        # Pre-LayerNorm
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_layers,
            norm       = encoder_norm,
        )

        # ── Reconstruction head (pre-training) ────────────
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, input_dim),
        )

        # ── Classification head (fine-tuning) ─────────────
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # ── Anomaly score head (reconstruction-based) ─────
        self.anomaly_proj = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:       torch.Tensor,          # (batch, seq_len, input_dim)
        src_key_padding_mask: torch.Tensor = None,  # (batch, seq_len+1)
    ) -> dict:
        B, T, _ = x.shape

        # Project features to model dimension
        x_proj = self.input_proj(x)    # (B, T, d_model)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        x_with_cls = torch.cat([cls_tokens, x_proj], dim=1)  # (B, T+1, d_model)

        # Add positional encoding
        x_enc = self.pos_enc(x_with_cls)   # (B, T+1, d_model)

        # Transformer encoder
        encoded = self.encoder(
            x_enc,
            src_key_padding_mask=src_key_padding_mask
        )   # (B, T+1, d_model)

        # [CLS] token representation (index 0)
        cls_repr = encoded[:, 0, :]         # (B, d_model)
        # Invoice token representations (index 1:)
        inv_repr = encoded[:, 1:, :]        # (B, T, d_model)

        # Reconstruction output (for all positions)
        reconstructed = self.reconstruction_head(inv_repr)   # (B, T, input_dim)

        # Classification logit (from [CLS])
        class_logit = self.classification_head(cls_repr).squeeze(-1)  # (B,)

        return {
            "cls_repr":      cls_repr,
            "inv_repr":      inv_repr,
            "reconstructed": reconstructed,
            "class_logit":   class_logit,
        }

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Anomaly score = mean reconstruction error across the sequence.
        High error → sequence deviates from learned normal patterns.
        Combined with classification logit for ensemble.
        """
        out  = self.forward(x)
        # Reconstruction MSE per invoice position
        recon_error = F.mse_loss(out["reconstructed"], x, reduction="none")
        # Mean over features and positions
        anomaly_score = recon_error.mean(dim=(1, 2))   # (B,)
        return anomaly_score

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Combined fraud probability: classification + reconstruction-based signal."""
        out = self.forward(x)
        class_prob = torch.sigmoid(out["class_logit"])
        # Normalise reconstruction error to [0,1] via sigmoid
        recon_err  = F.mse_loss(out["reconstructed"], x, reduction="none").mean(dim=(1,2))
        recon_prob = torch.sigmoid((recon_err - recon_err.mean()) / (recon_err.std() + 1e-8))
        # Weighted combination
        return 0.7 * class_prob + 0.3 * recon_prob


# ─────────────────────────────────────────────────────────────
# SECTION 5 — PHASE 1: MASKED INVOICE MODELLING (PRE-TRAINING)
# ─────────────────────────────────────────────────────────────
print("\n[4/8] Phase 1: Masked Invoice Modelling pre-training...")

D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 3
D_FF        = 256
DROPOUT     = 0.2
PRETRAIN_EPOCHS = 30
PRETRAIN_LR     = 5e-4
BATCH_SIZE      = 256

model = TemporalTransformer(
    input_dim   = INPUT_DIM,
    d_model     = D_MODEL,
    n_heads     = N_HEADS,
    n_layers    = N_LAYERS,
    d_ff        = D_FF,
    dropout     = DROPOUT,
    max_seq_len = SEQ_LEN + 2,
).to(DEVICE)

print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Pre-train on CLEAN sequences only (teach normal invoice patterns)
S_train_clean = S_train[ys_train == 0]
pretrain_ds   = InvoiceSequenceDataset(S_train_clean,
                                        np.zeros(len(S_train_clean), dtype=np.int32),
                                        mask_prob=0.15, mode="pretrain")
pretrain_loader = DataLoader(pretrain_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)

pretrain_opt   = optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-5)
pretrain_sched = optim.lr_scheduler.CosineAnnealingLR(pretrain_opt, T_max=PRETRAIN_EPOCHS)

recon_criterion = nn.MSELoss()
pretrain_losses = []

print(f"  Pre-training on {len(S_train_clean)} clean sequences for {PRETRAIN_EPOCHS} epochs...")
for epoch in range(1, PRETRAIN_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in pretrain_loader:
        seq_masked, seq_orig, mask, _ = batch
        seq_masked = seq_masked.to(DEVICE)
        seq_orig   = seq_orig.to(DEVICE)
        mask       = mask.to(DEVICE)      # (B, seq_len) bool

        pretrain_opt.zero_grad()
        out  = model(seq_masked)
        recon = out["reconstructed"]      # (B, seq_len, input_dim)

        # Only compute reconstruction loss on MASKED positions
        # to prevent the model from just copying unmasked inputs
        if mask.any():
            # mask: (B, seq_len) → expand to (B, seq_len, input_dim)
            mask_expanded = mask.unsqueeze(-1).expand_as(recon)
            loss = F.mse_loss(
                recon[mask_expanded],
                seq_orig[mask_expanded]
            )
        else:
            loss = recon_criterion(recon, seq_orig)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        pretrain_opt.step()
        epoch_loss += loss.item()

    pretrain_sched.step()
    avg_loss = epoch_loss / len(pretrain_loader)
    pretrain_losses.append(avg_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{PRETRAIN_EPOCHS} | recon_loss={avg_loss:.5f}")

print("  Pre-training complete.")


# ─────────────────────────────────────────────────────────────
# SECTION 6 — PHASE 2: SUPERVISED FINE-TUNING
# ─────────────────────────────────────────────────────────────
print("\n[5/8] Phase 2: Supervised fine-tuning for fraud classification...")

FINETUNE_EPOCHS = 60
FINETUNE_LR     = 1e-4     # Lower LR to preserve pre-trained weights
PATIENCE        = 15

# Class weights for imbalanced fine-tuning data
n_clean_ft = (ys_train == 0).sum()
n_fraud_ft = (ys_train == 1).sum()
pos_weight = torch.tensor([n_clean_ft / max(n_fraud_ft, 1)],
                           dtype=torch.float).to(DEVICE)
print(f"  Positive class weight: {pos_weight.item():.2f}")

# Focal loss for fine-tuning (handles hard negatives)
class SequenceFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t  = torch.where(targets == 1, prob, 1 - prob)
        alpha_t = torch.where(targets == 1,
                              torch.tensor(self.alpha, device=logits.device),
                              torch.tensor(1 - self.alpha, device=logits.device))
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()

train_ds = InvoiceSequenceDataset(S_train, ys_train, mode="supervised")
val_ds   = InvoiceSequenceDataset(S_val,   ys_val,   mode="supervised")
test_ds  = InvoiceSequenceDataset(S_test,  ys_test,  mode="supervised")

# Weighted sampler for fine-tuning
sample_weights = np.where(ys_train == 1,
                           float(pos_weight.item()), 1.0)
from torch.utils.data import WeightedRandomSampler
ft_sampler = WeightedRandomSampler(
    torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                           sampler=ft_sampler, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0)

# Freeze encoder for first few epochs then unfreeze (gradual unfreezing)
for param in model.encoder.parameters():
    param.requires_grad = False

ft_criterion = SequenceFocalLoss(alpha=0.75, gamma=2.0)
ft_opt       = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=FINETUNE_LR, weight_decay=1e-5
)
ft_sched = optim.lr_scheduler.OneCycleLR(
    ft_opt, max_lr=FINETUNE_LR,
    steps_per_epoch=len(train_loader),
    epochs=FINETUNE_EPOCHS,
    pct_start=0.1
)

best_val_auc  = 0.0
best_state    = None
no_improve    = 0
ft_losses     = []
val_auc_hist  = []

UNFREEZE_EPOCH = 10   # Unfreeze encoder after 10 warm-up epochs

for epoch in range(1, FINETUNE_EPOCHS + 1):
    # Gradual unfreezing
    if epoch == UNFREEZE_EPOCH:
        print(f"  Unfreezing encoder at epoch {epoch}")
        for param in model.encoder.parameters():
            param.requires_grad = True
        # Re-create optimiser with full params and lower LR for encoder
        ft_opt = optim.AdamW([
            {"params": model.encoder.parameters(),         "lr": FINETUNE_LR * 0.1},
            {"params": model.classification_head.parameters(), "lr": FINETUNE_LR},
            {"params": model.input_proj.parameters(),     "lr": FINETUNE_LR * 0.3},
            {"params": model.cls_token,                   "lr": FINETUNE_LR},
        ], weight_decay=1e-5)
        ft_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            ft_opt, T_0=20, T_mult=2
        )

    # Training step
    model.train()
    epoch_loss = 0.0
    for seqs, labels in train_loader:
        seqs, labels = seqs.to(DEVICE), labels.float().to(DEVICE)
        ft_opt.zero_grad()
        out    = model(seqs)
        loss   = ft_criterion(out["class_logit"], labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        ft_opt.step()
        if epoch < UNFREEZE_EPOCH:
            ft_sched.step()
        epoch_loss += loss.item()

    if epoch >= UNFREEZE_EPOCH:
        ft_sched.step(epoch - UNFREEZE_EPOCH)

    # Validation
    model.eval()
    val_probs, val_labels_list = [], []
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs = seqs.to(DEVICE)
            out  = model(seqs)
            prob = torch.sigmoid(out["class_logit"]).cpu().numpy()
            val_probs.extend(prob.tolist())
            val_labels_list.extend(labels.numpy().tolist())

    val_auc = roc_auc_score(val_labels_list, val_probs)
    ft_losses.append(epoch_loss / len(train_loader))
    val_auc_hist.append(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve   = 0
    else:
        no_improve += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{FINETUNE_EPOCHS} | "
              f"loss={epoch_loss/len(train_loader):.4f} | "
              f"val_auc={val_auc:.4f}")

    if no_improve >= PATIENCE:
        print(f"  Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)
print(f"\n  Best val AUC: {best_val_auc:.4f}")


# ─────────────────────────────────────────────────────────────
# SECTION 7 — THRESHOLD CALIBRATION
# ─────────────────────────────────────────────────────────────
print("\n[6/8] Calibrating classification threshold...")

model.eval()
val_probs_arr = []
val_labs_arr  = []
with torch.no_grad():
    for seqs, labels in val_loader:
        seqs = seqs.to(DEVICE)
        prob = model.predict_proba(seqs).cpu().numpy()
        val_probs_arr.extend(prob.tolist())
        val_labs_arr.extend(labels.numpy().tolist())

val_probs_arr = np.array(val_probs_arr)
val_labs_arr  = np.array(val_labs_arr).astype(int)

best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.05, 0.95, 0.01):
    preds = (val_probs_arr >= t).astype(int)
    f1    = f1_score(val_labs_arr, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"  Optimal threshold: {best_t:.2f}")
print(f"  Val F1           : {best_f1:.4f}")


# ─────────────────────────────────────────────────────────────
# SECTION 8 — RECONSTRUCTION-BASED ANOMALY ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n[7/8] Analysing reconstruction-based anomaly scores...")

model.eval()
test_recon_errors = []
test_class_probs  = []
test_labels_list  = []

with torch.no_grad():
    for seqs, labels in test_loader:
        seqs = seqs.to(DEVICE)
        out  = model(seqs)
        # Reconstruction error per sequence
        recon_err = F.mse_loss(out["reconstructed"], seqs, reduction="none").mean(dim=(1,2))
        class_p   = torch.sigmoid(out["class_logit"])
        test_recon_errors.extend(recon_err.cpu().numpy().tolist())
        test_class_probs.extend(class_p.cpu().numpy().tolist())
        test_labels_list.extend(labels.numpy().tolist())

test_recon_errors = np.array(test_recon_errors)
test_class_probs  = np.array(test_class_probs)
test_labels_arr   = np.array(test_labels_list).astype(int)

# Combined score
test_recon_norm = (test_recon_errors - test_recon_errors.mean()) / (test_recon_errors.std() + 1e-8)
test_recon_prob = 1 / (1 + np.exp(-test_recon_norm))
test_combined   = 0.7 * test_class_probs + 0.3 * test_recon_prob
test_preds      = (test_combined >= best_t).astype(int)

# Metrics
t_auc_roc  = roc_auc_score(test_labels_arr, test_combined)
t_auc_pr   = average_precision_score(test_labels_arr, test_combined)
t_f1       = f1_score(test_labels_arr, test_preds, zero_division=0)
t_recall   = recall_score(test_labels_arr, test_preds, zero_division=0)
t_precision= precision_score(test_labels_arr, test_preds, zero_division=0)

print(f"\n  ┌──────────────────────────────────────────────┐")
print(f"  │      TEMPORAL TRANSFORMER TEST PERFORMANCE    │")
print(f"  ├──────────────────────────────────────────────┤")
print(f"  │  AUC-ROC    : {t_auc_roc:.4f}                     │")
print(f"  │  AUC-PR     : {t_auc_pr:.4f}                     │")
print(f"  │  F1 Score   : {t_f1:.4f}                     │")
print(f"  │  Recall     : {t_recall:.4f}                     │")
print(f"  │  Precision  : {t_precision:.4f}                     │")
print(f"  │  Threshold  : {best_t:.2f}                        │")
print(f"  └──────────────────────────────────────────────┘")

print("\n  Classification Report:")
print(classification_report(test_labels_arr, test_preds,
                             target_names=["Legitimate", "Fraud"]))

# Reconstruction error distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(test_recon_errors[test_labels_arr==0], bins=50, alpha=0.6,
             color="#2ECC71", label="Legitimate", density=True)
axes[0].hist(test_recon_errors[test_labels_arr==1], bins=50, alpha=0.6,
             color="#E74C3C", label="Fraud",      density=True)
axes[0].set_xlabel("Reconstruction Error (MSE)")
axes[0].set_ylabel("Density")
axes[0].set_title("Reconstruction Error Distribution (Test)")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].hist(test_combined[test_labels_arr==0], bins=50, alpha=0.6,
             color="#2ECC71", label="Legitimate", density=True)
axes[1].hist(test_combined[test_labels_arr==1], bins=50, alpha=0.6,
             color="#E74C3C", label="Fraud",      density=True)
axes[1].axvline(best_t, color="#F39C12", ls="--", lw=2,
                label=f"threshold={best_t:.2f}")
axes[1].set_xlabel("Combined Fraud Score")
axes[1].set_ylabel("Density")
axes[1].set_title("Combined Score Distribution (Test)")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/transformer_score_distributions.png", dpi=150)
plt.close()
print("  Saved: plots/transformer_score_distributions.png")

# Training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(pretrain_losses, color="#9B59B6"); axes[0].set_title("Pre-train Recon Loss")
axes[1].plot(ft_losses,       color="#E74C3C"); axes[1].set_title("Fine-tune Loss")
axes[2].plot(val_auc_hist,    color="#3498DB"); axes[2].set_title("Val AUC-ROC")
for ax in axes:
    ax.grid(alpha=0.3); ax.set_xlabel("Epoch")
plt.tight_layout()
plt.savefig("plots/transformer_training_curves.png", dpi=150)
plt.close()
print("  Saved: plots/transformer_training_curves.png")

# Attention pattern analysis (sample one batch)
print("\n  Analysing attention patterns on sample sequences...")
model.eval()
sample_seqs = torch.FloatTensor(S_test[:8]).to(DEVICE)
with torch.no_grad():
    # Access attention weights from first encoder layer
    model.encoder.layers[0].self_attn.batch_first = True
    # Forward pass to get encoded sequence
    x_proj     = model.input_proj(sample_seqs)
    cls_tokens = model.cls_token.expand(8, -1, -1)
    x_with_cls = torch.cat([cls_tokens, x_proj], dim=1)
    x_enc      = model.pos_enc(x_with_cls)
    # Get attention weights (need_weights=True)
    _, attn_weights = model.encoder.layers[0].self_attn(
        x_enc, x_enc, x_enc, need_weights=True, average_attn_weights=True
    )

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    im = axes[i].imshow(
        attn_weights[i].cpu().numpy(),
        cmap="Blues", aspect="auto"
    )
    axes[i].set_title(f"Sample {i+1} — {'FRAUD' if ys_test[i]==1 else 'LEGIT'}")
    axes[i].set_xlabel("Key position"); axes[i].set_ylabel("Query position")
plt.suptitle("Self-Attention Patterns (Layer 1)")
plt.tight_layout()
plt.savefig("plots/attention_patterns.png", dpi=150)
plt.close()
print("  Saved: plots/attention_patterns.png")


# ─────────────────────────────────────────────────────────────
# SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("\n[8/8] Saving model artifacts...")

torch.save(model, "transformer/temporal_transformer.pt")
print("  ✓ transformer/temporal_transformer.pt")

torch.save(model.state_dict(), "transformer/temporal_transformer_state_dict.pt")
print("  ✓ transformer/temporal_transformer_state_dict.pt")

metadata = {
    "created_at":            datetime.datetime.utcnow().isoformat(),
    "model_type":            "TemporalTransformer (BERT-style MIM + Supervised)",
    "input_dim":             INPUT_DIM,
    "d_model":               D_MODEL,
    "n_heads":               N_HEADS,
    "n_layers":              N_LAYERS,
    "d_ff":                  D_FF,
    "dropout":               DROPOUT,
    "seq_len":               SEQ_LEN,
    "pretrain_strategy":     "Masked Invoice Modelling (clean sequences only)",
    "finetune_strategy":     "Supervised focal loss + gradual unfreezing",
    "optimal_threshold":     float(best_t),
    "score_formula":         "0.7 * class_prob + 0.3 * sigmoid(normalised_recon_error)",
    "test_auc_roc":          float(t_auc_roc),
    "test_auc_pr":           float(t_auc_pr),
    "test_f1":               float(t_f1),
    "test_recall":           float(t_recall),
    "test_precision":        float(t_precision),
    "pretrain_epochs":       PRETRAIN_EPOCHS,
    "finetune_epochs":       FINETUNE_EPOCHS,
    "unfreeze_epoch":        UNFREEZE_EPOCH,
    "val_auc_roc":           float(best_val_auc),
}
with open("transformer/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("  ✓ transformer/metadata.json")

print("\n" + "=" * 60)
print("Notebook 07 COMPLETE — Temporal Transformer saved.")
print(f"AUC-ROC  : {t_auc_roc:.4f}")
print(f"AUC-PR   : {t_auc_pr:.4f}")
print(f"F1       : {t_f1:.4f}")
print(f"Recall   : {t_recall:.4f}")
print(f"Threshold: {best_t:.2f}")
print("=" * 60)
