# ============================================================
# SCFinShield-AI | GraphSAGE + GAT GNN
# ============================================================
import os, json, datetime, warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    classification_report
)

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

os.makedirs("graphsage", exist_ok=True)
os.makedirs("gat",       exist_ok=True)
os.makedirs("plots",     exist_ok=True)

print("=" * 60)
print("SCFinShield-AI  |  Notebook 05: GraphSAGE + GAT")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# SECTION 1 — LOAD ELLIPTIC DATASET
# ─────────────────────────────────────────────────────────────
print("\n[1/8] Loading Elliptic Bitcoin Dataset...")

try:
    from torch_geometric.datasets import EllipticBitcoinDataset
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.loader import NeighborLoader

    dataset = EllipticBitcoinDataset(
        root="/kaggle/working/elliptic",
        transform=NormalizeFeatures()
    )
    data = dataset[0]

    print(f"  Nodes          : {data.num_nodes}")
    print(f"  Edges          : {data.num_edges}")
    print(f"  Node features  : {data.num_node_features}")
    print(f"  Classes        : {data.num_classes}")

    # Elliptic: 0=licit, 1=illicit, 2=unknown
    # Create binary masks excluding unknown nodes
    labelled_mask = data.y != 2
    fraud_mask    = (data.y == 1) & labelled_mask
    clean_mask    = (data.y == 0) & labelled_mask

    print(f"  Labelled nodes : {labelled_mask.sum().item()}")
    print(f"  Illicit (fraud): {fraud_mask.sum().item()}")
    print(f"  Licit (clean)  : {clean_mask.sum().item()}")
    print(f"  Fraud rate     : {fraud_mask.sum().item()/labelled_mask.sum().item():.4f}")

    # Remap labels: illicit→1, licit→0, unknown→-1
    y_binary = torch.where(data.y == 1, torch.ones_like(data.y),
               torch.where(data.y == 0, torch.zeros_like(data.y),
               torch.full_like(data.y, -1)))
    data.y = y_binary

    INPUT_DIM = data.num_node_features
    ELLIPTIC_AVAILABLE = True

except Exception as e:
    print(f"  Elliptic dataset not available: {e}")
    print("  Falling back to synthetic SCF graph...")
    ELLIPTIC_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# SECTION 1b — SYNTHETIC FALLBACK (if Elliptic not available)
# ─────────────────────────────────────────────────────────────
if not ELLIPTIC_AVAILABLE:
    from torch_geometric.data import Data

    print("  Generating synthetic supply chain fraud graph...")
    N_NODES    = 5000
    FRAUD_RATE = 0.05
    N_EDGES    = 25000
    INPUT_DIM  = 64

    np.random.seed(42)
    y_np     = (np.random.rand(N_NODES) < FRAUD_RATE).astype(int)
    # Fraudulent nodes have abnormal features
    feat_clean = np.random.randn(N_NODES, INPUT_DIM).astype(np.float32)
    fraud_idx  = np.where(y_np == 1)[0]
    # Add anomalous pattern to fraud node features
    feat_clean[fraud_idx] += np.random.randn(len(fraud_idx), INPUT_DIM) * 2.0
    feat_clean[fraud_idx, :5] += 5.0   # Strong signal in first 5 features

    # Generate edges — fraud nodes form denser clusters (carousel-like)
    edge_src, edge_dst = [], []
    # Random edges
    edge_src.extend(np.random.randint(0, N_NODES, N_EDGES).tolist())
    edge_dst.extend(np.random.randint(0, N_NODES, N_EDGES).tolist())
    # Fraud cluster edges (carousel rings)
    for i in range(len(fraud_idx) - 1):
        edge_src.append(int(fraud_idx[i]))
        edge_dst.append(int(fraud_idx[(i+1) % len(fraud_idx)]))
        edge_src.append(int(fraud_idx[(i+1) % len(fraud_idx)]))
        edge_dst.append(int(fraud_idx[i]))

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    x = torch.FloatTensor(feat_clean)
    y_binary = torch.LongTensor(y_np)

    data = Data(x=x, edge_index=edge_index, y=y_binary)
    print(f"  Synthetic graph: {N_NODES} nodes, {len(edge_src)} edges")
    labelled_mask = torch.ones(N_NODES, dtype=torch.bool)


# ─────────────────────────────────────────────────────────────
# SECTION 2 — TRAIN/VAL/TEST MASKS
# ─────────────────────────────────────────────────────────────
print("\n[2/8] Creating train/val/test masks...")

labelled_indices = labelled_mask.nonzero(as_tuple=True)[0].numpy()
from sklearn.model_selection import train_test_split as tts

y_lab = data.y[labelled_mask].numpy()
tr_idx, te_idx = tts(labelled_indices, test_size=0.2,
                     stratify=y_lab, random_state=42)
tr_idx, va_idx = tts(tr_idx, test_size=0.2,
                     stratify=data.y[tr_idx].numpy(), random_state=42)

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[tr_idx] = True
val_mask[va_idx]   = True
test_mask[te_idx]  = True

print(f"  Train: {train_mask.sum().item()}  (fraud={data.y[train_mask].float().mean():.4f})")
print(f"  Val  : {val_mask.sum().item()}    (fraud={data.y[val_mask].float().mean():.4f})")
print(f"  Test : {test_mask.sum().item()}   (fraud={data.y[test_mask].float().mean():.4f})")

data = data.to(DEVICE)
train_mask, val_mask, test_mask = train_mask.to(DEVICE), val_mask.to(DEVICE), test_mask.to(DEVICE)


# ─────────────────────────────────────────────────────────────
# SECTION 3 — MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────
print("\n[3/8] Defining GraphSAGE and GAT models...")

try:
    from torch_geometric.nn import SAGEConv, GATConv, BatchNorm
    PYG_AVAILABLE = True
except ImportError:
    print("  torch_geometric not available — using manual GNN implementation")
    PYG_AVAILABLE = False


if PYG_AVAILABLE:
    class GraphSAGEFraud(nn.Module):
        """
        GraphSAGE for fraud node classification.
        - Mean aggregation (robust to varying degree)
        - BatchNorm after each conv layer (stabilises training on heterogeneous graphs)
        - Skip connections between layers (residual learning)
        - Dropout for regularisation
        - L2-normalised output embeddings
        """
        def __init__(self, input_dim: int, hidden_dim: int = 128,
                     output_dim: int = 64, n_layers: int = 3,
                     dropout: float = 0.4):
            super().__init__()
            self.n_layers = n_layers
            self.dropout  = dropout

            self.convs = nn.ModuleList()
            self.bns   = nn.ModuleList()

            # Input layer
            self.convs.append(SAGEConv(input_dim, hidden_dim, aggr="mean"))
            self.bns.append(BatchNorm(hidden_dim))

            # Hidden layers
            for _ in range(n_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))
                self.bns.append(BatchNorm(hidden_dim))

            # Output embedding layer
            self.convs.append(SAGEConv(hidden_dim, output_dim, aggr="mean"))
            self.bns.append(BatchNorm(output_dim))

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(output_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, 1)
            )
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

        def forward(self, x, edge_index):
            for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
                x_new = conv(x, edge_index)
                x_new = bn(x_new)
                x_new = F.gelu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
                # Residual connection where dims match
                if x.shape == x_new.shape:
                    x = x + x_new
                else:
                    x = x_new
            # L2 normalise embeddings
            embeddings = F.normalize(x, p=2, dim=1)
            logits     = self.classifier(embeddings).squeeze(-1)
            return logits, embeddings


    class GATFraud(nn.Module):
        """
        Graph Attention Network — attention weights identify which
        neighbouring nodes most influence the fraud classification.
        The attention weights are interpretable signals for analysts.
        """
        def __init__(self, input_dim: int, hidden_dim: int = 64,
                     output_dim: int = 32, heads: int = 4,
                     dropout: float = 0.4):
            super().__init__()
            self.dropout = dropout

            self.conv1 = GATConv(input_dim,  hidden_dim,
                                 heads=heads, dropout=dropout, concat=True)
            self.bn1   = BatchNorm(hidden_dim * heads)

            self.conv2 = GATConv(hidden_dim * heads, hidden_dim,
                                 heads=heads, dropout=dropout, concat=True)
            self.bn2   = BatchNorm(hidden_dim * heads)

            self.conv3 = GATConv(hidden_dim * heads, output_dim,
                                 heads=1, dropout=dropout, concat=False)
            self.bn3   = BatchNorm(output_dim)

            self.classifier = nn.Linear(output_dim, 1)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

        def forward(self, x, edge_index):
            x = F.dropout(x, p=self.dropout * 0.5, training=self.training)

            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.conv3(x, edge_index)
            x = self.bn3(x)
            embeddings = F.normalize(x, p=2, dim=1)
            logits     = self.classifier(embeddings).squeeze(-1)
            return logits, embeddings

else:
    # Fallback: simple MLP if PyG not available
    class GraphSAGEFraud(nn.Module):
        def __init__(self, input_dim, hidden_dim=128, output_dim=64,
                     n_layers=3, dropout=0.4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim),
            )
            self.classifier = nn.Linear(output_dim, 1)
        def forward(self, x, edge_index=None):
            emb = F.normalize(self.net(x), p=2, dim=1)
            return self.classifier(emb).squeeze(-1), emb

    GATFraud = GraphSAGEFraud   # Use same architecture


# ─────────────────────────────────────────────────────────────
# SECTION 4 — TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────

# Compute class weights for loss weighting
y_train_np = data.y[train_mask].cpu().numpy()
n_clean    = (y_train_np == 0).sum()
n_fraud    = (y_train_np == 1).sum()
pos_weight = torch.tensor([n_clean / max(n_fraud, 1)], dtype=torch.float).to(DEVICE)
print(f"\n  Positive class weight: {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_epoch_gnn(model, data, mask, optimiser):
    model.train()
    optimiser.zero_grad()
    logits, _ = model(data.x, data.edge_index)
    loss = criterion(logits[mask].float(), data.y[mask].float())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimiser.step()
    return loss.item()


@torch.no_grad()
def evaluate_gnn(model, data, mask, threshold=0.5):
    model.eval()
    logits, embeddings = model(data.x, data.edge_index)
    probs  = torch.sigmoid(logits[mask]).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    preds  = (probs >= threshold).astype(int)

    valid = labels >= 0  # exclude unknown (-1)
    if valid.sum() == 0:
        return {"auc_roc": 0.5, "f1": 0.0, "probs": probs, "labels": labels}
    return {
        "auc_roc":    roc_auc_score(labels[valid], probs[valid]),
        "auc_pr":     average_precision_score(labels[valid], probs[valid]),
        "f1":         f1_score(labels[valid], preds[valid], zero_division=0),
        "recall":     (preds[valid][labels[valid]==1]).mean() if (labels[valid]==1).any() else 0.0,
        "probs":      probs,
        "labels":     labels,
        "embeddings": embeddings.cpu().numpy(),
    }


def find_threshold_gnn(model, data, mask):
    model.eval()
    with torch.no_grad():
        probs  = torch.sigmoid(model(data.x, data.edge_index)[0][mask]).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    valid  = labels >= 0
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.02):
        preds = (probs[valid] >= t).astype(int)
        f1    = f1_score(labels[valid], preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def train_model(model, model_name, epochs=150, lr=5e-4, weight_decay=5e-5, patience=20):
    """Full training loop with early stopping and LR scheduling."""
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=10, factor=0.5, verbose=False
    )

    best_auc   = 0.0
    best_state = None
    no_improve = 0
    train_losses, val_auc_hist = [], []

    for epoch in range(1, epochs + 1):
        loss = train_epoch_gnn(model, data, train_mask, opt)
        val_m = evaluate_gnn(model, data, val_mask)
        val_auc = val_m["auc_roc"]

        sched.step(val_auc)
        train_losses.append(loss)
        val_auc_hist.append(val_auc)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 25 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | loss={loss:.4f} | val_auc={val_auc:.4f}")

        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, train_losses, val_auc_hist, best_auc


# ─────────────────────────────────────────────────────────────
# SECTION 5 — TRAIN GRAPHSAGE
# ─────────────────────────────────────────────────────────────
print("\n[4/8] Training GraphSAGE...")

graphsage = GraphSAGEFraud(
    input_dim=INPUT_DIM, hidden_dim=128, output_dim=64,
    n_layers=3, dropout=0.4
).to(DEVICE)

print(f"  GraphSAGE parameters: {sum(p.numel() for p in graphsage.parameters()):,}")

graphsage, gs_losses, gs_auc_hist, gs_best_auc = train_model(
    graphsage, "GraphSAGE", epochs=200, lr=5e-4, weight_decay=5e-5, patience=25
)

gs_threshold, gs_val_f1 = find_threshold_gnn(graphsage, data, val_mask)
gs_test = evaluate_gnn(graphsage, data, test_mask, threshold=gs_threshold)

print(f"\n  GraphSAGE Test Results:")
print(f"    AUC-ROC : {gs_test['auc_roc']:.4f}")
print(f"    AUC-PR  : {gs_test['auc_pr']:.4f}")
print(f"    F1      : {gs_test['f1']:.4f}")
print(f"    Recall  : {gs_test['recall']:.4f}")


# ─────────────────────────────────────────────────────────────
# SECTION 6 — TRAIN GAT
# ─────────────────────────────────────────────────────────────
print("\n[5/8] Training GAT...")

gat = GATFraud(
    input_dim=INPUT_DIM, hidden_dim=64, output_dim=32,
    heads=4, dropout=0.4
).to(DEVICE)

print(f"  GAT parameters: {sum(p.numel() for p in gat.parameters()):,}")

gat, gat_losses, gat_auc_hist, gat_best_auc = train_model(
    gat, "GAT", epochs=200, lr=3e-4, weight_decay=1e-4, patience=25
)

gat_threshold, gat_val_f1 = find_threshold_gnn(gat, data, val_mask)
gat_test = evaluate_gnn(gat, data, test_mask, threshold=gat_threshold)

print(f"\n  GAT Test Results:")
print(f"    AUC-ROC : {gat_test['auc_roc']:.4f}")
print(f"    AUC-PR  : {gat_test['auc_pr']:.4f}")
print(f"    F1      : {gat_test['f1']:.4f}")
print(f"    Recall  : {gat_test['recall']:.4f}")


# ─────────────────────────────────────────────────────────────
# SECTION 7 — TRAINING CURVES & NODE EMBEDDING VISUALISATION
# ─────────────────────────────────────────────────────────────
print("\n[6/8] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0,0].plot(gs_losses,   color="#E74C3C"); axes[0,0].set_title("GraphSAGE Train Loss")
axes[0,1].plot(gs_auc_hist, color="#3498DB"); axes[0,1].set_title("GraphSAGE Val AUC")
axes[1,0].plot(gat_losses,  color="#E74C3C"); axes[1,0].set_title("GAT Train Loss")
axes[1,1].plot(gat_auc_hist,color="#3498DB"); axes[1,1].set_title("GAT Val AUC")
for ax in axes.flatten():
    ax.grid(alpha=0.3); ax.set_xlabel("Epoch")
plt.tight_layout()
plt.savefig("plots/gnn_training_curves.png", dpi=150)
plt.close()

# t-SNE on GraphSAGE node embeddings (test set)
try:
    from sklearn.manifold import TSNE
    embs   = gs_test["embeddings"]
    labels = gs_test["labels"]
    valid  = labels >= 0
    if valid.sum() > 100:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        embs_2d = tsne.fit_transform(embs[valid][:2000])
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            embs_2d[:, 0], embs_2d[:, 1],
            c=labels[valid][:2000].astype(int),
            cmap="RdYlGn_r", alpha=0.5, s=8
        )
        plt.colorbar(scatter, ax=ax, label="Fraud (1) / Legit (0)")
        ax.set_title("GraphSAGE Node Embeddings (t-SNE)")
        plt.tight_layout()
        plt.savefig("plots/gnn_tsne_embeddings.png", dpi=150)
        plt.close()
        print("  Saved: plots/gnn_tsne_embeddings.png")
except Exception as e:
    print(f"  t-SNE skipped: {e}")


# ─────────────────────────────────────────────────────────────
# SECTION 8 — SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("\n[7/8] Saving model artifacts...")

# GraphSAGE
torch.save(graphsage, "graphsage/graphsage_model.pt")
print("  ✓ graphsage/graphsage_model.pt")

# Pre-compute all node embeddings for fast lookup at inference
graphsage.eval()
with torch.no_grad():
    _, all_embeddings = graphsage(data.x, data.edge_index)
torch.save(all_embeddings.cpu(), "graphsage/node_embeddings.pt")
print("  ✓ graphsage/node_embeddings.pt")

# GAT
torch.save(gat, "gat/gat_model.pt")
print("  ✓ gat/gat_model.pt")

# Metadata (GraphSAGE)
metadata = {
    "created_at":          datetime.datetime.utcnow().isoformat(),
    "model_type":          "GraphSAGE + GAT",
    "dataset":             "Elliptic Bitcoin" if ELLIPTIC_AVAILABLE else "Synthetic SCF",
    "input_dim":           INPUT_DIM,
    "graphsage": {
        "hidden_dim": 128, "output_dim": 64, "n_layers": 3, "dropout": 0.4,
        "optimal_threshold": float(gs_threshold),
        "test_auc_roc": float(gs_test["auc_roc"]),
        "test_auc_pr":  float(gs_test["auc_pr"]),
        "test_f1":      float(gs_test["f1"]),
        "test_recall":  float(gs_test["recall"]),
    },
    "gat": {
        "hidden_dim": 64, "output_dim": 32, "heads": 4, "dropout": 0.4,
        "optimal_threshold": float(gat_threshold),
        "test_auc_roc": float(gat_test["auc_roc"]),
        "test_auc_pr":  float(gat_test["auc_pr"]),
        "test_f1":      float(gat_test["f1"]),
        "test_recall":  float(gat_test["recall"]),
    },
}
with open("graphsage/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("  ✓ graphsage/metadata.json")

print("\n" + "=" * 60)
print("Notebook 05 COMPLETE — GraphSAGE + GAT saved.")
print(f"\nGraphSAGE  AUC-ROC: {gs_test['auc_roc']:.4f}  F1: {gs_test['f1']:.4f}")
print(f"GAT        AUC-ROC: {gat_test['auc_roc']:.4f}  F1: {gat_test['f1']:.4f}")
print("=" * 60)
