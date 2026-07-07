# ML Model Training Notebooks

These scripts are designed to be run as **Kaggle notebooks** (offline training).
After training, copy the output `.pkl` / `.pt` files into
`backend/services/ml/model_registry/<subdirectory>/`.

---

## Directory Structure & Model Registry Mapping

| Notebook | Subdirectory | Output → `model_registry/` path |
|---|---|---|
| `notebook_01_preprocessing_pipeline.py` | `preprocessing/` | `model_registry/preprocessing/scaler.pkl`, `pca.pkl`, `label_encoders.pkl`, `feature_columns.pkl` |
| `notebook_02_dnn_classifier_bayesian.py` | `dnn/` | `model_registry/dnn/fraud_classifier.pt` |
| `notebook_03_isolation_forest.py` | `anomaly/` | `model_registry/anomaly/isolation_forest.pkl` |
| `notebook_04_siamese_network.py` | `siamese/` | `model_registry/siamese/siamese_network.pt` |
| `notebook_05_graphsage_gnn.py` | `graphsage/` | `model_registry/graphsage/graphsage_model.pt` |
| `notebook_06_xgboost_ensemble.py` | `ensemble/` | `model_registry/ensemble/xgboost_ensemble.pkl`, `shap_explainer.pkl` |
| `notebook_07_temporal_transformer.py` | `transformer/` | `model_registry/transformer/temporal_transformer.pt` |
| `notebook_08_model_registry_validation.py` | `validation/` | `model_registry/model_card.json` |

---

## Training Order

Run notebooks in this exact order to avoid dependency issues:

1. **Notebook 01 — Preprocessing Pipeline** (`preprocessing/`)
   - Dataset: DataCo Smart Supply Chain (Kaggle)
   - Applies SMOTE, fits RobustScaler and PCA
   - Must be run first — all other notebooks consume its output arrays

2. **Notebook 02 — DNN Classifier + Bayesian Optimisation** (`dnn/`)
   - Inputs: `training/X_train_pca.npy`, `training/y_train_bal.npy`, etc.
   - Uses Optuna for hyperparameter search (50 trials)
   - Outputs: `fraud_classifier.pt`

3. **Notebook 03 — Isolation Forest** (`anomaly/`)
   - Unsupervised anomaly detector trained on clean invoices only
   - Outputs: `isolation_forest.pkl`

4. **Notebook 04 — Siamese Network** (`siamese/`)
   - Near-duplicate invoice detection via contrastive learning
   - Outputs: `siamese_network.pt`

5. **Notebook 05 — GraphSAGE GNN** 
   - Graph-based fraud detection on the Elliptic or AMLSim dataset
   - Outputs: `graphsage_model.pt`, `node_embeddings.pt`

6. **Notebook 06 — XGBoost Ensemble** (`ensemble/`)
   - Meta-learner stacking outputs of all prior models
   - Includes SHAP explainability
   - Outputs: `xgboost_ensemble.pkl`, `shap_explainer.pkl`

7. **Notebook 07 — Temporal Transformer** (`transformer/`)
   - Sequential attention-based transaction patterns modeling
   - Outputs: `temporal_transformer.pt`

8. **Notebook 08 — Model Registry Validation** (`validation/`)
   - End-to-end inference verification on test set
   - Outputs: `model_card.json`

---

## After Training

Place the output files in the `model_registry/` path expected by `backend/services/ml/model_loader.py`:

```
backend/services/ml/model_registry/
├── preprocessing/
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── label_encoders.pkl
│   └── feature_columns.pkl
├── dnn/
│   └── fraud_classifier.pt
├── anomaly/
│   └── isolation_forest.pkl
├── siamese/
│   └── siamese_network.pt
├── transformer/
│   └── temporal_transformer.pt
├── graphsage/
│   └── graphsage_model.pt        
├── ensemble/
│   ├── xgboost_ensemble.pkl
│   └── shap_explainer.pkl
└── model_card.json
```
