import json
import pickle
from pathlib import Path
from typing import Any

from backend.core.config import get_settings
from backend.core.logger import setup_logger

logger = setup_logger()


class ModelRegistry:
    _instance: "ModelRegistry | None" = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_path = Path(self.settings.model_registry_path)
        self.models: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self._load_all()

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(self, key: str) -> Any:
        return self.models.get(key)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def _load_all(self) -> None:
        self._load_pickle("scaler", "preprocessing/scaler.pkl", optional=True)
        self._load_pickle("pca", "preprocessing/pca.pkl", optional=True)
        self._load_pickle("label_encoders", "preprocessing/label_encoders.pkl", optional=True)
        self._load_pickle("feature_columns", "preprocessing/feature_columns.pkl", optional=True)
        self._load_pickle("isolation_forest", "anomaly/isolation_forest.pkl", optional=True)
        self._load_pickle("xgboost_ensemble", "ensemble/xgboost_ensemble.pkl", optional=True)
        self._load_pickle("shap_explainer", "ensemble/shap_explainer.pkl", optional=True)
        self._load_torch("dnn", "dnn/fraud_classifier.pt", optional=True)
        self._load_torch("siamese", "siamese/siamese_network.pt", optional=True)
        self._load_torch("transformer", "transformer/temporal_transformer.pt", optional=True)
        self._load_torch("graphsage", "graphsage/graphsage_model.pt", optional=True)
        self._load_metadata()

    def _load_pickle(self, key: str, relative_path: str, optional: bool = False) -> None:
        path = self.base_path / relative_path
        try:
            with path.open("rb") as file:
                self.models[key] = pickle.load(file)
            logger.info("Loaded model artifact: {}", key)
        except FileNotFoundError:
            if not optional:
                logger.warning("Model artifact not found: {}", path)
            self.models[key] = None
        except Exception as exc:
            logger.warning("Could not load model artifact {}: {}", path, exc)
            self.models[key] = None

    def _load_torch(self, key: str, relative_path: str, optional: bool = False) -> None:
        path = self.base_path / relative_path
        try:
            import torch
        except ImportError:
            self.models[key] = None
            return

        try:
            model = torch.load(path, map_location="cpu", weights_only=False)
            if hasattr(model, "eval"):
                model.eval()
            self.models[key] = model
            logger.info("Loaded PyTorch model: {}", key)
        except FileNotFoundError:
            if not optional:
                logger.warning("PyTorch model not found: {}", path)
            self.models[key] = None
        except Exception as exc:
            logger.warning("Could not load PyTorch model {}: {}", path, exc)
            self.models[key] = None

    def _load_metadata(self) -> None:
        for path in self.base_path.glob("*/metadata.json"):
            try:
                self.metadata[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not load metadata {}: {}", path, exc)
