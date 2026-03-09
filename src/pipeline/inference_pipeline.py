"""
inference_pipeline.py
Pipeline d'inférence Production-Ready.
Charge le modèle une seule fois (singleton) et expose predict().
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionResult:
    """Résultat structuré d'une prédiction."""

    def __init__(
        self,
        text: str,
        category: str,
        confidence: float,
        probabilities: dict,
        latency_ms: float,
        meta: dict,
    ):
        self.text = text
        self.category = category
        self.confidence = confidence
        self.probabilities = probabilities
        self.latency_ms = latency_ms
        self.meta = meta

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "latency_ms": round(self.latency_ms, 2),
            "priority": self.meta.get("priority", "UNKNOWN"),
            "icon": self.meta.get("icon", "✈️"),
        }

    def __repr__(self) -> str:
        return (
            f"<PredictionResult category={self.category} "
            f"confidence={self.confidence:.2%} latency={self.latency_ms:.1f}ms>"
        )


class InferencePipeline:
    """
    Pipeline d'inférence singleton.

    Pattern Singleton : le modèle est chargé une seule fois en mémoire
    pour éviter la latence de désérialisation à chaque requête API.

    Usage :
        pipe = InferencePipeline.get_instance()
        result = pipe.predict("RWY 28L CLSD DUE TO MAINTENANCE")
        print(result.category, result.confidence)
    """

    _instance = None

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.get()
        self._feature_pipeline = None
        self._model = None
        self._categories_meta = {}
        self._loaded = False

    @classmethod
    def get_instance(cls) -> InferencePipeline:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> InferencePipeline:
        """Charge le pipeline de features + le modèle depuis le disque."""
        if self._loaded:
            return self

        fp_path = Path("data/processed/feature_pipeline.pkl")
        model_path = Path(self.cfg.model.artifacts_dir) / "best_model.pkl"

        for path in [fp_path, model_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Artifact manquant : {path}\nLance d'abord : uv run python scripts/train.py"
                )

        logger.info("[Inference] Loading artifacts...")
        self._feature_pipeline = joblib.load(fp_path)
        self._model = joblib.load(model_path)
        self._categories_meta = self.cfg.categories or {}
        self._loaded = True
        logger.info(
            f"[Inference] ✅ Ready | Classes: {list(self._feature_pipeline.label_encoder.classes_)}"
        )
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, text: str) -> PredictionResult:
        """Prédit la catégorie d'un NOTAM texte brut."""
        self._ensure_loaded()
        t0 = time.time()

        df_input = self._build_input_df(text)
        X = self._feature_pipeline.transform(df_input)
        pred_enc = self._model.predict(X)[0]
        proba = self._model.predict_proba(X)[0]
        classes = self._feature_pipeline.label_encoder.classes_

        category = self._feature_pipeline.decode_labels(np.array([pred_enc]))[0]
        confidence = float(proba[pred_enc])
        latency_ms = (time.time() - t0) * 1000

        result = PredictionResult(
            text=text,
            category=category,
            confidence=confidence,
            probabilities={c: float(p) for c, p in zip(classes, proba, strict=False)},
            latency_ms=latency_ms,
            meta=self._categories_meta.get(category, {}),
        )
        logger.debug(f"[Predict] {result}")
        return result

    def predict_batch(self, texts: list) -> list:
        """Prédit une liste de NOTAMs."""
        return [self.predict(t) for t in texts]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_input_df(self, text: str) -> pd.DataFrame:
        t = str(text)
        return pd.DataFrame(
            [
                {
                    "body_text": t,
                    "char_count": len(t),
                    "word_count": len(t.split()),
                    "upper_ratio": sum(c.isupper() for c in t) / max(len(t), 1),
                    "digit_ratio": sum(c.isdigit() for c in t) / max(len(t), 1),
                    "slash_count": t.count("/"),
                    "has_time_pattern": int(bool(re.search(r"\b\d{4}Z?\b", t))),
                    "has_coordinates": int(bool(re.search(r"\d{4}[NS]\d{5}[EW]", t))),
                }
            ]
        )

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()
