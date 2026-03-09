"""
predictor.py
Encapsule le chargement du modèle et l'inférence pour le POC Streamlit.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent.parent
PIPELINE_PATH = BASE_DIR / "data" / "processed" / "feature_pipeline.pkl"
MODEL_PATH    = BASE_DIR / "data" / "processed" / "models" / "best_model.pkl"

# ── Métadonnées des catégories ────────────────────────────────────────────────
CATEGORY_META = {
    "RUNWAY_CLOSURE": {
        "icon":        "🛬",
        "color":       "#ef4444",
        "priority":    "CRITICAL",
        "description": "Fermeture totale ou partielle d'une piste d'atterrissage ou de décollage.",
        "action":      "Vérifier les alternatives de piste disponibles. Recalculer les performances.",
    },
    "NAVIGATION_AID": {
        "icon":        "📡",
        "color":       "#3b82f6",
        "priority":    "HIGH",
        "description": "Panne ou dégradation d'un équipement de navigation (VOR, ILS, DME, NDB).",
        "action":      "Vérifier les procédures alternatives. Notifier l'équipage.",
    },
    "AIRSPACE_RESTRICTION": {
        "icon":        "🚫",
        "color":       "#f59e0b",
        "priority":    "CRITICAL",
        "description": "Zone aérienne restreinte, interdite ou avec procédures spéciales actives.",
        "action":      "Ajuster la route. Obtenir clairance si nécessaire.",
    },
    "LIGHTING": {
        "icon":        "💡",
        "color":       "#8b5cf6",
        "priority":    "MEDIUM",
        "description": "Dysfonctionnement des systèmes d'éclairage (PAPI, ALS, balisage de piste).",
        "action":      "Évaluer les minimums d'approche applicables.",
    },
    "OBSTACLE": {
        "icon":        "🏗️",
        "color":       "#10b981",
        "priority":    "HIGH",
        "description": "Nouvel obstacle (grue, tour, éolienne) dans la zone d'approche ou de départ.",
        "action":      "Vérifier les procédures de dégagement d'obstacle (ODP/SID).",
    },
    "AERODROME_PROCEDURE": {
        "icon":        "📋",
        "color":       "#ec4899",
        "priority":    "LOW",
        "description": "Changement de procédures opérationnelles de l'aérodrome (ATIS, carburant, douanes).",
        "action":      "Mettre à jour le briefing opérationnel.",
    },
}

PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


class NOTAMPredictor:
    """
    Charge le pipeline de features + le modèle entraîné,
    et expose une interface de prédiction simple.
    """

    def __init__(self):
        self._pipeline = None
        self._model    = None
        self._loaded   = False

    def load(self):
        if self._loaded:
            return self
        try:
            self._pipeline = joblib.load(PIPELINE_PATH)
            self._model    = joblib.load(MODEL_PATH)
            self._loaded   = True
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Modèle introuvable : {e}\n"
                "Lance d'abord : uv run python data/build_splits.py "
                "puis entraîne le modèle via le notebook 03_Modeling.ipynb"
            )
        return self

    def predict_one(self, text: str) -> dict:
        """
        Prédit la catégorie d'un NOTAM texte.

        Retourne :
            {
              "category":    str,
              "confidence":  float,
              "probabilities": {cat: float, ...},
              "meta":        dict (icône, couleur, priorité, action),
            }
        """
        self._ensure_loaded()
        df_input = pd.DataFrame({"body_text": [text]})
        df_input = self._pipeline.preprocessor.transform.__func__(
            self._pipeline.preprocessor, df_input
        ) if False else self._build_input_df(text)

        X = self._pipeline.transform(df_input)
        pred_enc  = self._model.predict(X)[0]
        proba     = self._model.predict_proba(X)[0]
        classes   = self._pipeline.label_encoder.classes_

        category  = self._pipeline.decode_labels(np.array([pred_enc]))[0]
        confidence= float(proba[pred_enc])

        return {
            "category":      category,
            "confidence":    confidence,
            "probabilities": {
                cls: float(p) for cls, p in zip(classes, proba)
            },
            "meta": CATEGORY_META.get(category, {}),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Prédit une liste de NOTAMs."""
        return [self.predict_one(t) for t in texts]

    def _build_input_df(self, text: str) -> pd.DataFrame:
        """Construit un DataFrame avec toutes les meta-features nécessaires."""
        t = str(text)
        import re
        return pd.DataFrame([{
            "body_text":        t,
            "char_count":       len(t),
            "word_count":       len(t.split()),
            "upper_ratio":      sum(c.isupper() for c in t) / max(len(t), 1),
            "digit_ratio":      sum(c.isdigit() for c in t) / max(len(t), 1),
            "slash_count":      t.count("/"),
            "has_time_pattern": int(bool(re.search(r"\b\d{4}Z?\b", t))),
            "has_coordinates":  int(bool(re.search(r"\d{4}[NS]\d{5}[EW]", t))),
        }])

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()


# ── Singleton ────────────────────────────────────────────────────────────────
predictor = NOTAMPredictor()