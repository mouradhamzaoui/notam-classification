"""
conftest.py
Fixtures partagées par tous les tests.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Assure que src/ est dans le path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES — DONNÉES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def sample_notams() -> list[dict]:
    """NOTAMs de test représentatifs de chaque catégorie."""
    return [
        {"text": "RWY 28L CLSD DUE TO CONSTRUCTION WIP", "category": "RUNWAY_CLOSURE"},
        {"text": "ILS CAT II RWY 10R NOT AVAILABLE", "category": "NAVIGATION_AID"},
        {
            "text": "RESTRICTED AREA R-2508 ACTIVE SFC-18000FT MSL",
            "category": "AIRSPACE_RESTRICTION",
        },
        {"text": "PAPI RWY 36 OTS", "category": "LIGHTING"},
        {"text": "NEW CRANE 520FT AGL WITHIN 3NM OF LFPG ARP", "category": "OBSTACLE"},
        {"text": "FUEL NOT AVBL 2H DAILY DUE MAINTENANCE", "category": "AERODROME_PROCEDURE"},
    ]


@pytest.fixture(scope="session")
def sample_df(sample_notams) -> pd.DataFrame:
    """DataFrame de test avec les colonnes requises."""
    rows = []
    for i, n in enumerate(sample_notams * 50):  # 300 lignes
        rows.append(
            {
                "notam_id": f"A{i:04d}/24",
                "body_text": f"{n['text']} REF{i:04d}",  # ← unique grâce au suffixe
                "category": n["category"],
                "icao_location": "LFPG",
                "q_code": "QMRLC",
                "char_count": len(n["text"]),
                "word_count": len(n["text"].split()),
                "upper_ratio": 0.9,
                "digit_ratio": 0.05,
                "slash_count": 0,
                "has_time_pattern": 0,
                "has_coordinates": 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def trained_pipeline(sample_df):
    """Pipeline de features entraîné sur les données de test."""
    from src.features.feature_engineering import NOTAMFeaturePipeline

    pipeline = NOTAMFeaturePipeline(max_tfidf_features=500)
    X = sample_df.drop(columns=["category"])
    y = sample_df["category"]
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture(scope="session")
def trained_model(trained_pipeline, sample_df):
    """Modèle LinearSVC entraîné sur les données de test."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    X = sample_df.drop(columns=["category"])
    y = sample_df["category"]
    X_feat = trained_pipeline.transform(X)
    y_enc = trained_pipeline.encode_labels(y)

    model = CalibratedClassifierCV(
        estimator=LinearSVC(C=1.0, max_iter=500, random_state=42),
        cv=3,
        method="sigmoid",
    )
    model.fit(X_feat, y_enc)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES — API
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def mock_inference_pipeline(trained_pipeline, trained_model):
    """Mock du InferencePipeline pour les tests API (évite de charger les fichiers)."""
    from src.pipeline.inference_pipeline import InferencePipeline, PredictionResult

    mock = MagicMock(spec=InferencePipeline)
    mock._loaded = True
    mock._feature_pipeline = trained_pipeline
    mock._model = trained_model

    def _predict(text: str) -> PredictionResult:
        import re
        import time

        t = str(text)
        df_input = pd.DataFrame(
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
        X = trained_pipeline.transform(df_input)
        pred_enc = trained_model.predict(X)[0]
        proba = trained_model.predict_proba(X)[0]
        classes = trained_pipeline.label_encoder.classes_
        category = trained_pipeline.decode_labels(np.array([pred_enc]))[0]

        return PredictionResult(
            text=text,
            category=category,
            confidence=float(proba[pred_enc]),
            probabilities={c: float(p) for c, p in zip(classes, proba, strict=False)},
            latency_ms=5.0,
            meta={"priority": "HIGH", "icon": "✈️"},
        )

    mock.predict.side_effect = _predict
    mock.predict_batch.side_effect = lambda texts: [_predict(t) for t in texts]
    return mock


@pytest.fixture(scope="session")
def mock_db():
    """Mock de DatabaseManager pour les tests API."""
    mock = MagicMock()
    mock.log_prediction.return_value = None
    mock.get_recent_predictions.return_value = []
    mock.get_best_run.return_value = None
    return mock


@pytest.fixture(scope="session")
def test_client(mock_inference_pipeline, mock_db):
    """Client de test FastAPI avec dépendances mockées."""
    from fastapi.testclient import TestClient

    from src.api.dependencies import get_db, get_inference_pipeline
    from src.api.main import app

    app.dependency_overrides[get_inference_pipeline] = lambda: mock_inference_pipeline
    app.dependency_overrides[get_db] = lambda: mock_db

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
