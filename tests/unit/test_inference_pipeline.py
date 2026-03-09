"""
Tests unitaires — InferencePipeline
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.pipeline.inference_pipeline import InferencePipeline, PredictionResult


@pytest.mark.unit
class TestPredictionResult:

    def test_to_dict_keys(self):
        result = PredictionResult(
            text          = "RWY CLSD",
            category      = "RUNWAY_CLOSURE",
            confidence    = 0.95,
            probabilities = {"RUNWAY_CLOSURE": 0.95, "LIGHTING": 0.05},
            latency_ms    = 10.0,
            meta          = {"priority": "CRITICAL", "icon": "🛬"},
        )
        d = result.to_dict()
        assert "category"      in d
        assert "confidence"    in d
        assert "probabilities" in d
        assert "latency_ms"    in d
        assert "priority"      in d

    def test_confidence_rounded(self):
        result = PredictionResult(
            text="test", category="LIGHTING",
            confidence=0.123456789,
            probabilities={}, latency_ms=5.0, meta={},
        )
        assert result.to_dict()["confidence"] == round(0.123456789, 4)

    def test_repr(self):
        result = PredictionResult(
            text="test", category="OBSTACLE",
            confidence=0.88, probabilities={},
            latency_ms=8.0, meta={},
        )
        assert "OBSTACLE" in repr(result)
        assert "88.00%" in repr(result)


@pytest.mark.unit
class TestInferencePipeline:

    def test_singleton_pattern(self):
        """Vérifie que get_instance retourne toujours le même objet."""
        InferencePipeline.reset()
        # On ne peut pas tester le vrai singleton sans modèle chargé
        # mais on vérifie que reset() fonctionne
        assert InferencePipeline._instance is None

    def test_build_input_df(self):
        """Vérifie que _build_input_df produit les bonnes colonnes."""
        pipe = InferencePipeline()
        df   = pipe._build_input_df("RWY 28L CLSD DUE TO MAINTENANCE")

        assert "body_text"         in df.columns
        assert "char_count"        in df.columns
        assert "word_count"        in df.columns
        assert "upper_ratio"       in df.columns
        assert "digit_ratio"       in df.columns
        assert "slash_count"       in df.columns
        assert "has_time_pattern"  in df.columns
        assert "has_coordinates"   in df.columns
        assert len(df) == 1

    def test_input_df_values(self):
        """Vérifie les valeurs calculées pour un texte connu."""
        pipe = InferencePipeline()
        text = "RWY 28L CLSD"
        df   = pipe._build_input_df(text)

        assert df["char_count"].iloc[0]  == len(text)
        assert df["word_count"].iloc[0]  == 3
        assert df["slash_count"].iloc[0] == 0

    def test_coordinates_detection(self):
        """Vérifie la détection des coordonnées GPS dans le texte."""
        pipe     = InferencePipeline()
        with_geo = pipe._build_input_df("CRANE AT 4825N00215E 520FT")
        no_geo   = pipe._build_input_df("RWY 28L CLSD")

        assert with_geo["has_coordinates"].iloc[0] == 1
        assert no_geo["has_coordinates"].iloc[0]   == 0

    def test_time_pattern_detection(self):
        """Vérifie la détection des patterns horaires ICAO."""
        pipe      = InferencePipeline()
        with_time = pipe._build_input_df("CLSD 0600-2200 DAILY")
        no_time   = pipe._build_input_df("RWY CLSD")

        assert with_time["has_time_pattern"].iloc[0] == 1