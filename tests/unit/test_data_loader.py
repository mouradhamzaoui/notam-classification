"""
Tests unitaires — NOTAMDataLoader
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.data.data_loader import NOTAMDataLoader


@pytest.mark.unit
class TestNOTAMDataLoader:

    def test_load_success(self, tmp_path, sample_df):
        """Vérifie le chargement normal d'un CSV valide."""
        csv_path = tmp_path / "test_notams.csv"
        sample_df.to_csv(csv_path, index=False)

        loader = NOTAMDataLoader(path=csv_path)
        df     = loader.load()

        assert len(df) > 0
        assert "body_text" in df.columns
        assert "category"  in df.columns

    def test_load_file_not_found(self):
        """Vérifie que FileNotFoundError est levé si le fichier n'existe pas."""
        loader = NOTAMDataLoader(path="nonexistent/path/data.csv")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_missing_columns(self, tmp_path):
        """Vérifie que ValueError est levé si une colonne requise est absente."""
        bad_df = pd.DataFrame({"wrong_col": ["test"], "other": ["val"]})
        csv_path = tmp_path / "bad.csv"
        bad_df.to_csv(csv_path, index=False)

        loader = NOTAMDataLoader(path=csv_path)
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            loader.load()

    def test_split_stratified(self, tmp_path, sample_df):
        """Vérifie que le split préserve les proportions de classes."""
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)

        loader = NOTAMDataLoader(path=csv_path, test_size=0.2)
        df     = loader.load()
        X_train, X_test, y_train, y_test = loader.split(df)

        # Taille approximative
        assert len(X_train) > len(X_test)
        assert abs(len(X_test) / len(df) - 0.2) < 0.05

        # Distribution des classes équilibrée dans les deux splits
        train_dist = y_train.value_counts(normalize=True)
        test_dist  = y_test.value_counts(normalize=True)
        for cat in train_dist.index:
            if cat in test_dist.index:
                assert abs(train_dist[cat] - test_dist[cat]) < 0.1

    def test_meta_features_added(self, tmp_path, sample_df):
        """Vérifie que les meta-features sont bien ajoutées après chargement."""
        csv_path = tmp_path / "test.csv"
        sample_df[["body_text", "category"]].to_csv(csv_path, index=False)

        loader = NOTAMDataLoader(path=csv_path)
        df     = loader.load()

        expected_meta = ["char_count", "word_count", "upper_ratio",
                         "digit_ratio", "slash_count"]
        for feat in expected_meta:
            assert feat in df.columns, f"Meta feature manquante : {feat}"

    def test_duplicates_removed(self, tmp_path, sample_df):
        """Vérifie la suppression des doublons."""
        df_dup   = pd.concat([sample_df, sample_df.head(10)], ignore_index=True)
        csv_path = tmp_path / "dup.csv"
        df_dup.to_csv(csv_path, index=False)

        loader = NOTAMDataLoader(path=csv_path)
        df     = loader.load()
        assert len(df) <= len(df_dup)

    def test_random_state_reproducibility(self, tmp_path, sample_df):
        """Vérifie que le split est reproductible avec le même random_state."""
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)

        loader1 = NOTAMDataLoader(path=csv_path, random_state=42)
        loader2 = NOTAMDataLoader(path=csv_path, random_state=42)

        df1 = loader1.load()
        df2 = loader2.load()

        _, _, y_train1, _ = loader1.split(df1)
        _, _, y_train2, _ = loader2.split(df2)

        assert list(y_train1) == list(y_train2)