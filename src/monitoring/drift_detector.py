"""
drift_detector.py
Détection de dérive des données et du modèle avec Evidently AI.

Deux types de dérive surveillés :
  1. Data Drift   : le vocabulaire/distribution des NOTAMs entrants change
  2. Model Drift  : la distribution des prédictions change (concept drift)

Métriques utilisées :
  - PSI  (Population Stability Index) : mesure le shift de distribution
    PSI < 0.10 → stable
    PSI 0.10-0.20 → léger changement
    PSI > 0.20 → dérive significative → ALERTE

  - KS Test (Kolmogorov-Smirnov) : test statistique de similarité
    p-value < 0.05 → distributions significativement différentes

  - Chi² Test : pour les features catégorielles
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
    ColumnDistributionMetric,
)
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
)

from src.utils.config import Config
from src.utils.logger import get_logger

logger  = get_logger(__name__)
REPORTS = Path("reports/drift")
REPORTS.mkdir(parents=True, exist_ok=True)


class NOTAMDriftDetector:
    """
    Détecteur de dérive pour le système NOTAM.

    Principe de fonctionnement :
        1. Référence (reference) : dataset d'entraînement → distribution de référence
        2. Courant   (current)   : prédictions récentes en production
        3. Comparaison : Evidently compare les deux distributions

    Features surveillées :
        - char_count      : longueur des NOTAMs
        - word_count      : nombre de mots
        - upper_ratio     : ratio majuscules
        - digit_ratio     : ratio chiffres
        - has_coordinates : présence de coordonnées GPS
        - prediction      : distribution des catégories prédites

    Usage :
        detector = NOTAMDriftDetector()
        detector.load_reference(df_train)
        report = detector.run(df_production)
        alert  = detector.check_alert(report)
    """

    NUMERICAL_FEATURES = [
        "char_count", "word_count",
        "upper_ratio", "digit_ratio",
        "slash_count", "has_coordinates",
    ]
    CATEGORICAL_FEATURES = ["predicted_category"]
    TARGET_COL           = "predicted_category"

    def __init__(self, cfg: Config | None = None):
        self.cfg        = cfg or Config.get()
        self._reference: pd.DataFrame | None = None
        self._psi_threshold = float(
            self.cfg.thresholds.drift_psi_alert or 0.20
        )

    # ── Reference data ────────────────────────────────────────────────────────

    def load_reference(self, df: pd.DataFrame) -> NOTAMDriftDetector:
        """Définit le dataset de référence (training set)."""
        self._reference = self._prepare(df)
        logger.info(
            f"[Drift] Reference loaded → {len(self._reference)} rows, "
            f"features: {self.NUMERICAL_FEATURES}"
        )
        return self

    def load_reference_from_csv(
        self, path: str = "data/processed/notams_clean.csv"
    ) -> NOTAMDriftDetector:
        """Charge la référence depuis le CSV de training."""
        df = pd.read_csv(path)
        # Simule une colonne de prédiction sur le training set
        df["predicted_category"] = df["category"]
        return self.load_reference(df)

    # ── Current data ──────────────────────────────────────────────────────────

    def build_current_from_db(self, limit: int = 500) -> pd.DataFrame | None:
        """
        Construit le dataset courant depuis les logs de prédiction en base.
        """
        try:
            from src.tracking.database import DatabaseManager
            db   = DatabaseManager.get_instance()
            logs = db.get_recent_predictions(limit=limit)

            if not logs:
                logger.warning("[Drift] No predictions in DB yet")
                return None

            rows = []
            for log in logs:
                rows.append({
                    "char_count":         log.char_count or 0,
                    "word_count":         log.word_count or 0,
                    "upper_ratio":        log.upper_ratio or 0.0,
                    "digit_ratio":        log.digit_ratio or 0.0,
                    "slash_count":        0,
                    "has_coordinates":    0,
                    "predicted_category": log.predicted,
                    "confidence":         log.confidence,
                })

            df = pd.DataFrame(rows)
            logger.info(f"[Drift] Current data → {len(df)} predictions from DB")
            return df

        except Exception as e:
            logger.error(f"[Drift] Failed to build current data: {e}")
            return None

    # ── Reports ───────────────────────────────────────────────────────────────

    def run_data_drift_report(
        self, current: pd.DataFrame, save: bool = True
    ) -> dict:
        """
        Génère un rapport de dérive des données complet.
        Retourne un dict avec les métriques clés.
        """
        if self._reference is None:
            raise RuntimeError("Call load_reference() first")

        ref = self._prepare(self._reference)
        cur = self._prepare(current)

        column_mapping = ColumnMapping(
            target          = self.TARGET_COL,
            numerical_features  = self.NUMERICAL_FEATURES,
            categorical_features= self.CATEGORICAL_FEATURES,
        )

        # ── Rapport complet ───────────────────────────────────────────────────
        report = Report(metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ] + [
            ColumnDriftMetric(column_name=col)
            for col in self.NUMERICAL_FEATURES
        ])

        report.run(
            reference_data = ref,
            current_data   = cur,
            column_mapping = column_mapping,
        )

        # ── Sauvegarde HTML ───────────────────────────────────────────────────
        if save:
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = REPORTS / f"drift_report_{ts}.html"
            report.save_html(str(html_path))
            logger.info(f"[Drift] Report saved → {html_path}")

        # ── Extraction des métriques clés ────────────────────────────────────
        result     = report.as_dict()
        metrics    = self._extract_metrics(result)
        logger.info(f"[Drift] Metrics: {metrics}")
        return metrics

    def run_test_suite(
        self, current: pd.DataFrame, save: bool = True
    ) -> dict:
        """
        Exécute une suite de tests de dérive avec pass/fail.
        """
        if self._reference is None:
            raise RuntimeError("Call load_reference() first")

        ref = self._prepare(self._reference)
        cur = self._prepare(current)

        column_mapping = ColumnMapping(
            numerical_features   = self.NUMERICAL_FEATURES,
            categorical_features = self.CATEGORICAL_FEATURES,
        )

        suite = TestSuite(tests=[
            DataDriftTestPreset(),
            TestNumberOfDriftedColumns(lte=2),
            TestShareOfDriftedColumns(lte=0.3),
        ])

        suite.run(
            reference_data = ref,
            current_data   = cur,
            column_mapping = column_mapping,
        )

        if save:
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = REPORTS / f"test_suite_{ts}.html"
            suite.save_html(str(html_path))
            logger.info(f"[Drift] Test suite saved → {html_path}")

        result    = suite.as_dict()
        passed    = sum(1 for t in result.get("tests", []) if t.get("status") == "SUCCESS")
        failed    = sum(1 for t in result.get("tests", []) if t.get("status") == "FAIL")
        total     = len(result.get("tests", []))

        summary = {
            "passed":     passed,
            "failed":     failed,
            "total":      total,
            "pass_rate":  round(passed / total, 2) if total > 0 else 0,
            "alert":      failed > 0,
        }
        logger.info(f"[Drift] Test suite: {passed}/{total} passed")
        return summary

    def check_alert(self, metrics: dict) -> bool:
        """
        Retourne True si une alerte de dérive doit être déclenchée.
        Critères :
          - dataset_drift détecté
          - OU share de colonnes driftées > 30%
        """
        alert = (
            metrics.get("dataset_drift", False)
            or metrics.get("drift_share", 0) > 0.30
        )
        if alert:
            logger.warning(
                f"[Drift] ⚠️  ALERT — Data drift detected! "
                f"drift_share={metrics.get('drift_share', 0):.0%}"
            )
        else:
            logger.info("[Drift] ✅ No significant drift detected")
        return alert

    def generate_synthetic_production_data(
        self, n: int = 200, drift: bool = False
    ) -> pd.DataFrame:
        """
        Génère des données de production simulées pour le monitoring.

        Args:
            n     : nombre de prédictions à simuler
            drift : si True, introduit une dérive artificielle
                    (simule un changement de distribution)
        """
        import random
        random.seed(42)

        categories = [
            "RUNWAY_CLOSURE", "NAVIGATION_AID", "AIRSPACE_RESTRICTION",
            "LIGHTING", "OBSTACLE", "AERODROME_PROCEDURE",
        ]

        if drift:
            # Dérive : surreprésentation de RUNWAY_CLOSURE et OBSTACLE
            weights = [0.40, 0.05, 0.05, 0.05, 0.40, 0.05]
            logger.warning("[Drift] Generating DRIFTED production data")
        else:
            # Distribution normale équilibrée
            weights = [1/6] * 6

        rows = []
        for _ in range(n):
            cat = random.choices(categories, weights=weights)[0]
            rows.append({
                "char_count":         random.randint(15, 80) + (50 if drift else 0),
                "word_count":         random.randint(3, 15),
                "upper_ratio":        random.uniform(0.7, 1.0),
                "digit_ratio":        random.uniform(0.0, 0.3),
                "slash_count":        random.randint(0, 3),
                "has_coordinates":    int(cat == "OBSTACLE"),
                "predicted_category": cat,
                "confidence":         random.uniform(0.6, 0.99),
            })
        return pd.DataFrame(rows)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def load_reference_from_csv(
        self, path: str = "data/processed/notams_clean.csv"
    ) -> NOTAMDriftDetector:
        """Charge la référence depuis le CSV de training."""
        df = pd.read_csv(path)
        df["predicted_category"] = df["category"]
        # Recalcule les meta-features si absentes
        t = df["body_text"].astype(str)
        if "upper_ratio" not in df.columns:
            df["upper_ratio"] = t.apply(
                lambda s: sum(c.isupper() for c in s) / max(len(s), 1)
            )
        if "digit_ratio" not in df.columns:
            df["digit_ratio"] = t.apply(
                lambda s: sum(c.isdigit() for c in s) / max(len(s), 1)
            )
        if "slash_count" not in df.columns:
            df["slash_count"] = t.str.count("/")
        if "has_coordinates" not in df.columns:
            import re
            df["has_coordinates"] = t.apply(
                lambda s: int(bool(re.search(r"\d{4}[NS]\d{5}[EW]", s)))
            )
        return self.load_reference(df)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare un DataFrame pour Evidently.
        Garantit que TOUTES les colonnes requises sont présentes dans les deux sets.
        """
        cols = self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES
        result = df.copy()

        # Ajoute les colonnes manquantes avec valeur 0
        for col in cols:
            if col not in result.columns:
                result[col] = 0

        result = result[cols].copy()

        # Conversion numérique stricte
        for col in self.NUMERICAL_FEATURES:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)

        # Conversion catégorielle
        for col in self.CATEGORICAL_FEATURES:
            result[col] = result[col].astype(str).fillna("UNKNOWN")

        return result.reset_index(drop=True)

    def _extract_metrics(self, result: dict) -> dict:
        """Extrait les métriques clés du rapport Evidently."""
        metrics = {
            "dataset_drift":   False,
            "drift_share":     0.0,
            "n_drifted":       0,
            "n_features":      len(self.NUMERICAL_FEATURES),
            "timestamp":       datetime.utcnow().isoformat(),
        }
        try:
            for metric in result.get("metrics", []):
                if metric.get("metric") == "DatasetDriftMetric":
                    r = metric.get("result", {})
                    metrics["dataset_drift"] = r.get("dataset_drift", False)
                    metrics["drift_share"]   = r.get("share_of_drifted_columns", 0.0)
                    metrics["n_drifted"]     = r.get("number_of_drifted_columns", 0)
        except Exception as e:
            logger.warning(f"[Drift] Could not extract metrics: {e}")
        return metrics