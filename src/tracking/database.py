"""
database.py
Gestion de la connexion PostgreSQL et des modèles ORM SQLAlchemy.

Tables :
  - experiments  : métadonnées des runs MLflow
  - predictions  : log de chaque prédiction en production
  - model_metrics: métriques historiques par version de modèle
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)
Base = declarative_base()


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLES ORM
# ══════════════════════════════════════════════════════════════════════════════


class ExperimentRun(Base):
    """
    Enregistre chaque run d'entraînement MLflow.
    Permet la traçabilité complète exigée en contexte aéronautique.
    """

    __tablename__ = "experiment_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), unique=True, nullable=False, index=True)
    experiment_name = Column(String(128), nullable=False)
    model_name = Column(String(128), nullable=False)
    status = Column(String(32), default="RUNNING")

    # Hyperparamètres (JSON sérialisé)
    params = Column(Text, default="{}")

    # Métriques
    f1_macro = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    cv_f1_mean = Column(Float, nullable=True)
    cv_f1_std = Column(Float, nullable=True)
    train_time_s = Column(Float, nullable=True)

    # Métadonnées
    dataset_size = Column(Integer, nullable=True)
    n_features = Column(Integer, nullable=True)
    is_best = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return (
            f"<ExperimentRun run_id={self.run_id[:8]}... "
            f"model={self.model_name} f1={self.f1_macro:.4f}>"
        )


class PredictionLog(Base):
    """
    Log de chaque prédiction effectuée via l'API.
    Utilisé pour le monitoring de la dérive des données (Evidently).
    """

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    input_text = Column(Text, nullable=False)
    predicted = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=True)
    model_version = Column(String(32), default="latest")

    # Features extraites (pour le drift monitoring)
    char_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    upper_ratio = Column(Float, nullable=True)
    digit_ratio = Column(Float, nullable=True)

    # Feedback utilisateur (optionnel)
    true_label = Column(String(64), nullable=True)
    is_correct = Column(Boolean, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_pred_logs_predicted", "predicted"),
        Index("ix_pred_logs_created", "created_at"),
    )

    def __repr__(self):
        return f"<PredictionLog predicted={self.predicted} conf={self.confidence:.2f}>"


class ModelMetric(Base):
    """
    Historique des métriques par version de modèle.
    Permet de comparer les performances au fil du temps.
    """

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(32), nullable=False)
    metric_name = Column(String(64), nullable=False)
    metric_value = Column(Float, nullable=False)
    category = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_model_metrics_version", "model_version"),)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ══════════════════════════════════════════════════════════════════════════════


class DatabaseManager:
    """
    Gestionnaire de connexion à la base de données.

    Supporte deux modes :
      - PostgreSQL (production) : via DATABASE_URL ou config.yaml
      - SQLite    (fallback/dev): si PostgreSQL est inaccessible

    Usage :
        db = DatabaseManager.get_instance()
        with db.session() as session:
            session.add(PredictionLog(...))
    """

    _instance: DatabaseManager | None = None

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.get()
        self._engine = None
        self._Session = None

    @classmethod
    def get_instance(cls) -> DatabaseManager:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._connect()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    # ── Connexion ─────────────────────────────────────────────────────────────

    def _connect(self):
        """Tente PostgreSQL, fallback sur SQLite si indisponible."""
        pg_url = self.cfg.get_db_url()

        try:
            engine = create_engine(
                pg_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # vérifie la connexion avant usage
                echo=False,
            )
            # Test de connexion
            with engine.connect() as conn:
                conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            self._engine = engine
            self._Session = sessionmaker(bind=engine)
            logger.info(
                f"[DB] ✅ Connected to PostgreSQL → {self.cfg.database.host}:{self.cfg.database.port}/{self.cfg.database.name}"
            )

        except Exception as e:
            logger.warning(f"[DB] PostgreSQL unavailable ({e}), falling back to SQLite")
            sqlite_path = Path("data/processed/notam_dev.db")
            sqlite_path.parent.mkdir(exist_ok=True)
            self._engine = create_engine(
                f"sqlite:///{sqlite_path}",
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
            self._Session = sessionmaker(bind=self._engine)
            logger.info(f"[DB] ✅ Connected to SQLite → {sqlite_path}")

        # Crée les tables si inexistantes
        Base.metadata.create_all(self._engine)
        logger.info("[DB] Tables initialized")

    # ── Session context manager ───────────────────────────────────────────────

    def session(self) -> Session:
        """Retourne une session SQLAlchemy (à utiliser avec `with`)."""
        return self._Session()

    # ── Helpers CRUD ──────────────────────────────────────────────────────────

    def log_prediction(self, pred_result, model_version: str = "latest"):
        """Persiste une prédiction dans prediction_logs."""
        import re

        text = pred_result.text
        log = PredictionLog(
            input_text=text[:500],
            predicted=pred_result.category,
            confidence=pred_result.confidence,
            latency_ms=pred_result.latency_ms,
            model_version=model_version,
            char_count=len(text),
            word_count=len(text.split()),
            upper_ratio=sum(c.isupper() for c in text) / max(len(text), 1),
            digit_ratio=sum(c.isdigit() for c in text) / max(len(text), 1),
        )
        try:
            with self.session() as s:
                s.add(log)
                s.commit()
        except Exception as e:
            logger.error(f"[DB] Failed to log prediction: {e}")

    def save_experiment_run(self, run_id: str, artifacts) -> ExperimentRun:
        """Persiste les métadonnées d'un run d'entraînement."""
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=self.cfg.mlflow.experiment_name,
            model_name=artifacts.model_name,
            status="FINISHED",
            params=json.dumps(artifacts.params, default=str),
            f1_macro=artifacts.metrics.get("test_f1_macro"),
            accuracy=artifacts.metrics.get("test_accuracy"),
            cv_f1_mean=artifacts.metrics.get("cv_f1_mean"),
            cv_f1_std=artifacts.metrics.get("cv_f1_std"),
            train_time_s=artifacts.train_time_s,
        )
        try:
            with self.session() as s:
                # Marque les anciens runs comme non-best
                s.query(ExperimentRun).update({"is_best": False})
                run.is_best = True
                s.add(run)
                s.commit()
                s.refresh(run)
                logger.info(f"[DB] Experiment run saved → id={run.id}")
        except Exception as e:
            logger.error(f"[DB] Failed to save experiment run: {e}")
        return run

    def get_recent_predictions(self, limit: int = 100):
        """Récupère les N dernières prédictions pour le monitoring."""
        with self.session() as s:
            return (
                s.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(limit).all()
            )

    def get_best_run(self) -> ExperimentRun | None:
        """Retourne le meilleur run d'entraînement."""
        with self.session() as s:
            return (
                s.query(ExperimentRun)
                .filter(ExperimentRun.is_best)
                .order_by(ExperimentRun.f1_macro.desc())
                .first()
            )
