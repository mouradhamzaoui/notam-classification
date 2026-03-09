"""
dependencies.py
Dépendances FastAPI injectables (Dependency Injection pattern).
"""

from __future__ import annotations
from functools import lru_cache

from src.utils.config import Config
from src.utils.logger import get_logger
from src.pipeline.inference_pipeline import InferencePipeline
from src.tracking.database import DatabaseManager

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config.get()


def get_inference_pipeline() -> InferencePipeline:
    """Retourne le singleton InferencePipeline (modèle chargé une fois)."""
    return InferencePipeline.get_instance()


def get_db() -> DatabaseManager:
    """Retourne le singleton DatabaseManager."""
    return DatabaseManager.get_instance()