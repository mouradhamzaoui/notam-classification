"""
config.py
Chargement et validation centralisés de la configuration YAML.
Pattern Singleton pour éviter les relectures multiples du fichier.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Charge les variables d'environnement depuis .env si présent
load_dotenv()

# ── Chemins par défaut ────────────────────────────────────────────────────────
CONFIG_DIR       = Path(__file__).parent.parent.parent / "config"
MAIN_CONFIG_PATH = CONFIG_DIR / "config.yaml"
MODEL_CONFIG_PATH= CONFIG_DIR / "model_config.yaml"


class Config:
    """
    Singleton de configuration.
    Fusionne config.yaml + model_config.yaml + variables d'environnement.

    Priorité : ENV vars > config.yaml > valeurs par défaut

    Usage :
        from src.utils.config import Config
        cfg = Config.get()
        print(cfg.data.text_col)          # "body_text"
        print(cfg.mlflow.tracking_uri)    # "http://localhost:5000"
    """

    _instance: Config | None = None
    _raw:      dict           = {}

    def __new__(cls) -> Config:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    @classmethod
    def get(cls) -> Config:
        return cls()

    @classmethod
    def reset(cls):
        """Force le rechargement (utile pour les tests)."""
        cls._instance = None

    # ── Chargement ────────────────────────────────────────────────────────────

    def _load(self):
        main  = self._read_yaml(MAIN_CONFIG_PATH)
        model = self._read_yaml(MODEL_CONFIG_PATH)
        self._raw = {**main, "model_config": model}
        self._override_from_env()
        logger.debug("Configuration loaded successfully")

    @staticmethod
    def _read_yaml(path: Path) -> dict:
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _override_from_env(self):
        """
        Surcharge les valeurs sensibles depuis les variables d'environnement.
        Ex: NOTAM_DB_PASSWORD → database.password
        """
        env_map = {
            "NOTAM_DB_HOST":     ("database", "host"),
            "NOTAM_DB_PORT":     ("database", "port"),
            "NOTAM_DB_NAME":     ("database", "name"),
            "NOTAM_DB_USER":     ("database", "user"),
            "NOTAM_DB_PASSWORD": ("database", "password"),
            "NOTAM_MLFLOW_URI":  ("mlflow",   "tracking_uri"),
        }
        for env_key, (section, key) in env_map.items():
            val = os.getenv(env_key)
            if val and section in self._raw:
                self._raw[section][key] = val

    # ── Accesseurs typés ──────────────────────────────────────────────────────

    @property
    def project(self)  -> _Section: return _Section(self._raw.get("project", {}))

    @property
    def data(self)     -> _Section: return _Section(self._raw.get("data", {}))

    @property
    def features(self) -> _Section: return _Section(self._raw.get("features", {}))

    @property
    def model(self)    -> _Section: return _Section(self._raw.get("model", {}))

    @property
    def mlflow(self)   -> _Section: return _Section(self._raw.get("mlflow", {}))

    @property
    def api(self)      -> _Section: return _Section(self._raw.get("api", {}))

    @property
    def logging(self)  -> _Section: return _Section(self._raw.get("logging", {}))

    @property
    def database(self) -> _Section: return _Section(self._raw.get("database", {}))

    @property
    def categories(self) -> dict:
        return self._raw.get("model_config", {}).get("categories", {})

    @property
    def thresholds(self) -> _Section:
        return _Section(self._raw.get("model_config", {}).get("thresholds", {}))

    @property
    def candidates(self) -> dict:
        return self._raw.get("model_config", {}).get("candidates", {})

    def get_db_url(self) -> str:
        db = self._raw.get("database", {})
        return (
            f"postgresql://{db.get('user','notam_user')}:"
            f"{db.get('password','notam_pass')}@"
            f"{db.get('host','localhost')}:"
            f"{db.get('port',5432)}/"
            f"{db.get('name','notam_db')}"
        )

    def __repr__(self) -> str:
        return f"<Config project='{self.project.name}' v{self.project.version}>"


class _Section:
    """Accès aux valeurs de config par attribut ou clé avec défaut sécurisé."""

    def __init__(self, data: dict):
        self._data = data or {}

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        return self._data.get(key)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __repr__(self) -> str:
        return f"<Section {self._data}>"