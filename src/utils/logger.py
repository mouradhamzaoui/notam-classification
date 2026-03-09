"""
logger.py
Logger centralisé avec Rich pour un output professionnel en console
et rotation des fichiers de logs.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ── Thème Rich ────────────────────────────────────────────────────────────────
RICH_THEME = Theme({
    "info":     "cyan",
    "warning":  "yellow bold",
    "error":    "red bold",
    "critical": "red bold reverse",
    "success":  "green bold",
})

console = Console(theme=RICH_THEME)


def get_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Retourne un logger configuré avec :
    - Handler Rich (console colorisée)
    - Handler fichier rotatif (10MB max, 5 backups)

    Usage :
        logger = get_logger(__name__)
        logger.info("Pipeline started")
        logger.warning("Low confidence prediction: 0.52")
        logger.error("Model file not found")
    """
    logger = logging.getLogger(name)

    # Évite les handlers dupliqués en cas de re-import
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ── Rich console handler ──────────────────────────────────────────────────
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        rich_tracebacks=True,
        markup=True,
    )
    rich_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(rich_handler)

    # ── File handler rotatif ──────────────────────────────────────────────────
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        log_file = log_path / f"notam_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,   # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# ── Logger racine du projet ───────────────────────────────────────────────────
root_logger = get_logger("notam_classification")