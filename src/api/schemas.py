"""
schemas.py
Schémas Pydantic pour la validation des requêtes et réponses API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ══════════════════════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════


class ClassifyRequest(BaseModel):
    """Requête de classification d'un NOTAM."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Texte brut du NOTAM à classifier",
        examples=["RWY 28L CLSD DUE TO CONSTRUCTION WIP"],
    )
    model_version: str | None = Field(
        default="latest",
        description="Version du modèle à utiliser",
    )

    @field_validator("text")
    @classmethod
    def clean_text(cls, v: str) -> str:
        return v.strip().upper()

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "RWY 28L CLSD DUE TO CONSTRUCTION WIP",
                "model_version": "latest",
            }
        }
    }


class BatchClassifyRequest(BaseModel):
    """Requête de classification par lot."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Liste de NOTAMs à classifier (max 100)",
    )
    model_version: str | None = Field(default="latest")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip().upper() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("Au moins un texte non vide est requis")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "RWY 10L CLSD DUE TO MAINTENANCE",
                    "ILS RWY 28R NOT AVAILABLE",
                    "RESTRICTED AREA R-4009 ACTIVE",
                ]
            }
        }
    }


class FeedbackRequest(BaseModel):
    """Feedback utilisateur sur une prédiction."""

    prediction_id: int = Field(..., description="ID de la prédiction")
    true_label: str = Field(..., description="Vraie catégorie selon l'expert")
    comment: str | None = Field(default=None, max_length=500)


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════


class CategoryMeta(BaseModel):
    priority: str
    icon: str
    color: str
    description: str | None = None
    action: str | None = None


class ClassifyResponse(BaseModel):
    """Réponse de classification d'un NOTAM."""

    category: str = Field(..., description="Catégorie ICAO prédite")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Score de confiance [0-1]")
    probabilities: dict[str, float] = Field(
        ..., description="Distribution de probabilité sur toutes les classes"
    )
    latency_ms: float = Field(..., description="Latence d'inférence en ms")
    priority: str = Field(..., description="Niveau de priorité opérationnel")
    icon: str = Field(..., description="Icône de la catégorie")
    model_version: str = Field(default="latest")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_schema_extra": {
            "example": {
                "category": "RUNWAY_CLOSURE",
                "confidence": 0.9823,
                "probabilities": {"RUNWAY_CLOSURE": 0.9823, "LIGHTING": 0.0102},
                "latency_ms": 12.4,
                "priority": "CRITICAL",
                "icon": "🛬",
                "model_version": "latest",
            }
        }
    }


class BatchClassifyResponse(BaseModel):
    """Réponse de classification par lot."""

    results: list[ClassifyResponse]
    total: int
    duration_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """État de santé de l'API."""

    status: str
    version: str
    model_loaded: bool
    db_connected: bool
    uptime_s: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfoResponse(BaseModel):
    """Informations sur le modèle déployé."""

    model_name: str
    version: str
    n_features: int
    classes: list[str]
    f1_macro: float | None = None
    accuracy: float | None = None
    trained_at: str | None = None


class PredictionLogResponse(BaseModel):
    """Log d'une prédiction depuis la base de données."""

    id: int
    predicted: str
    confidence: float
    latency_ms: float | None
    model_version: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée."""

    error: str
    detail: str | None = None
    code: int
