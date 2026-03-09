"""
routers/health.py
Endpoints de santé et informations sur le modèle.
"""

import time
from datetime import datetime

from fastapi import APIRouter, Depends

from src.api.dependencies import get_config, get_db, get_inference_pipeline
from src.api.schemas import HealthResponse, ModelInfoResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])
START_TIME = time.time()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health Check",
    description="Vérifie l'état de l'API, du modèle et de la base de données.",
)
async def health_check(
    cfg=Depends(get_config),
    pipeline=Depends(get_inference_pipeline),
    db=Depends(get_db),
):
    model_loaded = False
    try:
        _ = pipeline._model
        model_loaded = pipeline._loaded
    except Exception:
        pass

    db_connected = False
    try:
        db._engine.connect()
        db_connected = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=cfg.project.version,
        model_loaded=model_loaded,
        db_connected=db_connected,
        uptime_s=round(time.time() - START_TIME, 1),
    )


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Retourne les informations sur le modèle déployé.",
)
async def model_info(
    cfg=Depends(get_config),
    pipeline=Depends(get_inference_pipeline),
    db=Depends(get_db),
):
    classes = list(pipeline._feature_pipeline.label_encoder.classes_)
    n_features = pipeline._feature_pipeline.n_features

    # Récupère les métriques du dernier run en base
    f1, acc, trained_at = None, None, None
    try:
        best = db.get_best_run()
        if best:
            f1 = best.f1_macro
            acc = best.accuracy
            trained_at = best.created_at.isoformat() if best.created_at else None
    except Exception:
        pass

    return ModelInfoResponse(
        model_name=cfg.model.name or "LinearSVC",
        version="latest",
        n_features=n_features,
        classes=classes,
        f1_macro=f1,
        accuracy=acc,
        trained_at=trained_at,
    )
